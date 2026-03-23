#!/usr/bin/env python3
"""
Prompt instance selection experiment.

Measure: relevance (mean cosine of final prompts vs user query embedding)
Vary: candidate prompts per class
Algorithms: SampledGreedySelector, IndividualPromptSelector (LLM), NaiveSelector

Run from repo root:
  python experiments/exp_prompt_instance_selection_relevance_candidates_per_class.py

Datasets (CSV under data/): diffusion_db, liar, race
  --dataset all          run all three (default)
  --dataset diffusion_db | liar | race   single dataset

Outputs per run:
  experiments/outputs/csv/prompt_instance_relevance_candidates_<dataset>_<timestamp>.csv
  experiments/outputs/figs/prompt_instance_relevance_candidates_<dataset>_<timestamp>.png

Progress: logs overall % (datasets × candidate grid × 3 algorithms) and sub-steps per selector.
  --quiet  disable progress logs
"""

# -----------------------------------------------------------------------------
# Key parameters (defaults match argparse below; override via CLI)
# -----------------------------------------------------------------------------
# --dataset              all              # diffusion_db | liar | race | all
# --n                    50               # rows loaded per CSV
# --separated            True
# --seed                 42
# --n_clusters_primary   3
# --n_clusters_secondary 5
# --phi                  2                # secondary classes per sequence
# --small_k              5
# --large_k              30
# --user_input           "Create a portrait of a famous person"
# --candidate_values     10,100,500,1000,2000   # pool size c per class (truncated embeddings_db)
# --promptselector_top_l 3               # min(top_l, c) for IndividualPromptSelector via _RagProxy
# --sampled_greedy_max_sample 25         # SampledGreedy: min(c, this) random draws per class (full RAG DB)
# --naive_batch_size          15         # NaiveSelector max candidates per batch
# --naive_mock_llm            True       # True = top-fraction cosine mock; False = LLM tournament
# --naive_top_fraction        0.15       # mock mode only
#
# Fixed in code (no CLI):
#   Clustering                     kmeans
#   Class sequences                support-weighted random walk (pool seed+c per c)
#   RAG initial top_l              5 (full embeddings_db before truncation)
#   Embedding model                text-embedding-3-small
#   SampledGreedySelector          full RAG DB per class; sample_size = min(c, sampled_greedy_max_sample)
#   IndividualPromptSelector     truncated pool c/class; top_l on proxy; LLM rerank
#   NaiveSelector                  uses every item in rag.embeddings_db for that class (here: truncated
#                                   pool = all up to c per class), batched tournament; no subsampling
# -----------------------------------------------------------------------------

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_model.load_data import DataLoader
from preprocessing.clusterer import Clustering
from preprocessing.embedding import Embedding
from llm.rag import RAG
from llm.llm_interface import LLMInterface
from algorithms.prompt_selector import (
    IndividualPromptSelector,
    SampledGreedySelector,
    NaiveSelector,
)

DATASET_FILES = {
    "diffusion_db": "diffusion_db.csv",
    "liar": "liar.csv",
    "race": "race.csv",
}

DATASET_DISPLAY = {
    "diffusion_db": "DiffusionDB",
    "liar": "LIAR",
    "race": "RACE",
}

ALL_DATASET_KEYS = tuple(DATASET_FILES.keys())


def _dataset_csv_path(key: str) -> str:
    return os.path.join(ROOT, "data", DATASET_FILES[key])


def _parse_dataset_arg(raw: str) -> list[str]:
    s = (raw or "").strip().lower()
    if s in ("", "all"):
        return list(ALL_DATASET_KEYS)
    if s in DATASET_FILES:
        return [s]
    raise SystemExit(
        f"Unknown --dataset {raw!r}. Use: all, {', '.join(ALL_DATASET_KEYS)}"
    )


def _most_frequent_primary_class(pm):
    from collections import Counter

    c = Counter()
    for cp in pm.composite_prompts:
        if cp.primary.class_obj:
            c[cp.primary.class_obj.index] += 1
    return c.most_common(1)[0][0] if c else 0


def _secondary_classes(pm):
    s = set()
    for cp in pm.composite_prompts:
        for sec in cp.secondaries:
            if sec.class_obj:
                s.add(sec.class_obj.index)
    return sorted(s)


def _sample_class_sequence_support(pm, primary_class, phi, rng):
    all_sec = _secondary_classes(pm)
    if not all_sec or phi < 1:
        return [primary_class]

    def p_to_s(p):
        w = []
        for sc in all_sec:
            sup = 0.1
            if pm.primary_to_secondary_support:
                sup = pm.primary_to_secondary_support.get((p, sc), 0.1)
            w.append(sup)
        w = np.array(w, dtype=float)
        w /= w.sum()
        return rng.choice(all_sec, p=w)

    def s_to_s(prev):
        w = []
        for sc in all_sec:
            sup = 0.1
            if pm.secondary_to_secondary_support:
                sup = pm.secondary_to_secondary_support.get((prev, sc), 0.1)
            w.append(sup)
        w = np.array(w, dtype=float)
        w /= w.sum()
        return rng.choice(all_sec, p=w)

    first = p_to_s(primary_class)
    secs = [first]
    for _ in range(phi - 1):
        secs.append(s_to_s(secs[-1]))
    return [primary_class] + secs


def _cosine(a, b):
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _embed_texts(client: OpenAI, texts):
    resp = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return [np.array(item.embedding, dtype=float) for item in resp.data]


class _RagProxy:
    """Lightweight stand-in for RAG (embeddings_db + top_l) without deepcopying httpx locks."""

    def __init__(self, embeddings_db, top_l):
        self.embeddings_db = embeddings_db
        self.top_l = top_l


class ExperimentProgress:
    """Overall % = completed phases / (n_datasets × n_candidate_grid × 3 algorithms)."""

    def __init__(self, dataset_keys: List[str], n_candidates: int):
        self.dataset_keys = list(dataset_keys)
        self.n_datasets = max(1, len(self.dataset_keys))
        self.n_candidates = max(1, int(n_candidates))
        self.total_phases = max(1, self.n_datasets * self.n_candidates * 3)

    def phase_start(self, dataset_key: str, c: int, ci: int, algo: str, algo_slot: int) -> None:
        di = self.dataset_keys.index(dataset_key)
        done = di * (self.n_candidates * 3) + ci * 3 + algo_slot
        pct = 100.0 * done / self.total_phases
        print(
            f"\n  [{pct:5.1f}%] {dataset_key} | c={c} ({ci+1}/{self.n_candidates}) | {algo}",
            flush=True,
        )

    def phase_done(self, relevance: float) -> None:
        print(f"         → relevance={relevance:.4f}", flush=True)


def run_one_dataset(
    dataset_key: str,
    args,
    rng: np.random.Generator,
    ts: str,
    progress: Optional[ExperimentProgress] = None,
):
    csv_path = _dataset_csv_path(dataset_key)
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Dataset file not found: {csv_path}")

    display_name = DATASET_DISPLAY[dataset_key]
    out_csv_dir = os.path.join(ROOT, "experiments", "outputs", "csv")
    out_fig_dir = os.path.join(ROOT, "experiments", "outputs", "figs")
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_fig_dir, exist_ok=True)
    base = f"prompt_instance_relevance_candidates_{dataset_key}_{ts}"
    csv_out = os.path.join(out_csv_dir, f"{base}.csv")
    fig_out = os.path.join(out_fig_dir, f"{base}.png")

    print(f"\n=== Dataset: {display_name} ({csv_path}) ===")
    print("Loading + clustering + support...")
    loader = DataLoader(separated=args.separated, n=args.n)
    pm = loader.load_data(csv_path)
    clusterer = Clustering(pm, algorithm="kmeans")
    pm = clusterer.cluster(
        {
            "primary": {"n_clusters": args.n_clusters_primary},
            "secondary": {"n_clusters": args.n_clusters_secondary},
        }
    )

    primary_class = _most_frequent_primary_class(pm)
    sequences = [
        _sample_class_sequence_support(pm, primary_class, args.phi, rng)
        for _ in range(args.large_k)
    ]
    pm.k_class_sequences = sequences[: args.small_k]

    embedding = Embedding(pm)
    embedding.embed()
    llm = LLMInterface()
    rag = RAG(embedding, llm, top_l=5)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is required for embeddings and selectors.")
    client = OpenAI(api_key=api_key)
    user_emb = _embed_texts(client, [args.user_input])[0]

    candidate_values = [int(x.strip()) for x in args.candidate_values.split(",") if x.strip()]
    candidate_values = sorted(set(candidate_values))

    rows = []
    for ci, c in enumerate(candidate_values):
        rng_pool = np.random.default_rng(args.seed + c)

        by_class = {}
        for item in (rag.embeddings_db or []):
            by_class.setdefault(item["class_label"], []).append(item)

        truncated = []
        for class_label, items in by_class.items():
            if len(items) <= c:
                truncated.extend(items)
            else:
                idx = rng_pool.choice(len(items), size=c, replace=False)
                truncated.extend([items[i] for i in idx])

        top_l_ps = min(args.promptselector_top_l, c)
        rag_ps = _RagProxy(truncated, top_l=top_l_ps)
        rag_naive = _RagProxy(truncated, top_l=1)

        greedy_sample = min(c, args.sampled_greedy_max_sample)
        pm_g = deepcopy(pm)
        greedy = SampledGreedySelector(
            pm_g, rag, sample_size_per_class=greedy_sample, seed=args.seed
        )
        if progress:
            progress.phase_start(dataset_key, c, ci, "SampledGreedy", 0)
        cb_g = (lambda m: print(f"         · {m}", flush=True)) if progress else None
        prompts_g = greedy.select_prompts(args.user_input, args.phi, progress_callback=cb_g)
        emb_g = _embed_texts(client, prompts_g)
        rel_g = float(np.mean([_cosine(user_emb, e) for e in emb_g])) if emb_g else 0.0
        if progress:
            progress.phase_done(rel_g)

        pm_p = deepcopy(pm)
        selector = IndividualPromptSelector(pm_p, rag_ps)
        if progress:
            progress.phase_start(dataset_key, c, ci, "PromptSelector (LLM)", 1)
        cb_p = (lambda m: print(f"         · {m}", flush=True)) if progress else None
        prompts_p = selector.select_prompts(
            args.user_input, args.phi, progress_callback=cb_p
        )
        emb_p = _embed_texts(client, prompts_p)
        rel_p = float(np.mean([_cosine(user_emb, e) for e in emb_p])) if emb_p else 0.0
        if progress:
            progress.phase_done(rel_p)

        pm_n = deepcopy(pm)
        naive_sel = NaiveSelector(
            pm_n,
            rag_naive,
            max_batch_size=args.naive_batch_size,
            mock_llm=args.naive_mock_llm,
            mock_top_fraction=args.naive_top_fraction,
            seed=args.seed,
        )
        if progress:
            progress.phase_start(dataset_key, c, ci, "NaiveSelector", 2)
        cb_n = (lambda m: print(f"         · {m}", flush=True)) if progress else None
        prompts_b = naive_sel.select_prompts(
            args.user_input, args.phi, progress_callback=cb_n
        )
        emb_b = _embed_texts(client, prompts_b)
        rel_b = float(np.mean([_cosine(user_emb, e) for e in emb_b])) if emb_b else 0.0
        if progress:
            progress.phase_done(rel_b)

        rows.append(
            {
                "dataset": dataset_key,
                "dataset_display": display_name,
                "candidates_per_class": c,
                "sampled_greedy_effective_sample": greedy_sample,
                "relevance_sampled_greedy": round(rel_g, 4),
                "relevance_prompt_selector": round(rel_p, 4),
                "relevance_naive": round(rel_b, 4),
                "phi": args.phi,
                "small_k": args.small_k,
                "large_k": args.large_k,
                "naive_mock_llm": args.naive_mock_llm,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")

    try:
        import matplotlib.pyplot as plt

        df["candidates_per_class"] = df["candidates_per_class"].astype(int)
        xs = df["candidates_per_class"].tolist()

        plt.figure(figsize=(8, 4.5))
        plt.plot(
            df["candidates_per_class"],
            df["relevance_prompt_selector"],
            marker="s",
            label="PromptSelector",
        )
        plt.plot(
            df["candidates_per_class"],
            df["relevance_sampled_greedy"],
            marker="o",
            label="SampledGreedy",
        )
        plt.plot(
            df["candidates_per_class"],
            df["relevance_naive"],
            marker="^",
            label="NaiveSelector",
        )
        plt.xlabel("Average #Candidate_prompts per class")
        plt.ylabel("Average relevance (Cosine Similarity)")
        plt.title(
            f"Prompt instance selection: relevance vs candidates per class — {display_name}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(xs)
        plt.tight_layout()
        plt.savefig(fig_out, dpi=150)
        plt.close()
        print(f"Wrote {fig_out}")
    except ImportError:
        print("matplotlib not installed; skipped figure. pip install matplotlib")


def main():
    parser = argparse.ArgumentParser(
        description="Relevance vs candidate pool size for three prompt selectors."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="diffusion_db, liar, race, or all (default: all)",
    )
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument(
        "--separated", type=lambda x: str(x).lower() == "true", default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters_primary", type=int, default=3)
    parser.add_argument("--n_clusters_secondary", type=int, default=5)
    parser.add_argument("--phi", type=int, default=2)
    parser.add_argument("--small_k", type=int, default=5)
    parser.add_argument("--large_k", type=int, default=30)
    parser.add_argument(
        "--user_input",
        type=str,
        default="Create a portrait of a famous person",
    )
    parser.add_argument(
        "--candidate_values",
        type=str,
        default="10,100,500,1000,2000",
        help="Comma-separated candidate counts per class",
    )
    parser.add_argument("--promptselector_top_l", type=int, default=3)
    parser.add_argument(
        "--sampled_greedy_max_sample",
        type=int,
        default=25,
        help="SampledGreedy: sample at most this many prompts per class (capped by c on x-axis). "
        "Uses full RAG embeddings DB per class, not the truncated pool.",
    )
    parser.add_argument("--naive_batch_size", type=int, default=15)
    parser.add_argument(
        "--naive_mock_llm",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="true = cosine mock tournament; false = LLM per batch (costly)",
    )
    parser.add_argument(
        "--naive_top_fraction",
        type=float,
        default=0.15,
        help="NaiveSelector mock: random among top this fraction by cosine",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logs (overall %% and per-algorithm sub-steps).",
    )
    args = parser.parse_args()
    if args.sampled_greedy_max_sample < 1:
        raise SystemExit("--sampled_greedy_max_sample must be >= 1")

    datasets = _parse_dataset_arg(args.dataset)
    rng = np.random.default_rng(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    candidate_values = [
        int(x.strip()) for x in args.candidate_values.split(",") if x.strip()
    ]
    n_c_grid = len(sorted(set(candidate_values)))
    progress = None if args.quiet else ExperimentProgress(datasets, n_c_grid)

    if progress:
        print(
            f"\nProgress: {len(datasets)} dataset(s) × {n_c_grid} candidate grid × 3 algorithms "
            f"= {progress.total_phases} phases.\n",
            flush=True,
        )

    for key in datasets:
        run_one_dataset(key, args, rng, ts, progress=progress)

    if progress:
        print(f"\n  [100.0%] All datasets finished.\n", flush=True)


if __name__ == "__main__":
    main()
