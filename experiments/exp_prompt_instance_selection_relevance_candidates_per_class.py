#!/usr/bin/env python3
"""
Prompt instance selection experiment.

Measure: relevance
Vary: candidate prompts per class

Run from repo root:
  python experiments/exp_prompt_instance_selection_relevance_candidates_per_class.py

Optional:
  python experiments/exp_prompt_instance_selection_relevance_candidates_per_class.py --dataset 1 --n 50 --seed 42
"""

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime

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
from algorithms.prompt_selector import IndividualPromptSelector, SampledGreedySelector


DATASETS = {
    1: os.path.join(ROOT, "data", "diffusion_db.csv"),
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=1)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--separated", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters_primary", type=int, default=3)
    parser.add_argument("--n_clusters_secondary", type=int, default=5)
    parser.add_argument("--phi", type=int, default=2)
    parser.add_argument("--small_k", type=int, default=5)
    parser.add_argument("--large_k", type=int, default=30)
    parser.add_argument("--user_input", type=str, default="Create a portrait of a famous person")
    parser.add_argument(
        "--candidate_values",
        type=str,
        default="5,10,50,100",
        help="Comma-separated candidate counts per class to test",
    )
    parser.add_argument(
        "--promptselector_top_l",
        type=int,
        default=3,
        help="How many candidates per class are passed to the LLM for PromptSelector reranking. Kept smaller than candidate pool size.",
    )
    args = parser.parse_args()

    if args.dataset not in DATASETS:
        raise SystemExit(f"Unknown dataset index {args.dataset}. Known: {list(DATASETS)}")
    csv_path = DATASETS[args.dataset]
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Dataset file not found: {csv_path}")

    rng = np.random.default_rng(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv_dir = os.path.join(ROOT, "experiments", "outputs", "csv")
    out_fig_dir = os.path.join(ROOT, "experiments", "outputs", "figs")
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_fig_dir, exist_ok=True)
    base = f"prompt_instance_relevance_candidates_dataset{args.dataset}_{ts}"
    csv_out = os.path.join(out_csv_dir, f"{base}.csv")
    fig_out = os.path.join(out_fig_dir, f"{base}.png")

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

    # Build k_class_sequences (same for both selectors), no LLM: support-weighted sampling
    primary_class = _most_frequent_primary_class(pm)
    sequences = [
        _sample_class_sequence_support(pm, primary_class, args.phi, rng)
        for _ in range(args.large_k)
    ]
    pm.k_class_sequences = sequences[: args.small_k]

    # Ensure embeddings exist for RAG
    embedding = Embedding(pm)
    embedding.embed()
    llm = LLMInterface()
    rag = RAG(embedding, llm, top_l=5)

    api_key = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    user_emb = _embed_texts(client, [args.user_input])[0]

    candidate_values = [int(x.strip()) for x in args.candidate_values.split(",") if x.strip()]
    candidate_values = sorted(set(candidate_values))

    rows = []
    for c in candidate_values:
        # Build a per-class candidate pool of size <= c for PromptSelector
        # (PromptSelector will then take top_l from this pool).
        rng_pool = np.random.default_rng(args.seed + c)

        # Lightweight wrapper to avoid deepcopying OpenAI/httpx internals
        # (which can contain non-picklable locks).
        class _RagProxy:
            def __init__(self, embeddings_db, top_l):
                self.embeddings_db = embeddings_db
                self.top_l = top_l

        top_l = min(args.promptselector_top_l, c)

        # Truncate embeddings_db to at most c items per class_label
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

        rag_ps = _RagProxy(truncated, top_l=top_l)

        # SampledGreedySelector (no LLM)
        pm_g = deepcopy(pm)
        greedy = SampledGreedySelector(pm_g, rag, sample_size_per_class=c, seed=args.seed)
        prompts_g = greedy.select_prompts(args.user_input, args.phi)
        emb_g = _embed_texts(client, prompts_g)
        rel_g = float(np.mean([_cosine(user_emb, e) for e in emb_g])) if emb_g else 0.0

        # PromptSelector (LLM rerank) over truncated candidate pool
        rag.top_l = c
        pm_p = deepcopy(pm)
        selector = IndividualPromptSelector(pm_p, rag_ps)
        prompts_p = selector.select_prompts(args.user_input, args.phi)
        emb_p = _embed_texts(client, prompts_p)
        rel_p = float(np.mean([_cosine(user_emb, e) for e in emb_p])) if emb_p else 0.0

        rows.append(
            {
                "candidates_per_class": c,
                "relevance_sampled_greedy": round(rel_g, 4),
                "relevance_prompt_selector": round(rel_p, 4),
                "phi": args.phi,
                "small_k": args.small_k,
                "large_k": args.large_k,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_out, index=False)
    print(f"Wrote {csv_out}")

    try:
        import matplotlib.pyplot as plt

        df["candidates_per_class"] = df["candidates_per_class"].astype(int)
        xs = df["candidates_per_class"].tolist()

        plt.figure(figsize=(7, 4))
        plt.plot(
            df["candidates_per_class"],
            df["relevance_prompt_selector"],
            marker="s",
            label="PromptSelector (top-L + LLM)",
        )
        plt.plot(
            df["candidates_per_class"],
            df["relevance_sampled_greedy"],
            marker="o",
            label="SampledGreedySelector (sample + nearest)",
        )
        plt.xlabel("Candidate prompts per class")
        plt.ylabel("Relevance (cosine similarity)")
        plt.title("Prompt instance selection: relevance vs candidates per class")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(xs)
        plt.tight_layout()
        plt.savefig(fig_out, dpi=150)
        plt.close()
        print(f"Wrote {fig_out}")
    except ImportError:
        print("matplotlib not installed; skipped figure. pip install matplotlib")


if __name__ == "__main__":
    main()

