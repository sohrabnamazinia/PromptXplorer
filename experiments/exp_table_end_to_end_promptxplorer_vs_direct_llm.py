#!/usr/bin/env python3
"""
Table 3 experiment: End-to-End Evaluation (PromptXplorer vs Direct LLM).

Columns:
  Dataset | Input Prompt | PromptXplorer Result | Direct LLM Result

Run from repo root:
  python experiments/exp_table_end_to_end_promptxplorer_vs_direct_llm.py

Datasets (CSV under data/): diffusion_db, liar, race
  --dataset all          run all three (default)
  --dataset diffusion_db | liar | race   single dataset

Output per run:
  experiments/outputs/csv/TABLE_end_to_end_promptxplorer_vs_direct_<scope>_<timestamp>.csv
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from algorithms.k_set_coverage import KSetCoverage
from algorithms.prompt_selector import IndividualPromptSelector
from algorithms.sequence_construction import RandomWalk
from algorithms.sequence_ordering import OrderSequence
from data_model.load_data import DataLoader
from llm.llm_interface import LLMInterface
from llm.rag import RAG
from preprocessing.clusterer import Clustering
from preprocessing.embedding import Embedding


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

DEFAULT_INPUT_PROMPTS = {
    "diffusion_db": "Create a portrait of a famous person",
    "liar": "Determine whether the claim is true or false and explain briefly.",
    "race": "Read the passage and answer the question with a short justification.",
}


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


def _resolve_input_prompt(dataset_key: str, args) -> str:
    if args.input_prompt:
        return args.input_prompt
    return DEFAULT_INPUT_PROMPTS[dataset_key]


def _generate_from_prompt(llm: LLMInterface, prompt_text: str) -> str:
    msg = llm.llm.invoke(prompt_text)
    content = getattr(msg, "content", "")
    return str(content).strip()


def _run_promptxplorer_pipeline(dataset_key: str, args, user_input: str) -> str:
    dataset_path = _dataset_csv_path(dataset_key)
    if not os.path.isfile(dataset_path):
        raise SystemExit(f"Dataset file not found: {dataset_path}")

    loader = DataLoader(separated=args.separated, n=args.n)
    pm = loader.load_data(dataset_path)

    clusterer = Clustering(pm, algorithm="kmeans")
    pm = clusterer.cluster(
        {
            "primary": {"n_clusters": args.n_clusters_primary},
            "secondary": {"n_clusters": args.n_clusters_secondary},
        }
    )

    embedding = Embedding(pm)
    embedding.embed()

    # Sequence construction + coverage
    rw = RandomWalk(pm)
    sequences = rw.random_walk_iter(user_input, args.phi, args.large_k)
    kcov = KSetCoverage(pm, sequences)
    kcov.run_greedy_coverage(args.small_k)

    # Prompt selection
    llm = LLMInterface(model=args.model)
    rag = RAG(embedding, llm, top_l=args.top_l)
    selector = IndividualPromptSelector(pm, rag)
    selector.select_prompts(user_input, args.phi)

    # Ordering and final chosen prompt for end-to-end generation
    ordered = OrderSequence(pm).order_sequences()
    if not ordered:
        return user_input
    return ordered[0]


def main():
    parser = argparse.ArgumentParser(
        description="Table 3: PromptXplorer vs Direct LLM end-to-end comparison."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="diffusion_db, liar, race, or all (default: all)",
    )
    parser.add_argument(
        "--input_prompt",
        type=str,
        default=None,
        help="If set, use one shared input prompt for all selected datasets.",
    )
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument(
        "--separated", type=lambda x: str(x).lower() == "true", default=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters_primary", type=int, default=3)
    parser.add_argument("--n_clusters_secondary", type=int, default=5)
    parser.add_argument("--phi", type=int, default=3)
    parser.add_argument("--large_k", type=int, default=30)
    parser.add_argument("--small_k", type=int, default=5)
    parser.add_argument("--top_l", type=int, default=5)
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI chat model for both PromptXplorer and direct baseline generation.",
    )

    args = parser.parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")

    datasets = _parse_dataset_arg(args.dataset)
    scope = "all" if len(datasets) > 1 else datasets[0]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out_csv_dir = os.path.join(ROOT, "experiments", "outputs", "csv")
    os.makedirs(out_csv_dir, exist_ok=True)
    out_csv = os.path.join(
        out_csv_dir, f"TABLE_end_to_end_promptxplorer_vs_direct_{scope}_{ts}.csv"
    )

    rows = []
    direct_llm = LLMInterface(model=args.model)

    print(f"Running Table 3 on datasets: {', '.join(datasets)}")
    for d in datasets:
        display = DATASET_DISPLAY[d]
        user_input = _resolve_input_prompt(d, args)
        print(f"\n[{display}] building PromptXplorer output...")
        final_prompt = _run_promptxplorer_pipeline(d, args, user_input)

        print(f"[{display}] generating PromptXplorer result...")
        px_result = _generate_from_prompt(direct_llm, final_prompt)

        print(f"[{display}] generating Direct LLM baseline...")
        direct_result = _generate_from_prompt(direct_llm, user_input)

        rows.append(
            {
                "Dataset": display,
                "Input Prompt": user_input,
                "PromptXplorer Result": px_result,
                "Direct LLM Result": direct_result,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "Dataset",
            "Input Prompt",
            "PromptXplorer Result",
            "Direct LLM Result",
        ],
    )
    df.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
