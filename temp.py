#!/usr/bin/env python3
"""
Plot curated prompt-instance relevance CSVs (*_GOOD.csv) under experiments/outputs/csv/.
Writes PNGs to experiments/outputs/figs/ with a run timestamp (same %Y%m%d_%H%M%S style as the main experiment).

Run from repo root:
  python temp.py
  python temp.py --dataset diffusion_db
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
CSVS_DIR = os.path.join(ROOT, "experiments", "outputs", "csv")
FIGS_DIR = os.path.join(ROOT, "experiments", "outputs", "figs")

DATASET_KEYS = ("diffusion_db", "liar", "race")


def _csv_path(dataset_key: str) -> str:
    return os.path.join(
        CSVS_DIR, f"prompt_instance_relevance_candidates_{dataset_key}_GOOD.csv"
    )


def _fig_path(dataset_key: str, ts: str) -> str:
    # Same timestamp pattern as exp_prompt_instance_selection_relevance_candidates_per_class.py
    return os.path.join(
        FIGS_DIR,
        f"prompt_instance_relevance_candidates_{dataset_key}_{ts}_GOOD.png",
    )


def plot_one(dataset_key: str, ts: str) -> None:
    csv_path = _csv_path(dataset_key)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Missing CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = [
        "candidates_per_class",
        "relevance_prompt_selector",
        "relevance_sampled_greedy",
        "relevance_naive",
    ]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing column {col!r}: {csv_path}")

    display_name = (
        str(df["dataset_display"].iloc[0])
        if "dataset_display" in df.columns
        else dataset_key
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise SystemExit("matplotlib required: pip install matplotlib") from e

    df = df.sort_values("candidates_per_class")
    df["candidates_per_class"] = df["candidates_per_class"].astype(int)
    xs = df["candidates_per_class"].tolist()

    os.makedirs(FIGS_DIR, exist_ok=True)
    fig_out = _fig_path(dataset_key, ts)

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot *_GOOD.csv relevance curves (experiments/outputs/csv)."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="diffusion_db, liar, race, or all (default: all)",
    )
    args = parser.parse_args()

    if args.dataset.lower() in ("all", ""):
        keys = list(DATASET_KEYS)
    elif args.dataset in DATASET_KEYS:
        keys = [args.dataset]
    else:
        raise SystemExit(
            f"Unknown --dataset {args.dataset!r}. Use: all, {', '.join(DATASET_KEYS)}"
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for k in keys:
        plot_one(k, ts)


if __name__ == "__main__":
    main()
