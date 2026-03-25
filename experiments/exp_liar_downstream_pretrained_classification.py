#!/usr/bin/env python3
"""
LIAR downstream experiment: primary-only vs primary+secondaries (refined), binary labels.

Uses a *pretrained* NLI model in zero-shot mode (no weight updates on LIAR).
Default model is small (~250MB): typeform/distilbert-base-uncased-mnli. For a larger model when you
have disk space, pass e.g. --model facebook/bart-large-mnli (~1.6GB). Uses GPU if CUDA is available.

Metrics: accuracy, F1 (binary, positive class = 1), macro-F1.

Output:
  experiments/outputs/csv/TABLE_liar_downstream_<timestamp>.csv

Requires:
  pip install transformers torch

Run from repo root:
  python experiments/exp_liar_downstream_pretrained_classification.py
  python experiments/exp_liar_downstream_pretrained_classification.py --max_samples 500
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

LIAR_PATH = os.path.join(ROOT, "data", "liar.csv")

# Small MNLI checkpoint (~250MB) — default to avoid large HF downloads on space-constrained machines.
DEFAULT_ZS_MODEL = "typeform/distilbert-base-uncased-mnli"

FALSE_LIKE = {"false", "pants-fire"}
TRUE_LIKE = {"barely-true", "half-true", "mostly-true", "true"}


def _label_to_binary(raw: str) -> int:
    s = (raw or "").strip().lower()
    if s in FALSE_LIKE:
        return 0
    if s in TRUE_LIKE:
        return 1
    raise ValueError(f"Unknown LIAR label: {raw!r}")


def _stratified_subsample(df: pd.DataFrame, y: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    n = min(n, len(df))
    if n == len(df):
        return df.copy()
    from sklearn.model_selection import StratifiedShuffleSplit

    split = StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed)
    idx, _ = next(split.split(np.zeros(len(df)), y))
    return df.iloc[idx].reset_index(drop=True)


def _run_zero_shot(texts: list[str], model_name: str, batch_size: int, hypothesis_template: str):
    try:
        import torch
        from transformers import pipeline
    except ImportError as e:
        raise SystemExit(
            "Install: pip install transformers torch\n" + str(e)
        ) from e

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )
    labels = ["true", "false"]
    preds: list[int] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        outs = clf(
            batch,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        if isinstance(outs, dict):
            outs = [outs]
        for o in outs:
            top = o["labels"][0]
            preds.append(1 if top == "true" else 0)
    return np.array(preds, dtype=int)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LIAR binary zero-shot: primary vs refined text (TABLE_* CSV)."
    )
    parser.add_argument("--max_samples", type=int, default=800)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_ZS_MODEL,
        help=f"Hugging Face NLI model id (default: {DEFAULT_ZS_MODEL}). "
        "Larger e.g. facebook/bart-large-mnli needs ~1.6GB disk.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Lower if you run out of memory (large models on CPU need smaller batches).",
    )
    parser.add_argument(
        "--hypothesis_template",
        type=str,
        default="This political claim is {}.",
        help="Must contain {} for the candidate label (true/false).",
    )
    args = parser.parse_args()

    if "{}" not in args.hypothesis_template:
        raise SystemExit("--hypothesis_template must contain '{}' placeholder")

    df = pd.read_csv(LIAR_PATH)
    need = {"primary", "secondary_topics", "secondary_source", "secondary_setting", "label"}
    miss = need - set(df.columns)
    if miss:
        raise SystemExit(f"liar.csv missing columns: {miss}")

    df = df.dropna(subset=["primary", "label"])
    df["_y"] = df["label"].map(lambda x: _label_to_binary(str(x)))

    y_full = df["_y"].to_numpy()
    df = _stratified_subsample(df, y_full, args.max_samples, args.seed)
    y = df["_y"].to_numpy()

    text_primary = df["primary"].astype(str).str.strip().tolist()
    text_refined = (
        df["primary"].astype(str).str.strip()
        + " "
        + df["secondary_topics"].fillna("").astype(str).str.strip()
        + " "
        + df["secondary_source"].fillna("").astype(str).str.strip()
        + " "
        + df["secondary_setting"].fillna("").astype(str).str.strip()
    ).str.replace(r"\s+", " ", regex=True).str.strip().tolist()

    print(f"Samples: {len(df)} | model: {args.model}")
    print("Running zero-shot on primary-only text...")
    pred_p = _run_zero_shot(
        text_primary,
        args.model,
        args.batch_size,
        args.hypothesis_template,
    )
    print("Running zero-shot on refined text...")
    pred_r = _run_zero_shot(
        text_refined,
        args.model,
        args.batch_size,
        args.hypothesis_template,
    )

    from sklearn.metrics import accuracy_score, f1_score

    def stats(name: str, pred: np.ndarray) -> dict:
        return {
            "dataset": "liar",
            "condition": name,
            "n_samples": len(y),
            "accuracy": round(float(accuracy_score(y, pred)), 4),
            "f1_binary": round(
                float(f1_score(y, pred, average="binary", pos_label=1, zero_division=0)),
                4,
            ),
            "f1_macro": round(
                float(f1_score(y, pred, average="macro", zero_division=0)),
                4,
            ),
            "model": args.model,
            "seed": args.seed,
            "label_map": "0=false,pants-fire | 1=barely-true,half-true,mostly-true,true",
            "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        }

    rows = [
        stats("primary_only", pred_p),
        stats("refined_primary_plus_secondaries", pred_r),
    ]
    out_df = pd.DataFrame(rows)

    out_dir = os.path.join(ROOT, "experiments", "outputs", "csv")
    os.makedirs(out_dir, exist_ok=True)
    ts = rows[0]["timestamp"]
    out_path = os.path.join(out_dir, f"TABLE_liar_downstream_{ts}.csv")
    out_df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
