#!/usr/bin/env python3
"""
Table 1 experiment: Dataset Summary.

Outputs a single table CSV with columns:
  dataset name, description, #rows, modality/type, Average prompt length (words), Preprocessing approach

Notes:
- `#rows` is computed from the *raw source files* in `data/` (before preprocessing).
- `Average prompt length (words)` is computed from the processed dataset CSVs in `data/`,
  measured on the `primary` field (words; whitespace-delimited).
- `Average prompt length` is measured on the dataset `primary` field (words; whitespace-delimited).
- Other columns are concise, dataset-specific summaries aligned with preprocessing scripts.

Run from repo root:
  python experiments/exp_get_datasets_info.py

Output:
  experiments/outputs/csv/TABLE_dataset_summary_<timestamp>.csv
"""

from __future__ import annotations

import csv
import os
import re
import sys
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


_WORD_RE = re.compile(r"\S+")


def _dataset_path(filename: str) -> str:
    return os.path.join(ROOT, "data", filename)


def _count_rows_and_avg_words(csv_path: str, text_col: str = "primary") -> tuple[int, float]:
    """
    Stream-read a CSV (handles quoted multiline cells) to compute:
    - number of data rows
    - average word count for `text_col`
    """
    n = 0
    total_words = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or text_col not in set(reader.fieldnames):
            raise SystemExit(
                f"{os.path.basename(csv_path)} missing required column {text_col!r}. "
                f"Found: {reader.fieldnames}"
            )
        for row in reader:
            n += 1
            text = (row.get(text_col) or "").strip()
            if text:
                total_words += len(_WORD_RE.findall(text))
    avg = (total_words / n) if n else 0.0
    return n, round(float(avg), 1)


def _count_raw_diffusion_prompts(txt_path: str) -> int:
    n = 0
    with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def _count_raw_liar_tsv(tsv_path: str) -> int:
    """
    Count "usable" LIAR rows in the raw TSV: at least 14 columns and non-empty statement.
    """
    n = 0
    with open(tsv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 14:
                continue
            statement = (row[2] or "").strip()
            if not statement:
                continue
            n += 1
    return n


def _count_raw_race_csv(csv_path: str) -> int:
    """
    Count "usable" RACE rows in the raw CSV: required columns exist and non-empty article/question.
    """
    n = 0
    with open(csv_path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        req = {"article", "question", "A", "B", "C", "D", "answer"}
        if not reader.fieldnames or not req.issubset(set(reader.fieldnames)):
            raise SystemExit(
                f"{os.path.basename(csv_path)} missing required columns. Found: {reader.fieldnames}"
            )
        for row in reader:
            if (row.get("article") or "").strip() and (row.get("question") or "").strip():
                n += 1
    return n


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    datasets = [
        {
            "key": "diffusion_db",
            "name": "DiffusionDB",
            "file_processed": "diffusion_db.csv",
            "file_raw": "diffusion_prompts.txt",
            "description": "Text-to-image prompt dataset decomposed into primary prompts and modifier clauses.",
            "modality/type": "text-to-image prompts (generative image)",
            "preprocessing": "LLM-based prompt decomposition into `primary` + `secondary_1..N` modifiers (from `diffusion_prompts.txt`).",
            "raw_counter": _count_raw_diffusion_prompts,
        },
        {
            "key": "liar",
            "name": "LIAR",
            "file_processed": "liar.csv",
            "file_raw": "LIAR_Raw.tsv",
            "description": "Political claims with veracity labels and speaker/topic/venue metadata (PolitiFact LIAR).",
            "modality/type": "text classification + structured metadata (semi-structured)",
            "preprocessing": "From `LIAR_Raw.tsv`: `primary` = claim statement; secondaries = templated topic/source/setting strings from metadata; keep original label.",
            "raw_counter": _count_raw_liar_tsv,
        },
        {
            "key": "race",
            "name": "RACE",
            "file_processed": "race.csv",
            "file_raw": "RACE_Raw.csv",
            "description": "Reading comprehension multiple-choice questions paired with passages; label is the correct option (A–D).",
            "modality/type": "reading comprehension (question+options + passage text)",
            "preprocessing": "From `RACE_Raw.csv`: `primary` = question + A–D options; secondaries = K contiguous passage segments (`secondary_passage_*`) built paragraph-first then length-balanced.",
            "raw_counter": _count_raw_race_csv,
        },
    ]

    rows = []
    for d in datasets:
        raw_path = _dataset_path(d["file_raw"])
        proc_path = _dataset_path(d["file_processed"])
        if not os.path.isfile(raw_path):
            raise SystemExit(f"Raw dataset file not found: {raw_path}")
        if not os.path.isfile(proc_path):
            raise SystemExit(f"Processed dataset file not found: {proc_path}")

        n_rows = int(d["raw_counter"](raw_path))
        _, avg_words = _count_rows_and_avg_words(proc_path, text_col="primary")
        rows.append(
            {
                "dataset name": d["name"],
                "description": d["description"],
                "#rows": n_rows,
                "modality/type": d["modality/type"],
                "Average prompt length (words)": avg_words,
                "Preprocessing approach": d["preprocessing"],
            }
        )

    out_dir = os.path.join(ROOT, "experiments", "outputs", "csv")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"TABLE_dataset_summary_{ts}.csv")

    fieldnames = [
        "dataset name",
        "description",
        "#rows",
        "modality/type",
        "Average prompt length (words)",
        "Preprocessing approach",
    ]
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out_path}")
    for r in rows:
        print(
            f"- {r['dataset name']}: #rows(raw)={r['#rows']}, avg_primary_words={r['Average prompt length (words)']}"
        )


if __name__ == "__main__":
    main()

