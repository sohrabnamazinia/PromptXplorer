#!/usr/bin/env python3
"""
Convert LIAR TSV (data/LIAR_Raw.tsv) into PromptXplorer CSV (data/liar.csv).

Each output row: claim as primary + three readable secondary lines (topics, source, setting),
plus the veracity label.

Usage:
  python preprocessing/preprocess_liar_dataset.py -n 500
  python preprocessing/preprocess_liar_dataset.py --num-rows 1000
  python preprocessing/preprocess_liar_dataset.py   # all rows
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

# Project root (parent of preprocessing/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT, "data", "LIAR_Raw.tsv")
DEFAULT_OUTPUT = os.path.join(ROOT, "data", "liar.csv")

# LIAR columns (0-based): id, label, statement, subjects, speaker, job, state, party,
# barely_true, false, half_true, mostly_true, pants_fire, context
N_COLS = 14


def humanize_slug(slug: str) -> str:
    """Turn politifact-style slug into readable words."""
    if not slug or not str(slug).strip():
        return ""
    s = str(slug).strip().replace("_", "-")
    parts = [p for p in s.split("-") if p]
    if not parts:
        return ""
    return " ".join(p.capitalize() for p in parts)


def format_party(party: str) -> str:
    if not party or not str(party).strip():
        return ""
    p = str(party).strip().lower()
    if p == "none":
        return ""
    mapping = {
        "democrat": "Democrat",
        "republican": "Republican",
        "independent": "Independent",
        "organization": "an organization",
        "columnist": "a columnist",
    }
    return mapping.get(p, party.strip().capitalize())


def build_secondary_topics(subjects: str) -> str:
    """Readable sentence about policy/topic tags."""
    if not subjects or not str(subjects).strip():
        return (
            "No topic tags were supplied for this claim; the subject area is unspecified "
            "in the dataset."
        )
    tags = [t.strip().replace("-", " ") for t in str(subjects).split(",") if t.strip()]
    if not tags:
        return (
            "No topic tags were supplied for this claim; the subject area is unspecified "
            "in the dataset."
        )
    if len(tags) == 1:
        return (
            f"This political claim mainly concerns the topic “{tags[0]}” "
            f"(as categorized in the LIAR dataset)."
        )
    listed = ", ".join(f"“{t}”" for t in tags[:-1])
    return (
        f"This claim is associated with multiple topics in the dataset: {listed}, "
        f"and “{tags[-1]}”."
    )


def build_secondary_source(speaker: str, job: str, state: str, party: str) -> str:
    """Readable attribution line."""
    who = humanize_slug(speaker)
    job_t = str(job).strip() if job else ""
    state_t = str(state).strip() if state else ""
    party_s = format_party(party)

    if not who:
        who = "An unidentified speaker or entity"

    clauses = [f"The statement is attributed to {who}"]

    if job_t:
        clauses.append(f"whose role is described as “{job_t}”")

    if state_t:
        clauses.append(f"with geographic or jurisdictional context “{state_t}”")

    if party_s:
        if party_s in ("a columnist", "an organization"):
            clauses.append(f"listed in the data as {party_s}")
        else:
            clauses.append(f"with party affiliation {party_s}")

    body = ", ".join(clauses[1:]) if len(clauses) > 1 else ""
    if body:
        return f"{clauses[0]}, {body}."
    return f"{clauses[0]} (no further speaker details were recorded)."


def build_secondary_setting(context: str) -> str:
    """Readable venue / medium line."""
    if not context or not str(context).strip():
        return (
            "The original dataset does not specify where or through which medium "
            "this statement was made."
        )
    c = str(context).strip()
    c0 = c[0].upper() + c[1:] if len(c) > 1 else c.upper()
    return (
        f"According to the dataset, the statement appeared in this setting or medium: {c0}"
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Build data/liar.csv from LIAR TSV for PromptXplorer."
    )
    p.add_argument(
        "-n",
        "--num-rows",
        type=int,
        default=None,
        metavar="N",
        help="Maximum number of data rows to write (default: all rows in the TSV).",
    )
    p.add_argument(
        "--input",
        default=DEFAULT_INPUT,
        help=f"Path to LIAR TSV (default: {DEFAULT_INPUT}).",
    )
    p.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT}).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    in_path = args.input
    out_path = args.output

    if not os.path.isfile(in_path):
        print(f"Error: input file not found: {in_path}", file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fieldnames = [
        "primary",
        "secondary_topics",
        "secondary_source",
        "secondary_setting",
        "label",
    ]

    written = 0
    skipped = 0

    with open(in_path, "r", encoding="utf-8", newline="") as fin, open(
        out_path, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.reader(fin, delimiter="\t")
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if args.num_rows is not None and written >= args.num_rows:
                break
            if len(row) < N_COLS:
                skipped += 1
                continue

            stmt_id = row[0].strip()
            label = row[1].strip()
            statement = row[2].strip()

            if not statement:
                skipped += 1
                continue

            subjects = row[3]
            speaker = row[4]
            job = row[5]
            state = row[6]
            party = row[7]
            context = row[13]

            writer.writerow(
                {
                    "primary": statement,
                    "secondary_topics": build_secondary_topics(subjects),
                    "secondary_source": build_secondary_source(speaker, job, state, party),
                    "secondary_setting": build_secondary_setting(context),
                    "label": label,
                }
            )
            written += 1

    print(f"Wrote {written} rows to {out_path}")
    if skipped:
        print(f"Skipped {skipped} malformed or empty rows while reading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
