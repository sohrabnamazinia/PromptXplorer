#!/usr/bin/env python3
"""
Convert RACE CSV (data/RACE_Raw.csv) into PromptXplorer CSV (data/race.csv).

Primary: question + four options (A–D) as one reading-comprehension prompt.
Secondaries: up to K segments of the article (paragraph-first, then length-balanced).
Label: correct answer letter (A–D).

Usage:
  python preprocessing/preprocess_race_dataset.py -n 500
  python preprocessing/preprocess_race_dataset.py --num-rows 1000
  python preprocessing/preprocess_race_dataset.py
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(ROOT, "data", "RACE_Raw.csv")
DEFAULT_OUTPUT = os.path.join(ROOT, "data", "race.csv")

# Target size per passage segment (characters) — avoid tiny or huge secondaries
DEFAULT_MIN_CHUNK = 220
DEFAULT_MAX_CHUNK = 820
DEFAULT_NUM_SECONDARIES = 3
DEFAULT_MIN_ARTICLE = 60

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
BLANK_LINES = re.compile(r"\n\s*\n+")


def normalize_ws(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").strip())


def split_paragraphs(article: str) -> list[str]:
    """Paragraph-first: double newlines; fallback to single-newline blocks."""
    article = article.replace("\r\n", "\n").strip()
    if not article:
        return []
    parts = [p.strip() for p in BLANK_LINES.split(article) if p.strip()]
    if len(parts) <= 1 and "\n" in article:
        parts = [ln.strip() for ln in article.split("\n") if ln.strip()]
    return parts


def split_long_segment(text: str, min_c: int, max_c: int) -> list[str]:
    """Break an oversized segment on sentence boundaries; else hard-split."""
    text = text.strip()
    if len(text) <= max_c:
        return [text] if text else []

    sents = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    if len(sents) <= 1:
        out = []
        for i in range(0, len(text), max_c):
            piece = text[i : i + max_c].strip()
            if piece:
                out.append(piece)
        return out

    chunks: list[str] = []
    buf = ""
    for s in sents:
        cand = (buf + " " + s).strip() if buf else s
        if len(cand) <= max_c:
            buf = cand
        else:
            if buf:
                chunks.append(buf)
            if len(s) > max_c:
                chunks.extend(split_long_segment(s, min_c, max_c))
                buf = ""
            else:
                buf = s
    if buf:
        chunks.append(buf)

    # Merge very short leading/trailing pieces into neighbors
    merged: list[str] = []
    for c in chunks:
        if not merged:
            merged.append(c)
            continue
        if len(c) < min_c and len(merged[-1]) + 1 + len(c) <= max_c:
            merged[-1] = (merged[-1] + " " + c).strip()
        else:
            merged.append(c)
    return merged


def paragraphs_to_chunks(paragraphs: list[str], min_c: int, max_c: int) -> list[str]:
    """Greedy merge of paragraphs; respect min/max per chunk."""
    if not paragraphs:
        return []

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0

    def flush():
        nonlocal cur, cur_len
        if not cur:
            return
        block = "\n\n".join(cur)
        cur, cur_len = [], 0
        if len(block) > max_c:
            chunks.extend(split_long_segment(block, min_c, max_c))
        else:
            chunks.append(block)

    for p in paragraphs:
        lp = len(p)
        if lp > max_c:
            flush()
            chunks.extend(split_long_segment(p, min_c, max_c))
            continue

        extra = (2 if cur else 0) + lp
        if cur_len + extra <= max_c:
            cur.append(p)
            cur_len += extra
            continue

        # Would exceed max — flush what we have
        if cur:
            flush()
        # Start new with p (p fits in max_c here)
        cur = [p]
        cur_len = lp

    flush()

    # Second pass: merge consecutive tiny chunks
    i = 0
    out: list[str] = []
    while i < len(chunks):
        c = chunks[i]
        while (
            len(c) < min_c
            and i + 1 < len(chunks)
            and len(c) + 2 + len(chunks[i + 1]) <= max_c
        ):
            c = c + "\n\n" + chunks[i + 1]
            i += 1
        out.append(c)
        i += 1

    return out


def _truncate_to_max(text: str, max_c: int) -> str:
    """Trim to max_c, preferring a word boundary."""
    text = text.strip()
    if len(text) <= max_c:
        return text
    cut = text[:max_c]
    sp = cut.rfind(" ")
    if sp > int(max_c * 0.55):
        cut = cut[:sp]
    return cut.strip()


def _split_full_into_k_slices(full: str, k: int, max_c: int) -> list[str]:
    """
    Divide the full passage into exactly k contiguous slices (beginning / middle / … / end).
    Each slice is capped at max_c. No recursion — avoids merge/split cycles on huge texts.
    """
    n = len(full)
    if k <= 0:
        return []
    if not full.strip():
        return [""] * k
    if k == 1:
        return [_truncate_to_max(full, max_c)]

    out: list[str] = []
    for i in range(k):
        lo = (i * n) // k
        hi = ((i + 1) * n) // k if i < k - 1 else n
        raw = full[lo:hi].strip()
        out.append(_truncate_to_max(raw, max_c))
    return out


def pack_to_k_segments(chunks: list[str], k: int, min_c: int, max_c: int) -> list[str]:
    """
    Turn paragraph-aware chunks into exactly k secondaries.

    We join all chunks (already built paragraph-first) and split the full text into k
    balanced contiguous slices, each capped at max_c. Short articles may yield slices
    below min_c — that is allowed when the passage is small.
    """
    if not chunks:
        return [""] * k
    full = "\n\n".join(c.strip() for c in chunks if c and str(c).strip())
    if not full.strip():
        return [""] * k
    return _split_full_into_k_slices(full, k, max_c)


def build_primary(question: str, a: str, b: str, c: str, d: str) -> str:
    q = normalize_ws(question)
    lines = [
        q,
        "",
        "Choose the best answer according to the reading passage:",
        f"A) {normalize_ws(a)}",
        f"B) {normalize_ws(b)}",
        f"C) {normalize_ws(c)}",
        f"D) {normalize_ws(d)}",
    ]
    return "\n".join(lines)


def wrap_secondary(index: int, total: int, body: str) -> str:
    body = body.strip()
    if not body:
        return ""
    labels = {
        (0, 3): "Opening section of the reading passage",
        (1, 3): "Middle section of the reading passage",
        (2, 3): "Closing section of the reading passage",
    }
    if (index, total) in labels:
        head = labels[(index, total)]
    else:
        head = f"Section {index + 1} of {total} from the reading passage"
    return f"{head}:\n\n{body}"


def parse_args():
    p = argparse.ArgumentParser(description="Build data/race.csv from RACE_Raw.csv")
    p.add_argument(
        "-n",
        "--num-rows",
        type=int,
        default=None,
        metavar="N",
        help="Max rows to write (default: all).",
    )
    p.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV path.")
    p.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV path.")
    p.add_argument(
        "--min-chunk",
        type=int,
        default=DEFAULT_MIN_CHUNK,
        help=f"Soft minimum characters per passage segment (default {DEFAULT_MIN_CHUNK}).",
    )
    p.add_argument(
        "--max-chunk",
        type=int,
        default=DEFAULT_MAX_CHUNK,
        help=f"Maximum characters per passage segment (default {DEFAULT_MAX_CHUNK}).",
    )
    p.add_argument(
        "-k",
        "--num-secondaries",
        type=int,
        default=DEFAULT_NUM_SECONDARIES,
        help=f"Number of passage columns (default {DEFAULT_NUM_SECONDARIES}).",
    )
    p.add_argument(
        "--min-article",
        type=int,
        default=DEFAULT_MIN_ARTICLE,
        help=f"Skip rows with article shorter than this (default {DEFAULT_MIN_ARTICLE}).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.min_chunk >= args.max_chunk:
        print("Error: --min-chunk must be less than --max-chunk", file=sys.stderr)
        return 1
    k = max(1, int(args.num_secondaries))

    if not os.path.isfile(args.input):
        print(f"Error: input not found: {args.input}", file=sys.stderr)
        return 1

    fieldnames = ["primary"] + [f"secondary_passage_{i+1}" for i in range(k)] + ["label"]

    written = 0
    skipped = 0

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.input, newline="", encoding="utf-8", errors="replace") as fin, open(
        args.output, "w", encoding="utf-8", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        req = {"article", "question", "A", "B", "C", "D", "answer"}
        if not reader.fieldnames or not req.issubset(set(reader.fieldnames)):
            print("Error: CSV must have columns: article, question, A, B, C, D, answer", file=sys.stderr)
            return 1

        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if args.num_rows is not None and written >= args.num_rows:
                break

            article = (row.get("article") or "").strip()
            question = (row.get("question") or "").strip()
            opt_a = row.get("A") or ""
            opt_b = row.get("B") or ""
            opt_c = row.get("C") or ""
            opt_d = row.get("D") or ""
            ans = (row.get("answer") or "").strip().upper()

            if not article or not question:
                skipped += 1
                continue
            if len(article) < args.min_article:
                skipped += 1
                continue
            if ans not in ("A", "B", "C", "D"):
                skipped += 1
                continue

            paras = split_paragraphs(article)
            if not paras:
                skipped += 1
                continue

            chunks = paragraphs_to_chunks(paras, args.min_chunk, args.max_chunk)
            segments = pack_to_k_segments(chunks, k, args.min_chunk, args.max_chunk)

            primary = build_primary(question, opt_a, opt_b, opt_c, opt_d)
            out_row = {"primary": primary, "label": ans}
            nonempty = sum(1 for s in segments if s.strip())
            if nonempty == 0:
                skipped += 1
                continue

            for i in range(k):
                key = f"secondary_passage_{i+1}"
                seg = segments[i].strip() if i < len(segments) else ""
                out_row[key] = wrap_secondary(i, k, seg) if seg else ""

            writer.writerow(out_row)
            written += 1

    print(f"Wrote {written} rows to {args.output}")
    if skipped:
        print(f"Skipped {skipped} rows (empty, too short, bad answer, or no chunks).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
