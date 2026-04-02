#!/usr/bin/env python3
"""
Utilities for dataset-conditioned scalability benchmarks.

Goal: derive synthetic-but-data-conditioned structures from the *real* dataset CSVs in `data/`
without using any LLM calls or external embedding APIs.

Key ideas:
- Map secondary prompt strings to "cluster ids" using stable hashing into N bins.
- Build support counts (secondary->secondary transitions) from observed secondary sequences.
- Build per-class candidate pools from observed secondary strings.
- Provide a cheap deterministic "hash embedding" for text to simulate similarity scoring.
"""

from __future__ import annotations

import csv
import hashlib
import os
import re
from dataclasses import dataclass
from typing import Iterable


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def dataset_csv_path(dataset_key: str) -> str:
    if dataset_key not in DATASET_FILES:
        raise KeyError(dataset_key)
    return os.path.join(ROOT, "data", DATASET_FILES[dataset_key])


_WS = re.compile(r"\s+")


def _norm_text(s: str) -> str:
    return _WS.sub(" ", (s or "").strip())


def iter_dataset_secondaries(
    dataset_key: str, *, max_rows: int | None = None
) -> Iterable[tuple[str, list[str]]]:
    """
    Yield (primary, secondaries[]) from processed dataset CSVs:
    - diffusion_db.csv: primary + secondary_1..N
    - liar.csv: primary + secondary_topics/source/setting
    - race.csv: primary + secondary_passage_1..K
    """
    path = dataset_csv_path(dataset_key)
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "primary" not in set(reader.fieldnames):
            raise RuntimeError(f"{path} missing 'primary' column")

        sec_cols: list[str] = []
        if dataset_key == "diffusion_db":
            sec_cols = sorted([c for c in reader.fieldnames if c.startswith("secondary_")])
        elif dataset_key == "liar":
            sec_cols = ["secondary_topics", "secondary_source", "secondary_setting"]
        elif dataset_key == "race":
            sec_cols = sorted(
                [c for c in reader.fieldnames if c.startswith("secondary_passage_")]
            )
        else:
            sec_cols = [c for c in reader.fieldnames if c.startswith("secondary")]

        n = 0
        for row in reader:
            if max_rows is not None and n >= int(max_rows):
                break
            primary = _norm_text(str(row.get("primary") or ""))
            secs = []
            for c in sec_cols:
                v = _norm_text(str(row.get(c) or ""))
                if v:
                    secs.append(v)
            yield primary, secs
            n += 1


def hash_to_bin(text: str, n_bins: int) -> int:
    """
    Stable hash bucket in [0, n_bins-1].
    """
    n_bins = max(1, int(n_bins))
    h = hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()
    return int(h[:8], 16) % n_bins


@dataclass(frozen=True)
class SupportCounts:
    n_bins: int
    # primary is treated as a single node 0 in these benchmarks
    primary_to_secondary: dict[tuple[int, int], int]
    secondary_to_secondary: dict[tuple[int, int], int]
    secondary_bin_to_texts: dict[int, list[str]]


def build_support_from_dataset(
    dataset_key: str,
    *,
    n_bins: int,
    max_rows: int | None = None,
) -> SupportCounts:
    """
    Build support counts from observed secondary sequences.
    - primary_to_secondary: counts of first secondary bin (from a fixed primary node 0)
    - secondary_to_secondary: consecutive transitions along the secondary list
    Also collects per-bin candidate text pools.
    """
    p2s: dict[tuple[int, int], int] = {}
    s2s: dict[tuple[int, int], int] = {}
    bin_texts: dict[int, list[str]] = {}

    for _primary, secs in iter_dataset_secondaries(dataset_key, max_rows=max_rows):
        if not secs:
            continue
        bins = [hash_to_bin(t, n_bins) for t in secs]
        first = bins[0]
        p2s[(0, first)] = p2s.get((0, first), 0) + 1
        for a, b in zip(bins, bins[1:]):
            if a == b:
                continue
            s2s[(a, b)] = s2s.get((a, b), 0) + 1

        # collect texts per bin
        for t, b in zip(secs, bins):
            bin_texts.setdefault(b, []).append(t)

    # de-duplicate texts (keep order)
    for b, texts in list(bin_texts.items()):
        seen = set()
        uniq = []
        for t in texts:
            if t in seen:
                continue
            seen.add(t)
            uniq.append(t)
        bin_texts[b] = uniq

    return SupportCounts(
        n_bins=int(n_bins),
        primary_to_secondary=p2s,
        secondary_to_secondary=s2s,
        secondary_bin_to_texts=bin_texts,
    )


def hash_embedding(text: str, dim: int = 64) -> list[float]:
    """
    Cheap deterministic "embedding" using hashed bag-of-words with signed counts.
    Returns a unit-normalized vector (list of floats).
    """
    dim = max(8, int(dim))
    v = [0.0] * dim
    toks = [t for t in re.findall(r"[A-Za-z0-9]+", text.lower()) if t]
    if not toks:
        return v
    for tok in toks:
        h = hashlib.md5(tok.encode("utf-8", errors="ignore")).digest()
        idx = int.from_bytes(h[:2], "big") % dim
        sign = -1.0 if (h[2] % 2) else 1.0
        v[idx] += sign
    # normalize
    norm = sum(x * x for x in v) ** 0.5
    if norm > 0:
        v = [x / norm for x in v]
    return v


def dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for i in range(min(len(a), len(b))):
        s += a[i] * b[i]
    return float(s)


def collect_text_pools(
    dataset_key: str, *, max_rows: int = 300
) -> tuple[list[str], list[str]]:
    """
    Return (primary_texts, secondary_texts) from a small prefix of the dataset.
    Intended for quickly conditioning synthetic experiments on real dataset content.
    """
    prim: list[str] = []
    sec: list[str] = []
    for p, secs in iter_dataset_secondaries(dataset_key, max_rows=max_rows):
        if p:
            prim.append(p)
        for s in secs:
            if s:
                sec.append(s)
    return prim, sec


def estimate_avg_primary_words(dataset_key: str, *, max_rows: int = 500) -> float:
    """
    Quick estimate of average primary length (words) from the first max_rows rows.
    """
    n = 0
    tot = 0
    for p, _secs in iter_dataset_secondaries(dataset_key, max_rows=max_rows):
        if not p:
            continue
        n += 1
        tot += len(p.split())
    return float(tot / n) if n else 0.0

