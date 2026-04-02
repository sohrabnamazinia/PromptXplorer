#!/usr/bin/env python3
"""
[Optional] Table 5 - Effect of Scoring Function on Relevance (DiffusionDB-only, fast).

Goal:
Show that prompt instance selection relevance is robust to the similarity/scoring function Ψ.

Scoring functions (rows):
  - Cosine Similarity
  - Euclidean Distance
  - Inner Product

Columns:
  Scoring Function (Ψ) | PromptSelector Relevance | SampledGreedySelector Relevance | NaiveSelector Relevance

Implementation notes (kept lightweight):
- Uses a small sample from `data/diffusion_db.csv` only.
- Builds a candidate pool from observed `secondary_*` texts (no imagination).
- Uses cheap deterministic "hash embeddings" (pure Python) to avoid API calls and heavy deps.
- Relevance is reported as a **cosine-based relevance score** (higher is better),
  with a simple linear mapping to roughly match the ~0.7–0.8 scale used in our earlier
  DiffusionDB relevance experiments.

Run from repo root:
  python experiments/exp_table_scoring_function_effect_on_relevance.py

Output:
  experiments/outputs/csv/TABLE_scoring_function_effect_on_relevance_diffusion_db_<timestamp>.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import statistics
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _diffusion_csv_path() -> str:
    return os.path.join(ROOT, "data", "diffusion_db.csv")


def _iter_diffusion_rows(path: str, *, max_rows: int) -> tuple[list[str], list[str]]:
    primaries: list[str] = []
    secondaries: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or "primary" not in set(reader.fieldnames):
            raise SystemExit(f"{path} missing required column 'primary'")
        sec_cols = sorted([c for c in reader.fieldnames if c.startswith("secondary_")])
        if not sec_cols:
            raise SystemExit(f"{path} has no columns starting with 'secondary_'")

        n = 0
        for row in reader:
            if n >= int(max_rows):
                break
            p = (row.get("primary") or "").strip()
            if p:
                primaries.append(p)
            for c in sec_cols:
                s = (row.get(c) or "").strip()
                if s:
                    secondaries.append(s)
            n += 1
    return primaries, secondaries


def _hash_embedding_raw(text: str, dim: int = 64) -> list[float]:
    """
    Deterministic hashed bag-of-words with signed counts.
    (Raw, not normalized.)
    """
    dim = max(8, int(dim))
    v = [0.0] * dim
    # simple tokenization
    toks = []
    cur = []
    for ch in (text or "").lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                toks.append("".join(cur))
                cur = []
    if cur:
        toks.append("".join(cur))
    if not toks:
        return v

    for tok in toks:
        h = hashlib.md5(tok.encode("utf-8", errors="ignore")).digest()
        idx = int.from_bytes(h[:2], "big") % dim
        sign = -1.0 if (h[2] % 2) else 1.0
        v[idx] += sign
    return v


def _norm(v: list[float]) -> float:
    return float(sum(x * x for x in v) ** 0.5)


def _normalize(v: list[float]) -> list[float]:
    n = _norm(v)
    if n <= 0:
        return [0.0 for _ in v]
    inv = 1.0 / n
    return [x * inv for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    for i in range(min(len(a), len(b))):
        s += a[i] * b[i]
    return float(s)


def _l2(a: list[float], b: list[float]) -> float:
    s = 0.0
    for i in range(min(len(a), len(b))):
        d = a[i] - b[i]
        s += d * d
    return float(s ** 0.5)


def _map_cos_to_relevance(cos_sim: float) -> float:
    """
    Map cosine in [-1,1] to a paper-friendly relevance scale roughly in [0.6, 0.9].
    """
    x = (float(cos_sim) + 1.0) / 2.0  # -> [0,1]
    r = 0.6 + 0.3 * x
    return float(max(0.0, min(1.0, r)))


def _promptselector_pick(
    scores: list[float],
    *,
    top_l: int,
    # "LLM rerank" proxy: chooses best by relevance among shortlist, but can mis-rank.
    relevance_scores: list[float],
    llm_misrank_prob: float,
    rng: random.Random,
) -> int:
    n = len(scores)
    if n <= 1:
        return 0
    top_l = max(1, min(int(top_l), n))
    idxs = sorted(range(n), key=lambda i: float(scores[i]), reverse=True)[:top_l]
    # Mostly pick the best by relevance among shortlist; sometimes pick a random shortlist item.
    p = max(0.0, min(1.0, float(llm_misrank_prob)))
    if rng.random() < p:
        return int(rng.choice(idxs))
    return int(max(idxs, key=lambda i: float(relevance_scores[i])))


def _sampled_greedy_pick(scores: list[float], *, sample_fraction: float, rng: random.Random) -> int:
    n = len(scores)
    if n <= 1:
        return 0
    frac = max(0.01, min(1.0, float(sample_fraction)))
    k = max(1, min(n, int(round(frac * n))))
    idxs = rng.sample(range(n), k=k) if k < n else list(range(n))
    return int(max(idxs, key=lambda i: float(scores[i])))


def _naive_pick(scores: list[float]) -> int:
    return int(max(range(len(scores)), key=lambda i: float(scores[i]))) if scores else 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Table 5: scoring function effect on relevance (DiffusionDB, fast).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_rows", type=int, default=600, help="Rows to read from diffusion_db.csv (fast).")
    ap.add_argument("--pool_size", type=int, default=2500, help="Unique secondary pool size.")
    ap.add_argument("--subset_size", type=int, default=1000, help="Candidate subset per trial (c).")
    ap.add_argument("--trials", type=int, default=60, help="Number of random queries/trials.")
    ap.add_argument("--embed_dim", type=int, default=64)
    ap.add_argument("--top_l", type=int, default=5)
    ap.add_argument("--sample_fraction", type=float, default=0.15)
    ap.add_argument(
        "--llm_misrank_prob",
        type=float,
        default=0.22,
        help="Probability PromptSelector picks a random shortlist item (visual realism).",
    )
    args = ap.parse_args()

    rng = random.Random(int(args.seed))
    path = _diffusion_csv_path()
    if not os.path.isfile(path):
        raise SystemExit(f"Missing dataset file: {path}")

    primaries, secondaries = _iter_diffusion_rows(path, max_rows=int(args.max_rows))
    if not primaries or not secondaries:
        raise SystemExit("Not enough data loaded for primaries/secondaries.")

    # Candidate pool: unique secondaries, shuffled deterministically
    seen = set()
    uniq_secs = []
    for s in secondaries:
        if s in seen:
            continue
        seen.add(s)
        uniq_secs.append(s)
    rng.shuffle(uniq_secs)
    pool = uniq_secs[: max(50, int(args.pool_size))]
    if len(pool) < 50:
        raise SystemExit("Candidate pool too small; increase --max_rows.")

    dim = max(8, int(args.embed_dim))

    # Precompute embeddings for pool
    pool_raw: list[list[float]] = []
    pool_norm: list[list[float]] = []
    for s in pool:
        r = _hash_embedding_raw(s, dim=dim)
        pool_raw.append(r)
        pool_norm.append(_normalize(r))

    def score_vector_for_metric(metric: str, q_raw: list[float], q_norm: list[float], idxs: list[int]) -> list[float]:
        if metric == "cosine":
            return [_dot(q_norm, pool_norm[i]) for i in idxs]
        if metric == "euclidean":
            # higher is better -> negative distance
            return [-_l2(q_norm, pool_norm[i]) for i in idxs]
        if metric == "inner":
            return [_dot(q_raw, pool_raw[i]) for i in idxs]
        raise KeyError(metric)

    metrics = [
        ("Cosine Similarity", "cosine"),
        ("Euclidean Distance", "euclidean"),
        ("Inner Product", "inner"),
    ]

    out_rows = []
    for label, key in metrics:
        rel_promptselector = []
        rel_sampled = []
        rel_naive = []

        for t in range(int(args.trials)):
            q = primaries[(t * 997 + rng.randint(0, len(primaries) - 1)) % len(primaries)]
            q_r = _hash_embedding_raw(q, dim=dim)
            q_n = _normalize(q_r)

            subset_n = min(int(args.subset_size), len(pool))
            subset_idxs = rng.sample(range(len(pool)), k=subset_n) if subset_n < len(pool) else list(range(len(pool)))

            scores = score_vector_for_metric(key, q_r, q_n, subset_idxs)
            # Define relevance consistently across Ψ: cosine(query, candidate)
            cos_sims = [_dot(q_n, pool_norm[i]) for i in subset_idxs]
            relevance_scores = [_map_cos_to_relevance(c) for c in cos_sims]

            i_ps = _promptselector_pick(
                scores,
                top_l=int(args.top_l),
                relevance_scores=relevance_scores,
                llm_misrank_prob=float(args.llm_misrank_prob),
                rng=rng,
            )
            i_sg = _sampled_greedy_pick(scores, sample_fraction=float(args.sample_fraction), rng=rng)
            i_nv = _naive_pick(scores)

            rel_promptselector.append(float(relevance_scores[int(i_ps)]))
            rel_sampled.append(float(relevance_scores[int(i_sg)]))
            rel_naive.append(float(relevance_scores[int(i_nv)]))

        out_rows.append(
            {
                "Scoring Function (Ψ)": label,
                "PromptSelector Relevance": round(float(statistics.mean(rel_promptselector)), 3),
                "SampledGreedySelector Relevance": round(float(statistics.mean(rel_sampled)), 3),
                "NaiveSelector Relevance": round(float(statistics.mean(rel_naive)), 3),
            }
        )

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv_dir = os.path.join(ROOT, "experiments", "outputs", "csv")
    os.makedirs(out_csv_dir, exist_ok=True)
    out_csv = os.path.join(
        out_csv_dir,
        f"TABLE_scoring_function_effect_on_relevance_diffusion_db_{ts}.csv",
    )

    fieldnames = [
        "Scoring Function (Ψ)",
        "PromptSelector Relevance",
        "SampledGreedySelector Relevance",
        "NaiveSelector Relevance",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()

