#!/usr/bin/env python3
"""
Coverage vs φ: greedy vs stochastic k-set coverage. Y-axis is % of distinct classes
in the clustered dataset (fixed universe: all primary/secondary class ids that appear
in the data) covered by the k selected sequences.

Run from repo root:
  python experiments/exp_representative_selection_coverage_sequence_length.py

Datasets (CSV under data/): diffusion_db, liar, race
  --dataset all          run all three (default)
  --dataset diffusion_db only DiffusionDB
  --dataset liar         only LIAR
  --dataset race         only RACE

Outputs (one CSV + one PNG per dataset run):
  experiments/outputs/csv/rep_select_coverage_phi_<dataset>_<timestamp>.csv
  experiments/outputs/figs/rep_select_coverage_phi_<dataset>_<timestamp>.png

Optional:
  --n 200 --seed 42 --phi_values 1,3,5,7,9
"""

# -----------------------------------------------------------------------------
# Key parameters (defaults match argparse below; override via CLI)
# -----------------------------------------------------------------------------
# --dataset              all              # diffusion_db | liar | race | all
# --n                    200              # rows loaded per CSV (not "all")
# --separated            True
# --phi_values           1,3,5,7,9        # sequence lengths φ evaluated
# --large_k              50               # candidate class sequences per φ
# --small_k              5                # sequences selected (greedy / stochastic)
# --stochastic_sample_size 5              # pool size sampled each stochastic step
# --seed                 42
# --n_clusters_primary   5
# --n_clusters_secondary 10
#
# Fixed in code (no CLI):
#   Clustering algorithm    kmeans
#   Sequence sampling       support-weighted random walk (0.1 pseudo-mass)
#   Coverage denominator    all distinct class ids appearing in clustered data
# -----------------------------------------------------------------------------

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_model.load_data import DataLoader
from preprocessing.clusterer import Clustering
from algorithms.k_set_coverage import KSetCoverage

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


def _secondary_classes(pm):
    s = set()
    for cp in pm.composite_prompts:
        for sec in cp.secondaries:
            if sec.class_obj:
                s.add(sec.class_obj.index)
    return sorted(s)


def _sample_sequence_support(pm, primary_class, phi, rng):
    """One class sequence [primary, s1, ..., s_phi] via support-weighted steps (no LLM)."""
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


def _most_frequent_primary_class(pm):
    from collections import Counter

    c = Counter()
    for cp in pm.composite_prompts:
        if cp.primary.class_obj:
            c[cp.primary.class_obj.index] += 1
    if not c:
        return 0
    return c.most_common(1)[0][0]


def _parse_dataset_arg(raw: str) -> list[str]:
    s = (raw or "").strip().lower()
    if s in ("", "all"):
        return list(ALL_DATASET_KEYS)
    if s in DATASET_FILES:
        return [s]
    raise SystemExit(
        f"Unknown --dataset {raw!r}. Use: all, {', '.join(ALL_DATASET_KEYS)}"
    )


def run_one_dataset(dataset_key: str, args, rng: np.random.Generator, ts: str):
    csv_path = _dataset_csv_path(dataset_key)
    if not os.path.isfile(csv_path):
        raise SystemExit(f"Dataset file not found: {csv_path}")

    out_csv_dir = os.path.join(ROOT, "experiments", "outputs", "csv")
    out_fig_dir = os.path.join(ROOT, "experiments", "outputs", "figs")
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_fig_dir, exist_ok=True)

    base = f"rep_select_coverage_phi_{dataset_key}_{ts}"
    csv_path_out = os.path.join(out_csv_dir, f"{base}.csv")
    fig_path_out = os.path.join(out_fig_dir, f"{base}.png")
    display_name = DATASET_DISPLAY[dataset_key]

    print(f"\n=== Dataset: {display_name} ({csv_path}) ===")
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

    dataset_universe = set()
    for cp in pm.composite_prompts:
        if cp.primary.class_obj:
            dataset_universe.add(cp.primary.class_obj.index)
        for sec in cp.secondaries:
            if sec.class_obj:
                dataset_universe.add(sec.class_obj.index)
    n_dataset_classes = len(dataset_universe)

    phi_list = [int(x.strip()) for x in args.phi_values.split(",") if x.strip()]
    if not phi_list:
        raise SystemExit("--phi_values must list at least one integer φ")
    phi_list = sorted(set(phi_list))

    rows = []

    def union_coverage(selected):
        u = set()
        for seq in selected:
            u.update(seq)
        return len(u)

    for phi in phi_list:
        sequences = []
        for _ in range(args.large_k):
            sequences.append(_sample_sequence_support(pm, primary_class, phi, rng))
        seq_copy = [list(s) for s in sequences]

        kcov_g = KSetCoverage(pm, seq_copy)
        greedy_sel = kcov_g.run_greedy_coverage(args.small_k)
        cov_greedy = union_coverage(greedy_sel)

        kcov_s = KSetCoverage(pm, [list(s) for s in sequences])
        stoch_sel = kcov_s.run_stochastic_coverage(
            args.small_k, args.stochastic_sample_size, rng=rng
        )
        cov_stoch = union_coverage(stoch_sel)

        pool_universe = set()
        for seq in sequences:
            pool_universe.update(seq)
        n_pool = len(pool_universe)
        pct_g = (100.0 * cov_greedy / n_dataset_classes) if n_dataset_classes else 0.0
        pct_s = (100.0 * cov_stoch / n_dataset_classes) if n_dataset_classes else 0.0

        rows.append(
            {
                "dataset": dataset_key,
                "dataset_display": display_name,
                "phi": phi,
                "dataset_distinct_classes": n_dataset_classes,
                "pool_distinct_classes": n_pool,
                "coverage_greedy_count": cov_greedy,
                "coverage_stochastic_count": cov_stoch,
                "coverage_greedy_pct_of_dataset": round(pct_g, 2),
                "coverage_stochastic_pct_of_dataset": round(pct_s, 2),
                "small_k": args.small_k,
                "stochastic_sample_size": args.stochastic_sample_size,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path_out, index=False)
    print(f"Wrote {csv_path_out}")

    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(7, 4))
        plt.plot(
            df["phi"],
            df["coverage_greedy_pct_of_dataset"],
            marker="s",
            label="GreedyCoverage",
        )
        plt.plot(
            df["phi"],
            df["coverage_stochastic_pct_of_dataset"],
            marker="o",
            label=f"StochasticCoverage (sample={args.stochastic_sample_size}/step)",
        )
        plt.xlabel("Sequence length φ")
        plt.ylabel("Coverage score (%)")
        plt.ylim(0, 105)
        xticks = sorted({int(x) for x in df["phi"].tolist()})
        plt.xticks(xticks)
        plt.xlim(min(xticks) - 1.0, max(xticks) + 1.0)
        plt.title(
            f"Representative selection: coverage vs φ — {display_name}"
        )
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path_out, dpi=150)
        plt.close()
        print(f"Wrote {fig_path_out}")
    except ImportError:
        print("matplotlib not installed; skipped figure. pip install matplotlib")


def main():
    parser = argparse.ArgumentParser(
        description="Coverage vs φ for greedy vs stochastic k-set coverage."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        help="diffusion_db, liar, race, or all (default: all)",
    )
    parser.add_argument("--n", type=int, default=200, help="Rows from CSV (None = all)")
    parser.add_argument(
        "--separated", type=lambda x: str(x).lower() == "true", default=True
    )
    parser.add_argument(
        "--phi_values",
        type=str,
        default="1,3,5,7,9",
        help="Comma-separated φ values (default: 1,3,5,7,9)",
    )
    parser.add_argument("--large_k", type=int, default=50)
    parser.add_argument("--small_k", type=int, default=5)
    parser.add_argument(
        "--stochastic_sample_size",
        type=int,
        default=5,
        help="Candidates sampled per greedy step",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters_primary", type=int, default=5)
    parser.add_argument("--n_clusters_secondary", type=int, default=10)
    args = parser.parse_args()

    datasets = _parse_dataset_arg(args.dataset)
    rng = np.random.default_rng(args.seed)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Loading and clustering per dataset...")
    for key in datasets:
        run_one_dataset(key, args, rng, ts)


if __name__ == "__main__":
    main()
