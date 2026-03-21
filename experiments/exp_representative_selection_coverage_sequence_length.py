#!/usr/bin/env python3
"""
Coverage vs φ: greedy vs stochastic k-set coverage. Y-axis is % of distinct classes
in the clustered dataset (fixed universe: all primary/secondary class ids that appear
in the data) covered by the k selected sequences.

Run from repo root:
  python experiments/exp_representative_selection_coverage_sequence_length.py

Optional:
  python experiments/exp_representative_selection_coverage_sequence_length.py --dataset 1 --seed 42
  (φ is always 1, 3, 5, 7, 9 — use --phi_values to override)
"""

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

# Dataset index -> CSV path (extend when adding datasets 2 and 3)
DATASETS = {
    1: os.path.join(ROOT, "data", "diffusion_db.csv"),
}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=int, default=1, help="Dataset index (default 1)")
    parser.add_argument("--n", type=int, default=200, help="Rows from CSV (None = all)")
    parser.add_argument("--separated", type=lambda x: str(x).lower() == "true", default=True)
    parser.add_argument(
        "--phi_values",
        type=str,
        default="1,3,5,7,9",
        help="Comma-separated φ values to evaluate (default: 1,3,5,7,9)",
    )
    parser.add_argument("--large_k", type=int, default=50)
    parser.add_argument("--small_k", type=int, default=5)
    parser.add_argument(
        "--stochastic_sample_size",
        type=int,
        default=5,
        help="Candidates sampled per greedy step (smaller → stochastic worse vs full greedy)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_clusters_primary", type=int, default=5)
    parser.add_argument("--n_clusters_secondary", type=int, default=10)
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

    base = f"rep_select_coverage_phi_dataset{args.dataset}_{ts}"
    csv_path_out = os.path.join(out_csv_dir, f"{base}.csv")
    fig_path_out = os.path.join(out_fig_dir, f"{base}.png")

    print("Loading and clustering...")
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
        pct_g = (
            (100.0 * cov_greedy / n_dataset_classes) if n_dataset_classes else 0.0
        )
        pct_s = (
            (100.0 * cov_stoch / n_dataset_classes) if n_dataset_classes else 0.0
        )

        rows.append(
            {
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
            label="Greedy k-set coverage",
        )
        plt.plot(
            df["phi"],
            df["coverage_stochastic_pct_of_dataset"],
            marker="o",
            label=f"Stochastic (sample={args.stochastic_sample_size}/step)",
        )
        plt.xlabel("Sequence length φ (secondary classes per sequence)")
        plt.ylabel("Coverage (%)")
        plt.ylim(0, 105)
        xticks = sorted({int(x) for x in df["phi"].tolist()})
        plt.xticks(xticks)
        plt.xlim(min(xticks) - 1.0, max(xticks) + 1.0)
        plt.title("Representative selection: coverage vs φ")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path_out, dpi=150)
        plt.close()
        print(f"Wrote {fig_path_out}")
    except ImportError:
        print("matplotlib not installed; skipped figure. pip install matplotlib")


if __name__ == "__main__":
    main()
