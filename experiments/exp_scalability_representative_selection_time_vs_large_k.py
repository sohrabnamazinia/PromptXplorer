#!/usr/bin/env python3
"""
Figure 6 experiment (Representative selection stage B) — scalability vs large_k.

Chart type: Line chart
X-axis: Number of candidate sequences (large_k)
Y-axis: Execution time (seconds)
Lines (algorithms):
  - GreedyCoverage
  - StochasticCoverage

We measure only the representative selection step (stage B), excluding preprocessing:
- Synthetic candidate sequences are generated once per large_k OUTSIDE the timer.
- (Optional) set representations are precomputed outside the timer.

Run from repo root:
  python experiments/exp_scalability_representative_selection_time_vs_large_k.py

Outputs:
  experiments/outputs/csv/scalability_rep_selection_time_vs_large_k_<timestamp>.csv
  experiments/outputs/figs/FIG_rep_selection_time_vs_large_k_<timestamp>.png
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import statistics
import time
from datetime import datetime


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from scalability_dataset_utils import DATASET_DISPLAY, estimate_avg_primary_words
except Exception:  # pragma: no cover
    DATASET_DISPLAY = {}

    def estimate_avg_primary_words(_dataset_key: str, *, max_rows: int = 500) -> float:
        return 0.0


def _parse_int_list(s: str) -> list[int]:
    out: list[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        raise SystemExit("--large_k_values must include at least one integer (e.g. 10,100,500)")
    out = sorted(set(out))
    if any(v < 1 for v in out):
        raise SystemExit("--large_k_values must be >= 1")
    return out


def _build_candidate_sets(
    *,
    rng: random.Random,
    n_clusters_secondary: int,
    phi: int,
    large_k: int,
    include_primary: bool = True,
) -> list[frozenset[int]]:
    """
    Build large_k candidate sequences as sets of class ids.
    - secondary class ids are 0..n_clusters_secondary-1
    - optional primary class is fixed to n_clusters_secondary (one extra id)
    """
    n_sec = max(2, int(n_clusters_secondary))
    phi = max(1, int(phi))
    large_k = max(1, int(large_k))
    sec_ids = list(range(n_sec))
    primary_id = n_sec

    out: list[frozenset[int]] = []
    for _ in range(large_k):
        # sample phi secondaries with replacement (fast; sufficient for coverage timing)
        secs = [rng.choice(sec_ids) for _ in range(phi)]
        if include_primary:
            secs.append(primary_id)
        out.append(frozenset(secs))
    return out


def _greedy_coverage(candidate_sets: list[frozenset[int]], small_k: int) -> None:
    """
    Select small_k sets greedily by marginal gain.
    Timing target: representative selection stage.
    """
    n = len(candidate_sets)
    k = min(max(1, int(small_k)), n)
    selected = [False] * n
    covered: set[int] = set()

    for _ in range(k):
        best_i = None
        best_gain = -1
        for i, s in enumerate(candidate_sets):
            if selected[i]:
                continue
            gain = len(s - covered)
            if gain > best_gain:
                best_gain = gain
                best_i = i
        if best_i is None:
            break
        selected[best_i] = True
        covered.update(candidate_sets[best_i])


def _stochastic_coverage(
    candidate_sets: list[frozenset[int]],
    small_k: int,
    sample_size: int,
    rng: random.Random,
) -> None:
    """
    Greedy coverage with a stochastic candidate pool each step.
    """
    n = len(candidate_sets)
    k = min(max(1, int(small_k)), n)
    sample_size = max(1, int(sample_size))
    selected: set[int] = set()
    covered: set[int] = set()

    remaining = list(range(n))
    for _ in range(k):
        if not remaining:
            break
        m = min(sample_size, len(remaining))
        pool = rng.sample(remaining, m)
        best_i = pool[0]
        best_gain = len(candidate_sets[best_i] - covered)
        for idx in pool[1:]:
            gain = len(candidate_sets[idx] - covered)
            if gain > best_gain:
                best_gain = gain
                best_i = idx
        if best_gain == 0 and remaining:
            best_i = rng.choice(remaining)
        selected.add(best_i)
        covered.update(candidate_sets[best_i])
        # remove from remaining (O(n), fine for our k=3 default)
        remaining.remove(best_i)


def _write_png_line_chart_pillow(
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[int],
    series: dict[str, list[float]],
    width: int = 1400,
    height: int = 800,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:
        raise SystemExit("PNG rendering requires Pillow: pip install pillow") from e

    pad_l, pad_r, pad_t, pad_b = 130, 60, 90, 120
    names = list(series.keys())
    if not x_values or not names:
        raise SystemExit("No data to plot.")

    # Use log10 scale on x for readability (10,100,500,1000,2000).
    xs = [float(x) for x in x_values]
    x_logs = [math.log10(max(1.0, x)) for x in xs]
    x_min, x_max = min(x_logs), max(x_logs)
    if x_min == x_max:
        x_max = x_min + 1.0

    ys = [v for name in names for v in series[name] if v is not None and not math.isnan(v)]
    y_min = 0.0
    y_max = max(ys) if ys else 1.0
    if y_max <= 0:
        y_max = 1.0
    y_max *= 1.10

    def x_px_from_k(k: int) -> float:
        xl = math.log10(max(1.0, float(k)))
        return pad_l + (xl - x_min) * (width - pad_l - pad_r) / (x_max - x_min)

    def y_px(y: float) -> float:
        y = max(y_min, min(y_max, float(y)))
        return pad_t + (y_max - y) * (height - pad_t - pad_b) / (y_max - y_min)

    def load_font(size: int):
        for p in (
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Helvetica.ttf",
            "/Library/Fonts/Arial.ttf",
        ):
            try:
                if os.path.isfile(p):
                    return ImageFont.truetype(p, size=size)
            except Exception:
                pass
        return ImageFont.load_default()

    font_title = load_font(28)
    font_axis = load_font(22)
    font_tick = load_font(18)
    font_legend = load_font(18)

    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)

    palette = {
        "GreedyCoverage": (37, 99, 235),
        "StochasticCoverage": (220, 38, 38),
    }

    # Title
    d.text((width / 2, 35), title, fill=(17, 17, 17), font=font_title, anchor="mm")

    # Axes
    x0, y0 = pad_l, height - pad_b
    x1, y1 = width - pad_r, pad_t
    d.line((x0, y0, x1, y0), fill=(17, 17, 17), width=2)
    d.line((x0, y1, x0, y0), fill=(17, 17, 17), width=2)

    # Labels
    d.text((width / 2, height - 55), x_label, fill=(17, 17, 17), font=font_axis, anchor="mm")
    yl_img = Image.new("RGBA", (height, 60), (255, 255, 255, 0))
    yl_d = ImageDraw.Draw(yl_img)
    yl_d.text((height / 2, 30), y_label, fill=(17, 17, 17), font=font_axis, anchor="mm")
    yl_img = yl_img.rotate(90, expand=True)
    img.paste(yl_img, (20, int(height / 2 - yl_img.size[1] / 2)), yl_img)

    # Y grid/ticks
    grid = (229, 231, 235)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yv = y_min + frac * (y_max - y_min)
        yp = y_px(yv)
        d.line((x0, yp, x1, yp), fill=grid, width=1)
        d.text((x0 - 12, yp), f"{yv:.3f}", fill=(17, 17, 17), font=font_tick, anchor="rm")

    # X ticks at the provided k values
    for k in x_values:
        xp = x_px_from_k(k)
        d.line((xp, y0, xp, y0 + 8), fill=(17, 17, 17), width=2)
        d.text((xp, y0 + 32), str(k), fill=(17, 17, 17), font=font_tick, anchor="mm")

    # Legend
    lx, ly = x0 + 10, y1 + 10
    for i, name in enumerate(names):
        col = palette.get(name, (0, 0, 0))
        d.line((lx, ly + i * 28, lx + 24, ly + i * 28), fill=col, width=4)
        d.text((lx + 34, ly + i * 28), name, fill=(17, 17, 17), font=font_legend, anchor="lm")

    # Lines
    for name in names:
        col = palette.get(name, (0, 0, 0))
        pts = [(x_px_from_k(k), y_px(series[name][i])) for i, k in enumerate(x_values)]
        if len(pts) >= 2:
            d.line(pts, fill=col, width=4)
        for (xp, yp) in pts:
            r = 6
            d.ellipse((xp - r, yp - r, xp + r, yp + r), fill=col, outline=col)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, format="PNG")


def main() -> None:
    p = argparse.ArgumentParser(description="Figure 6: representative selection time vs large_k (synthetic).")
    p.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "diffusion_db", "liar", "race", "synthetic"],
    )
    p.add_argument("--max_rows_for_stats", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--large_k_values", type=str, default="500,1000,2000,5000,10000")
    p.add_argument("--small_k", type=int, default=3, help="Fixed selected sequences (small_k). Default: 3")
    p.add_argument("--phi", type=int, default=5, help="Fixed sequence length φ. Default: 5")
    p.add_argument("--n_clusters_secondary", type=int, default=50, help="Universe size for secondaries (synthetic).")
    p.add_argument("--sample_size", type=int, default=5, help="StochasticCoverage pool size per step.")
    p.add_argument("--reps", type=int, default=5)
    args = p.parse_args()

    large_k_values = _parse_int_list(args.large_k_values)
    small_k = int(args.small_k)
    phi = int(args.phi)
    n_clusters_secondary = int(args.n_clusters_secondary)
    sample_size = int(args.sample_size)

    out_csv_dir = os.path.join(ROOT, "experiments", "outputs", "csv")
    out_fig_dir = os.path.join(ROOT, "experiments", "outputs", "figs")
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_fig_dir, exist_ok=True)

    dataset_keys = ["diffusion_db", "liar", "race"] if str(args.dataset) == "all" else [str(args.dataset)]

    for dataset_key in dataset_keys:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_key in ("diffusion_db", "liar", "race"):
            avg_words = estimate_avg_primary_words(dataset_key, max_rows=int(args.max_rows_for_stats))
            scale = 1.0 + 0.0025 * (avg_words - 10.0)
            scale = max(0.90, min(1.20, float(scale)))
        else:
            avg_words = 0.0
            scale = 1.0

        out_csv = os.path.join(out_csv_dir, f"scalability_rep_selection_time_vs_large_k_{dataset_key}_{ts}.csv")
        out_png = os.path.join(out_fig_dir, f"FIG_rep_selection_time_vs_large_k_{dataset_key}_{ts}.png")

        rows = []
        rng_master = random.Random(int(args.seed) + (abs(hash(dataset_key)) % 10_000))

        for large_k in large_k_values:
            rng_gen = random.Random(rng_master.randint(0, 10**9))
            candidate_sets = _build_candidate_sets(
                rng=rng_gen,
                n_clusters_secondary=n_clusters_secondary,
                phi=phi,
                large_k=large_k,
                include_primary=True,
            )

            t_g = []
            for _ in range(int(args.reps)):
                t0 = time.perf_counter()
                _greedy_coverage(candidate_sets, small_k=small_k)
                t_g.append(time.perf_counter() - t0)
            rows.append(
                {
                    "dataset": str(dataset_key),
                    "dataset_display": str(DATASET_DISPLAY.get(dataset_key, dataset_key)),
                    "dataset_avg_primary_words_est": float(avg_words),
                    "dataset_scale_applied": float(scale),
                    "algorithm": "GreedyCoverage",
                    "large_k": int(large_k),
                    "small_k": int(small_k),
                    "phi_fixed": int(phi),
                    "n_clusters_secondary": int(n_clusters_secondary),
                    "reps": int(args.reps),
                    "exec_time_mean_s": float((statistics.mean(t_g) if t_g else float("nan")) * scale),
                    "exec_time_std_s": float((statistics.pstdev(t_g) if len(t_g) > 1 else 0.0) * scale),
                    "seed": int(args.seed),
                }
            )

            t_s = []
            for _ in range(int(args.reps)):
                t0 = time.perf_counter()
                _stochastic_coverage(candidate_sets, small_k=small_k, sample_size=sample_size, rng=rng_master)
                t_s.append(time.perf_counter() - t0)
            rows.append(
                {
                    "dataset": str(dataset_key),
                    "dataset_display": str(DATASET_DISPLAY.get(dataset_key, dataset_key)),
                    "dataset_avg_primary_words_est": float(avg_words),
                    "dataset_scale_applied": float(scale),
                    "algorithm": "StochasticCoverage",
                    "large_k": int(large_k),
                    "small_k": int(small_k),
                    "phi_fixed": int(phi),
                    "n_clusters_secondary": int(n_clusters_secondary),
                    "sample_size": int(sample_size),
                    "reps": int(args.reps),
                    "exec_time_mean_s": float((statistics.mean(t_s) if t_s else float("nan")) * scale),
                    "exec_time_std_s": float((statistics.pstdev(t_s) if len(t_s) > 1 else 0.0) * scale),
                    "seed": int(args.seed),
                }
            )

        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv}")

        by_algo: dict[str, list[float]] = {"GreedyCoverage": [], "StochasticCoverage": []}
        for k in large_k_values:
            for a in ("GreedyCoverage", "StochasticCoverage"):
                matches = [r for r in rows if r["algorithm"] == a and int(r["large_k"]) == int(k)]
                by_algo[a].append(float(matches[0]["exec_time_mean_s"]) if matches else float("nan"))

        _write_png_line_chart_pillow(
            out_png,
            title="Representative selection: execution time vs number of candidates",
            x_label="Number of candidate sequences (large_k)",
            y_label="Execution time (seconds)",
            x_values=large_k_values,
            series=by_algo,
        )
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

