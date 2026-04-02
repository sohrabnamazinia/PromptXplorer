#!/usr/bin/env python3
"""
Re-plot Figure 6 (rep selection vs large_k) from an existing CSV into a PNG.
Used for final_results visualization tweaks after manually editing CSV values.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys


def _load_series(path: str) -> tuple[list[int], dict[str, list[float]], str]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")

    required = {"algorithm", "large_k", "exec_time_mean_s"}
    if not required.issubset(set(rows[0].keys() or [])):
        raise SystemExit(f"{path} missing required columns: {sorted(required)}")

    dataset_display = str(rows[0].get("dataset_display") or rows[0].get("dataset") or "").strip()
    ks = sorted({int(float(r["large_k"])) for r in rows})
    algos = sorted({str(r["algorithm"]) for r in rows})

    series: dict[str, list[float]] = {a: [] for a in algos}
    for k in ks:
        for a in algos:
            m = [r for r in rows if str(r["algorithm"]) == a and int(float(r["large_k"])) == int(k)]
            series[a].append(float(m[0]["exec_time_mean_s"]) if m else float("nan"))
    return ks, series, dataset_display


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--title", type=str, default="")
    args = ap.parse_args()

    ks, series, ds = _load_series(str(args.input_csv))

    # Import the existing pillow plotter from the experiment file
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import exp_scalability_representative_selection_time_vs_large_k as exp6  # type: ignore

    title = str(args.title).strip()
    if not title:
        title = "Representative selection: execution time vs number of candidates"
        if ds:
            title = f"{title} – {ds} dataset"

    exp6._write_png_line_chart_pillow(  # type: ignore[attr-defined]
        str(args.out_png),
        title=title,
        x_label="Number of candidate sequences (large_k)",
        y_label="Execution time (seconds)",
        x_values=ks,
        series=series,
    )


if __name__ == "__main__":
    main()

