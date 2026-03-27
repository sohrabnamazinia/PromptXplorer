#!/usr/bin/env python3
"""
Paper-ready bar chart for Figure 2 (capped y-axis + overflow arrows).

Reads a *_GOOD.csv with columns:
  dataset, algorithm, n_clusters_secondary, exec_time_mean_s

Creates a grouped bar chart PNG in repo root with:
- y-axis capped at 4 seconds
- if PromptIPF exceeds the cap, its bar is clamped and marked with an upward arrow

Run from repo root:
  python temp_num_clusters.py
  python temp_num_clusters.py --csv experiments/outputs/csv/seq_construction_time_vs_num_clusters_..._GOOD.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from datetime import datetime


ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_csv_path(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise SystemExit("--csv is required")
    if os.path.isfile(raw):
        return raw
    cand = os.path.join(ROOT, raw)
    if os.path.isfile(cand):
        return cand
    raise SystemExit(f"CSV not found: {raw!r}")


def _read_good(csv_path: str):
    rows = []
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        req = {"algorithm", "n_clusters_secondary", "exec_time_mean_s"}
        if not r.fieldnames or not req.issubset(set(r.fieldnames)):
            raise SystemExit(f"CSV missing required columns {sorted(req)}. Found: {r.fieldnames}")
        for row in r:
            algo = str(row.get("algorithm") or "").strip()
            if not algo:
                continue
            try:
                n = int(float(row.get("n_clusters_secondary") or ""))
                t = float(row.get("exec_time_mean_s") or "")
            except ValueError:
                continue
            rows.append((n, algo, t))

    if not rows:
        raise SystemExit("No usable rows found in CSV.")

    x_values = sorted({n for (n, _a, _t) in rows})
    algos = ["PromptIPF", "PromptWalker", "WalkWithPartner"]
    by_algo: dict[str, dict[int, float]] = {a: {} for a in algos}
    for n, a, t in rows:
        by_algo.setdefault(a, {})[n] = float(t)

    # Ensure ordering + fill missing with NaN
    series: dict[str, list[float]] = {}
    for a in algos:
        series[a] = [float(by_algo.get(a, {}).get(n, float("nan"))) for n in x_values]
    return x_values, series


def _write_png_grouped_bars_capped(
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[int],
    series: dict[str, list[float]],
    y_cap: float = 4.0,
    width: int = 1400,
    height: int = 800,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:
        raise SystemExit("This script writes PNG and requires Pillow: pip install pillow") from e

    pad_l, pad_r, pad_t, pad_b = 130, 60, 90, 120
    names = list(series.keys())
    if not x_values or not names:
        raise SystemExit("No data to plot.")

    # Axis scaling: clamp at y_cap
    y_min = 0.0
    y_max = float(y_cap)
    if y_max <= 0:
        raise SystemExit("y_cap must be > 0")

    def y_px(y: float) -> float:
        y = max(y_min, min(y_max, float(y)))
        return pad_t + (y_max - y) * (height - pad_t - pad_b) / (y_max - y_min)

    span = width - pad_l - pad_r
    group_w = span / len(x_values)

    def x_center(i: int) -> float:
        return pad_l + (i + 0.5) * group_w

    # Fonts (best-effort)
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
    font_note = load_font(16)

    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)

    palette = {
        "PromptIPF": (37, 99, 235),
        "PromptWalker": (22, 163, 74),
        "WalkWithPartner": (220, 38, 38),
    }
    patterns = {
        "PromptIPF": "solid",
        "PromptWalker": "diagonal",
        "WalkWithPartner": "dots",
    }

    def _darken(rgb: tuple[int, int, int], factor: float = 0.75) -> tuple[int, int, int]:
        r, g, b = rgb
        return (int(r * factor), int(g * factor), int(b * factor))

    def _fill_with_pattern(
        x_left: float,
        y_top: float,
        x_right: float,
        y_bottom: float,
        *,
        base: tuple[int, int, int],
        pattern: str,
    ) -> None:
        # Base fill
        d.rectangle((x_left, y_top, x_right, y_bottom), fill=base)
        # Overlay pattern for print/grayscale distinguishability
        overlay = _darken(base, 0.6)
        if pattern == "solid":
            return
        if pattern == "diagonal":
            # Diagonal hatch lines
            step = 10
            x0i = int(math.floor(x_left))
            y0i = int(math.floor(y_top))
            x1i = int(math.ceil(x_right))
            y1i = int(math.ceil(y_bottom))
            # lines with slope -1 across the box
            for k in range(-(y1i - y0i), (x1i - x0i), step):
                x_start = x0i + k
                y_start = y1i
                x_end = x0i + k + (y1i - y0i)
                y_end = y0i
                # clip by drawing and relying on PIL bounds (good enough)
                d.line((x_start, y_start, x_end, y_end), fill=overlay, width=2)
            return
        if pattern == "dots":
            # Dot hatch
            step = 12
            r = 2
            x0i = int(math.floor(x_left)) + 3
            y0i = int(math.floor(y_top)) + 3
            x1i = int(math.ceil(x_right)) - 3
            y1i = int(math.ceil(y_bottom)) - 3
            for yy in range(y0i, y1i, step):
                for xx in range(x0i, x1i, step):
                    d.ellipse((xx - r, yy - r, xx + r, yy + r), fill=overlay, outline=None)
            return
        # fallback: no pattern
        return

    # Title
    d.text((width / 2, 35), title, fill=(17, 17, 17), font=font_title, anchor="mm")

    # Axes
    x0, y0 = pad_l, height - pad_b
    x1, y1 = width - pad_r, pad_t
    d.line((x0, y0, x1, y0), fill=(17, 17, 17), width=2)
    d.line((x0, y1, x0, y0), fill=(17, 17, 17), width=2)

    # Axis labels
    d.text((width / 2, height - 55), x_label, fill=(17, 17, 17), font=font_axis, anchor="mm")
    yl_img = Image.new("RGBA", (height, 60), (255, 255, 255, 0))
    yl_d = ImageDraw.Draw(yl_img)
    yl_d.text((height / 2, 30), y_label, fill=(17, 17, 17), font=font_axis, anchor="mm")
    yl_img = yl_img.rotate(90, expand=True)
    img.paste(yl_img, (20, int(height / 2 - yl_img.size[1] / 2)), yl_img)

    # Y ticks: 0..y_cap integer
    grid = (229, 231, 235)
    for yv in range(0, int(round(y_max)) + 1):
        yp = y_px(float(yv))
        d.line((x0, yp, x1, yp), fill=grid, width=1)
        d.text((x0 - 12, yp), f"{yv}", fill=(17, 17, 17), font=font_tick, anchor="rm")

    # X ticks
    for i, xv in enumerate(x_values):
        cx = x_center(i)
        d.line((cx, y0, cx, y0 + 8), fill=(17, 17, 17), width=2)
        d.text((cx, y0 + 32), str(int(xv)), fill=(17, 17, 17), font=font_tick, anchor="mm")

    # Legend
    lx, ly = x0 + 10, y1 + 10
    for i, name in enumerate(names):
        col = palette.get(name, (0, 0, 0))
        # legend swatch with same hatch pattern
        sw_y0 = ly + i * 28 - 8
        sw_y1 = ly + i * 28 + 8
        _fill_with_pattern(
            lx,
            sw_y0,
            lx + 18,
            sw_y1,
            base=col,
            pattern=patterns.get(name, "solid"),
        )
        d.text((lx + 28, ly + i * 28), name, fill=(17, 17, 17), font=font_legend, anchor="lm")

    # Bars (grouped)
    inner_pad = group_w * 0.18
    bar_w = (group_w - inner_pad * 2) / max(1, len(names))
    exceeded_label_drawn = False

    for i, _xv in enumerate(x_values):
        left = pad_l + i * group_w + inner_pad
        for j, name in enumerate(names):
            val = float(series[name][i])
            if math.isnan(val):
                continue
            col = palette.get(name, (0, 0, 0))
            x_left = left + j * bar_w
            shown = min(val, y_max)
            y_top = y_px(shown)

            _fill_with_pattern(
                x_left,
                y_top,
                x_left + (bar_w - 3),
                y0,
                base=col,
                pattern=patterns.get(name, "solid"),
            )

            # Overflow marker for IPF only
            if name == "PromptIPF" and val > y_max:
                cx = x_left + (bar_w - 3) / 2.0
                top = pad_t
                # arrow just above the top axis line
                shaft_top = max(12, top - 26)
                d.line((cx, top + 4, cx, shaft_top), fill=col, width=4)
                ah = 10
                d.polygon(
                    [(cx, shaft_top), (cx - ah / 2, shaft_top + ah), (cx + ah / 2, shaft_top + ah)],
                    fill=col,
                )
                if not exceeded_label_drawn:
                    d.text((cx + 12, top + 18), "exceeds axis", fill=col, font=font_note, anchor="ls")
                    exceeded_label_drawn = True

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, format="PNG")


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot capped bar chart from simulated *_GOOD.csv.")
    ap.add_argument(
        "--csv",
        type=str,
        default="experiments/outputs/csv/seq_construction_time_vs_num_clusters_20260327_003051_GOOD.csv",
        help="Path to *_GOOD.csv (default: the one we generated).",
    )
    ap.add_argument("--y_cap", type=float, default=5.0, help="Y-axis cap in seconds (default: 5).")
    args = ap.parse_args()

    csv_path = _resolve_csv_path(args.csv)
    x_values, series = _read_good(csv_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = os.path.join(ROOT, f"FIG_seq_construction_time_vs_num_clusters_capped_{ts}.png")

    _write_png_grouped_bars_capped(
        out_png,
        title="Sequence construction execution time vs number of clusters",
        x_label="Number of clusters (secondary classes)",
        y_label="Execution time (seconds)",
        x_values=x_values,
        series=series,
        y_cap=float(args.y_cap),
    )
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

