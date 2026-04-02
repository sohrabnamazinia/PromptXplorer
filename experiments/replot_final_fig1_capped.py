#!/usr/bin/env python3
"""
Re-plot Figure 1 (seq construction vs φ) from an existing CSV into a PNG with:
- y-axis cap (default 2.0s)
- overflow arrow + "exceeds axis" label
- connector segment from the last in-range point to the overflow arrow base (so it isn't disjoint)

This is meant for final_results visualization tweaks without changing the CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
from typing import Any


def _write_png_line_chart_capped(
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[int],
    series: dict[str, list[float]],
    y_cap: float = 2.0,
    promptipf_overflow_gamma: float = 1.0,
    width: int = 1400,
    height: int = 800,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:
        raise SystemExit("PNG rendering requires Pillow: pip install pillow") from e

    pad_l, pad_r, pad_t, pad_b = 140, 60, 90, 120
    names = list(series.keys())
    if not x_values or not names:
        raise SystemExit("No data to plot.")

    xs = [float(x) for x in x_values]
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_max = x_min + 1.0

    y_min = 0.0
    y_max = float(y_cap)
    if y_max <= 0:
        y_max = 1.0

    def x_px(x: float) -> float:
        return pad_l + (float(x) - x_min) * (width - pad_l - pad_r) / (x_max - x_min)

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
    font_note = load_font(16)

    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)

    palette = {
        "PromptWalker": (37, 99, 235),
        "PromptIPF": (220, 38, 38),
        "WalkWithPartner": (16, 185, 129),
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
        d.text((x0 - 12, yp), f"{yv:.2f}", fill=(17, 17, 17), font=font_tick, anchor="rm")

    # X ticks
    for xv in x_values:
        xp = x_px(float(xv))
        d.line((xp, y0, xp, y0 + 8), fill=(17, 17, 17), width=2)
        d.text((xp, y0 + 32), str(int(xv)), fill=(17, 17, 17), font=font_tick, anchor="mm")

    # Legend
    lx, ly = x0 + 10, y1 + 10
    for i, name in enumerate(names):
        col = palette.get(name, (0, 0, 0))
        d.line((lx, ly + i * 28, lx + 24, ly + i * 28), fill=col, width=4)
        d.text((lx + 34, ly + i * 28), name, fill=(17, 17, 17), font=font_legend, anchor="lm")

    # Series with overflow connector
    for name in names:
        col = palette.get(name, (0, 0, 0))
        ys = series[name]

        pts: list[tuple[float, float]] = []
        last_in_range_px: tuple[float, float] | None = None
        last_in_range_data: tuple[float, float] | None = None
        overflow_arrow_base_px: tuple[float, float] | None = None

        for i, xv in enumerate(x_values):
            yv = float(ys[i])
            if yv > y_cap:
                # Place overflow arrow at the *intersection* with y_cap using the true slope
                # from the previous (in-range) point to this (overflow) point.
                if last_in_range_data is not None:
                    x_prev, y_prev = last_in_range_data
                    x_over, y_over = float(xv), float(yv)
                    if y_over == y_prev:
                        x_int = x_over
                    else:
                        t = (float(y_cap) - float(y_prev)) / (float(y_over) - float(y_prev))
                        # clamp in [0,1] for safety
                        t = max(0.0, min(1.0, float(t)))
                        # Optional: visually "sharpen" only the overflow segment for PromptIPF,
                        # without changing the underlying CSV values.
                        if name == "PromptIPF":
                            g = float(promptipf_overflow_gamma)
                            if g > 0 and abs(g - 1.0) > 1e-9:
                                t = t**g
                        x_int = float(x_prev) + t * (float(x_over) - float(x_prev))
                    overflow_arrow_base_px = (x_px(float(x_int)), y_px(float(y_cap)))
                else:
                    overflow_arrow_base_px = (x_px(float(xv)), y_px(float(y_cap)))
                break
            p = (x_px(float(xv)), y_px(yv))
            pts.append(p)
            last_in_range_px = p
            last_in_range_data = (float(xv), float(yv))

        # draw polyline for in-range points
        if len(pts) >= 2:
            d.line(pts, fill=col, width=4)

        # markers for in-range points
        for (xp, yp) in pts:
            r = 6
            d.ellipse((xp - r, yp - r, xp + r, yp + r), fill=col, outline=col)

        # overflow: connector + arrow + label
        if overflow_arrow_base_px is not None:
            ox, oy = overflow_arrow_base_px
            if last_in_range_px is not None:
                lx2, ly2 = last_in_range_px
                d.line((lx2, ly2, ox, oy), fill=col, width=4)

            # arrow
            d.line((ox, oy - 6, ox, oy - 24), fill=col, width=3)
            d.polygon([(ox, oy - 28), (ox - 6, oy - 20), (ox + 6, oy - 20)], fill=col)
            d.text((ox + 10, oy - 26), "exceeds axis", fill=(17, 17, 17), font=font_note, anchor="ls")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, format="PNG")


def _load_series_from_csv(path: str) -> tuple[list[int], dict[str, list[float]]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"No rows in {path}")

    # Detect schema used by Figure 1 output
    required = {"algorithm", "phi", "exec_time_mean_s"}
    if not required.issubset(set(rows[0].keys() or [])):
        raise SystemExit(f"{path} missing required columns: {sorted(required)}")

    algos = sorted({str(r["algorithm"]) for r in rows})
    phis = sorted({int(float(r["phi"])) for r in rows})

    by_algo: dict[str, list[float]] = {a: [] for a in algos}
    for p in phis:
        for a in algos:
            m = [r for r in rows if str(r["algorithm"]) == a and int(float(r["phi"])) == int(p)]
            if not m:
                by_algo[a].append(float("nan"))
            else:
                by_algo[a].append(float(m[0]["exec_time_mean_s"]))
    return phis, by_algo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True)
    ap.add_argument("--out_png", required=True)
    ap.add_argument("--y_cap", type=float, default=2.0)
    ap.add_argument(
        "--promptipf_overflow_gamma",
        type=float,
        default=1.0,
        help=">1 makes PromptIPF overflow segment steeper (visual-only). Default: 1.0",
    )
    args = ap.parse_args()

    x_values, series = _load_series_from_csv(str(args.input_csv))
    _write_png_line_chart_capped(
        str(args.out_png),
        title="Sequence Construction Execution Time vs Sequence Length φ",
        x_label="Sequence length φ",
        y_label="Execution time (seconds)",
        x_values=x_values,
        series=series,
        y_cap=float(args.y_cap),
        promptipf_overflow_gamma=float(args.promptipf_overflow_gamma),
    )


if __name__ == "__main__":
    main()

