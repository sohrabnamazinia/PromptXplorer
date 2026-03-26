#!/usr/bin/env python3
"""
Re-plot sequence-construction timing with capped y-axis (paper-ready).

Reads a TABLE CSV produced by:
  experiments/exp_scalability_sequence_construction_time_vs_phi.py

and produces a new figure with:
- fixed y-axis max (default: 4.0 seconds)
- any point above the cap is drawn at the cap with an upward arrow marker to indicate "exceeds axis"

Run from repo root:
  python temp.py
  python temp.py --csv TABLE_seq_construction_time_vs_phi_20260325_231524.csv
  python temp.py --y_max 4

Outputs (repo root):
  FIG_seq_construction_time_vs_phi_capped_y<ymax>_<timestamp>.png

Notes:
- The script prefers writing PNG directly (no SVG conversion) using Pillow if available.
- If Pillow is unavailable, it renders to a temporary SVG internally and converts to PNG if possible.
- A final `.svg` is only written if PNG conversion is unavailable (or if you pass --keep_svg true).
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
from datetime import datetime


ROOT = os.path.dirname(os.path.abspath(__file__))


def _resolve_csv_path(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        raise SystemExit("--csv is required")
    # 1) user provided an existing path
    if os.path.isfile(raw):
        return raw
    # 2) filename in outputs folder
    cand = os.path.join(ROOT, "experiments", "outputs", "csv", raw)
    if os.path.isfile(cand):
        return cand
    # 3) filename in repo root
    cand2 = os.path.join(ROOT, raw)
    if os.path.isfile(cand2):
        return cand2
    raise SystemExit(f"CSV not found: {raw!r} (also tried {cand!r} and {cand2!r})")


def _read_table(csv_path: str) -> dict[str, list[tuple[float, float]]]:
    series: dict[str, list[tuple[float, float]]] = {}
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        r = csv.DictReader(f)
        req = {"algorithm", "phi", "exec_time_mean_s"}
        if not r.fieldnames or not req.issubset(set(r.fieldnames)):
            raise SystemExit(f"CSV missing required columns {sorted(req)}. Found: {r.fieldnames}")
        for row in r:
            algo = str(row.get("algorithm") or "").strip()
            if not algo:
                continue
            try:
                phi = float(row.get("phi") or "")
                y = float(row.get("exec_time_mean_s") or "")
            except ValueError:
                continue
            series.setdefault(algo, []).append((phi, y))

    # sort each series by phi
    for k in list(series.keys()):
        series[k] = sorted(series[k], key=lambda p: p[0])
    return series


def _write_svg_capped(
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
    y_max: float,
    width: int = 980,
    height: int = 560,
) -> None:
    pad_l, pad_r, pad_t, pad_b = 88, 36, 60, 78
    xs = [x for pts in series.values() for (x, _y) in pts]
    if not xs:
        raise SystemExit("No data points in CSV.")
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_max = x_min + 1.0

    y_min = 0.0
    y_cap = float(y_max)
    if y_cap <= 0:
        raise SystemExit("--y_max must be > 0")

    def x_px(x: float) -> float:
        return pad_l + (x - x_min) * (width - pad_l - pad_r) / (x_max - x_min)

    def y_px(y: float) -> float:
        # map [y_min..y_cap] to [height-pad_b .. pad_t]
        y = max(y_min, min(y_cap, float(y)))
        return pad_t + (y_cap - y) * (height - pad_t - pad_b) / (y_cap - y_min)

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    # professional-ish palette
    palette = {
        "PromptIPF": "#2563eb",  # blue
        "PromptWalker": "#16a34a",  # green
        "WalkWithPartner": "#dc2626",  # red
    }
    fallback_colors = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#ea580c"]
    names = list(series.keys())

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="17" '
        f'font-family="Arial">{esc(title)}</text>'
    )

    # axes
    x0 = pad_l
    y0 = height - pad_b
    x1 = width - pad_r
    y1 = pad_t
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}" stroke="#111" stroke-width="1"/>')

    # axis labels
    parts.append(f'<text x="{width/2:.1f}" y="{height-28}" text-anchor="middle" font-size="14" font-family="Arial">{esc(x_label)}</text>')
    parts.append(
        f'<text x="26" y="{height/2:.1f}" text-anchor="middle" font-size="14" font-family="Arial" '
        f'transform="rotate(-90 26 {height/2:.1f})">{esc(y_label)}</text>'
    )

    # y ticks at 0,1,2,3,4 (or integer up to cap)
    y_ticks = []
    cap_int = int(round(y_cap))
    if abs(y_cap - cap_int) < 1e-9 and cap_int in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        y_ticks = list(range(0, cap_int + 1))
    else:
        # 5 ticks including endpoints
        y_ticks = [y_min + i * (y_cap - y_min) / 4.0 for i in range(5)]

    for yv in y_ticks:
        yp = y_px(float(yv))
        parts.append(f'<line x1="{x0}" y1="{yp:.1f}" x2="{x1}" y2="{yp:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x0-10}" y="{yp+4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#111">{float(yv):g}</text>')

    # x ticks for unique x values
    for xv in sorted(set(xs)):
        xp = x_px(float(xv))
        parts.append(f'<line x1="{xp:.1f}" y1="{y0}" x2="{xp:.1f}" y2="{y0+6}" stroke="#111" stroke-width="1"/>')
        label = str(int(xv)) if float(xv).is_integer() else str(xv)
        parts.append(f'<text x="{xp:.1f}" y="{y0+26}" text-anchor="middle" font-size="12" font-family="Arial" fill="#111">{esc(label)}</text>')

    # legend
    legend_x = x0 + 10
    legend_y = y1 + 12
    for i, name in enumerate(names):
        col = palette.get(name, fallback_colors[i % len(fallback_colors)])
        ly = legend_y + i * 18
        parts.append(f'<rect x="{legend_x}" y="{ly-11}" width="14" height="3" fill="{col}"/>')
        parts.append(f'<text x="{legend_x+20}" y="{ly-7}" font-size="12" font-family="Arial" fill="#111">{esc(name)}</text>')

    # plot series
    for i, name in enumerate(names):
        pts = series[name]
        col = palette.get(name, fallback_colors[i % len(fallback_colors)])

        # Build a capped path that stops after the FIRST overflow point.
        path_pts: list[tuple[float, float, bool]] = []  # (x, y_clamped, overflow_here)
        overflow_x: float | None = None
        for (xv, yv) in pts:
            xv_f = float(xv)
            yv_f = float(yv)
            if overflow_x is not None:
                continue
            if yv_f > y_cap:
                overflow_x = xv_f
                path_pts.append((xv_f, y_cap, True))
            else:
                path_pts.append((xv_f, yv_f, False))

        # polyline path (clamped)
        segs = []
        for j, (xv_f, yv_c, _of) in enumerate(path_pts):
            cmd = "M" if j == 0 else "L"
            segs.append(f"{cmd}{x_px(xv_f):.1f},{y_px(yv_c):.1f}")
        if segs:
            parts.append(f'<path d="{" ".join(segs)}" fill="none" stroke="{col}" stroke-width="2"/>')

        # markers, and a single "exceeds axis" arrow+label at first overflow
        for (xv_f, yv_c, of) in path_pts:
            xp = x_px(xv_f)
            yp = y_px(yv_c)
            parts.append(f'<circle cx="{xp:.1f}" cy="{yp:.1f}" r="3.6" fill="{col}"/>')
            if of:
                y_top = pad_t
                shaft_bottom = y_top + 3
                shaft_top = max(10.0, y_top - 16)  # in top margin
                parts.append(
                    f'<line x1="{xp:.1f}" y1="{shaft_bottom:.1f}" x2="{xp:.1f}" y2="{shaft_top:.1f}" stroke="{col}" stroke-width="2"/>'
                )
                ah = 6.0
                parts.append(
                    f'<path d="M{xp:.1f},{shaft_top:.1f} L{xp-ah/2:.1f},{shaft_top+ah:.1f} L{xp+ah/2:.1f},{shaft_top+ah:.1f} Z" fill="{col}"/>'
                )
                # Place label slightly *below* the top axis line for readability.
                label_y = y_top + 14
                parts.append(
                    f'<text x="{xp+8:.1f}" y="{label_y:.1f}" text-anchor="start" font-size="11" font-family="Arial" fill="{col}">exceeds axis</text>'
                )
                break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")


def _svg_to_png(svg_path: str, png_path: str) -> bool:
    """
    Best-effort conversion to PNG.
    Tries cairosvg first, then common CLI tools if available.
    """
    # 1) cairosvg (python)
    try:
        import cairosvg  # type: ignore

        cairosvg.svg2png(url=svg_path, write_to=png_path)
        return True
    except Exception:
        pass

    # 2) rsvg-convert
    rsvg = shutil.which("rsvg-convert")
    if rsvg:
        try:
            subprocess.run([rsvg, "-o", png_path, svg_path], check=True)
            return True
        except Exception:
            pass

    # 3) imagemagick
    magick = shutil.which("magick") or shutil.which("convert")
    if magick:
        try:
            subprocess.run([magick, svg_path, png_path], check=True)
            return True
        except Exception:
            pass

    return False


def _write_png_capped_pillow(
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
    y_max: float,
    width: int = 1400,
    height: int = 800,
) -> None:
    """
    Render a paper-ready PNG directly (no SVG->PNG conversion) using Pillow.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow not available") from e

    pad_l, pad_r, pad_t, pad_b = 130, 60, 90, 120
    xs = [x for pts in series.values() for (x, _y) in pts]
    if not xs:
        raise SystemExit("No data points in CSV.")
    x_min, x_max = min(xs), max(xs)
    if x_min == x_max:
        x_max = x_min + 1.0

    y_min = 0.0
    y_cap = float(y_max)
    if y_cap <= 0:
        raise SystemExit("--y_max must be > 0")

    def x_px(x: float) -> float:
        return pad_l + (x - x_min) * (width - pad_l - pad_r) / (x_max - x_min)

    def y_px(y: float) -> float:
        y = max(y_min, min(y_cap, float(y)))
        return pad_t + (y_cap - y) * (height - pad_t - pad_b) / (y_cap - y_min)

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

    # Colors
    palette = {
        "PromptIPF": (37, 99, 235),  # blue
        "PromptWalker": (22, 163, 74),  # green
        "WalkWithPartner": (220, 38, 38),  # red
    }
    fallback = [
        (37, 99, 235),
        (22, 163, 74),
        (220, 38, 38),
        (124, 58, 237),
        (234, 88, 12),
    ]
    names = list(series.keys())

    # Title
    d.text((width / 2, 35), title, fill=(17, 17, 17), font=font_title, anchor="mm")

    # Axes
    x0, y0 = pad_l, height - pad_b
    x1, y1 = width - pad_r, pad_t
    d.line((x0, y0, x1, y0), fill=(17, 17, 17), width=2)
    d.line((x0, y1, x0, y0), fill=(17, 17, 17), width=2)

    # Axis labels
    d.text((width / 2, height - 55), x_label, fill=(17, 17, 17), font=font_axis, anchor="mm")
    # y label rotated: draw to small temp image then rotate
    yl_img = Image.new("RGBA", (height, 60), (255, 255, 255, 0))
    yl_d = ImageDraw.Draw(yl_img)
    yl_d.text((height / 2, 30), y_label, fill=(17, 17, 17), font=font_axis, anchor="mm")
    yl_img = yl_img.rotate(90, expand=True)
    img.paste(yl_img, (20, int(height / 2 - yl_img.size[1] / 2)), yl_img)

    # Y ticks (prefer integers up to cap if cap is small int)
    y_ticks: list[float]
    cap_int = int(round(y_cap))
    if abs(y_cap - cap_int) < 1e-9 and 1 <= cap_int <= 10:
        y_ticks = [float(i) for i in range(0, cap_int + 1)]
    else:
        y_ticks = [y_min + i * (y_cap - y_min) / 4.0 for i in range(5)]

    grid = (229, 231, 235)
    for yv in y_ticks:
        yp = y_px(yv)
        d.line((x0, yp, x1, yp), fill=grid, width=1)
        d.text((x0 - 12, yp), f"{yv:g}", fill=(17, 17, 17), font=font_tick, anchor="rm")

    # X ticks
    for xv in sorted(set(xs)):
        xp = x_px(float(xv))
        d.line((xp, y0, xp, y0 + 8), fill=(17, 17, 17), width=2)
        lab = str(int(xv)) if float(xv).is_integer() else str(xv)
        d.text((xp, y0 + 32), lab, fill=(17, 17, 17), font=font_tick, anchor="mm")

    # Legend
    lx, ly = x0 + 10, y1 + 10
    for i, name in enumerate(names):
        col = palette.get(name, fallback[i % len(fallback)])
        d.line((lx, ly + i * 28, lx + 24, ly + i * 28), fill=col, width=4)
        d.text((lx + 34, ly + i * 28), name, fill=(17, 17, 17), font=font_legend, anchor="lm")

    # Plot
    for i, name in enumerate(names):
        pts = series[name]
        col = palette.get(name, fallback[i % len(fallback)])
        # Build a capped path that stops after the FIRST overflow point.
        path_pts: list[tuple[float, float, bool]] = []  # (x, y_clamped, overflow_here)
        overflow_seen = False
        for (xv, yv) in pts:
            if overflow_seen:
                continue
            xv_f = float(xv)
            yv_f = float(yv)
            if yv_f > y_cap:
                overflow_seen = True
                path_pts.append((xv_f, y_cap, True))
            else:
                path_pts.append((xv_f, yv_f, False))

        # line (clamped)
        xy = [(x_px(xv_f), y_px(yv_c)) for (xv_f, yv_c, _of) in path_pts]
        if len(xy) >= 2:
            d.line(xy, fill=col, width=4)

        # markers, and one arrow+label at overflow
        for (xv_f, yv_c, of) in path_pts:
            xp = x_px(xv_f)
            yp = y_px(yv_c)
            r = 6
            d.ellipse((xp - r, yp - r, xp + r, yp + r), fill=col, outline=col)
            if of:
                top = pad_t
                shaft_top = max(12, top - 26)
                d.line((xp, top + 4, xp, shaft_top), fill=col, width=4)
                ah = 10
                d.polygon(
                    [(xp, shaft_top), (xp - ah / 2, shaft_top + ah), (xp + ah / 2, shaft_top + ah)],
                    fill=col,
                )
                # Place label slightly below the top axis line (not in the margin).
                d.text((xp + 10, top + 18), "exceeds axis", fill=col, font=font_note, anchor="ls")
                break

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, format="PNG")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cap y-axis and add overflow arrows for the timing figure.")
    ap.add_argument(
        "--csv",
        type=str,
        default="TABLE_seq_construction_time_vs_phi_20260325_231524.csv",
        help="Path or filename of the TABLE CSV (default: the one you referenced).",
    )
    ap.add_argument("--y_max", type=float, default=2.0, help="Y-axis maximum in seconds (default: 2.0).")
    ap.add_argument(
        "--keep_svg",
        type=lambda x: str(x).lower() == "true",
        default=False,
        help="If true, also write the final SVG alongside the PNG (default: false).",
    )
    args = ap.parse_args()

    csv_path = _resolve_csv_path(args.csv)
    series = _read_table(csv_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    y_max = float(args.y_max)
    y_tag = str(int(y_max)) if float(y_max).is_integer() else str(y_max).replace(".", "p")
    base = f"FIG_seq_construction_time_vs_phi_capped_y{y_tag}_{ts}"

    png_out = os.path.join(ROOT, f"{base}.png")
    svg_final = os.path.join(ROOT, f"{base}.svg")
    svg_tmp = os.path.join(ROOT, f".{base}.tmp.svg")

    # Prefer direct PNG rendering (no conversion needed).
    try:
        _write_png_capped_pillow(
            png_out,
            title="Sequence Construction Execution Time vs Sequence Length φ",
            x_label="Sequence length φ",
            y_label="Execution time (seconds)",
            series=series,
            y_max=y_max,
        )
        print(f"Wrote {png_out}")
        if bool(args.keep_svg):
            _write_svg_capped(
                svg_final,
                title="Sequence Construction Execution Time vs Sequence Length φ",
                x_label="Sequence length φ",
                y_label="Execution time (seconds)",
                series=series,
                y_max=y_max,
            )
            print(f"Wrote {svg_final}")
        return
    except ImportError:
        # Fall back to SVG + conversion route below.
        pass

    _write_svg_capped(
        svg_tmp,
        title="Sequence Construction Execution Time vs Sequence Length φ",
        x_label="Sequence length φ",
        y_label="Execution time (seconds)",
        series=series,
        y_max=y_max,
    )

    if _svg_to_png(svg_tmp, png_out):
        print(f"Wrote {png_out}")
        if bool(args.keep_svg):
            # Write a visible SVG too (paper-friendly vector fallback).
            try:
                if os.path.exists(svg_final):
                    os.remove(svg_final)
                os.replace(svg_tmp, svg_final)
            except Exception:
                # If replace fails, just leave the tmp SVG (still usable).
                pass
            else:
                print(f"Wrote {svg_final}")
        else:
            # Cleanup temp SVG so output is PNG-only.
            try:
                os.remove(svg_tmp)
            except Exception:
                pass
        return

    # Conversion failed: preserve SVG for manual conversion/debugging.
    try:
        if os.path.exists(svg_final):
            os.remove(svg_final)
        os.replace(svg_tmp, svg_final)
    except Exception:
        svg_final = svg_tmp

    raise SystemExit(
        "Could not generate PNG (no SVG->PNG converter available). "
        "Install one of: `pip install cairosvg` or `brew install librsvg` (rsvg-convert) or ImageMagick. "
        f"SVG written as a fallback: {svg_final}"
    )


if __name__ == "__main__":
    main()
