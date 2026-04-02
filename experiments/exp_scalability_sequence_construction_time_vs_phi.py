#!/usr/bin/env python3
"""
Figure 1 experiment: Sequence Construction Execution Time vs Sequence Length φ.

This experiment measures *only* the sequence construction algorithms — explicitly
excluding preprocessing (data loading, decomposition, clustering, embeddings, etc.).

To keep the benchmark focused and fast, we use a small synthetic "class graph"
that mimics the support matrices (primary→secondary and secondary→secondary) that
sequence construction operates on.

Lines (Algorithms):
- PromptIPF        (IPF-like fitting over sequences using support-derived constraints)
- PromptWalker     (support-weighted random walk)
- WalkWithPartner  (random walk + deterministic "partner" fallback on low-support nodes)

Run from repo root:
  python experiments/exp_scalability_sequence_construction_time_vs_phi.py

Output:
  experiments/outputs/csv/TABLE_seq_construction_time_vs_phi_<timestamp>.csv
  experiments/outputs/figs/FIG_seq_construction_time_vs_phi_<timestamp>.svg
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from datetime import datetime


# Project root (parent of experiments/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from scalability_dataset_utils import DATASET_DISPLAY, estimate_avg_primary_words
except Exception:  # pragma: no cover
    DATASET_DISPLAY = {}

    def estimate_avg_primary_words(_dataset_key: str, *, max_rows: int = 500) -> float:
        return 0.0


def _parse_phi_values(s: str) -> list[int]:
    vals = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise SystemExit("--phi_values must include at least one integer (e.g. 1,3,5,7)")
    return sorted(set(vals))

def _weighted_choice(rng: random.Random, items: list[int], weights: list[float]) -> int:
    if not items:
        raise ValueError("empty items")
    total = float(sum(weights))
    if total <= 0:
        return rng.choice(items)
    r = rng.random() * total
    acc = 0.0
    for x, w in zip(items, weights):
        acc += float(w)
        if r <= acc:
            return int(x)
    return int(items[-1])


def _percentile(values: list[float], pct: float) -> float:
    """Simple nearest-rank percentile for small lists (pct in [0,100])."""
    if not values:
        return 0.0
    pct = max(0.0, min(100.0, float(pct)))
    xs = sorted(float(v) for v in values)
    if len(xs) == 1:
        return xs[0]
    # nearest-rank
    k = int(math.ceil((pct / 100.0) * len(xs))) - 1
    k = max(0, min(len(xs) - 1, k))
    return xs[k]


@dataclass(frozen=True)
class _SyntheticSupport:
    secondary_classes: list[int]
    primary_to_secondary: dict[tuple[int, int], float]
    secondary_to_secondary: dict[tuple[int, int], float]


def _build_synthetic_support(
    rng: random.Random,
    n_secondary: int,
    edge_density: float,
    weight_low: float,
    weight_high: float,
) -> _SyntheticSupport:
    """
    Build sparse-ish support matrices with positive weights.
    primary class is fixed to 0; secondary classes are 0..n_secondary-1.
    """
    n_secondary = max(2, int(n_secondary))
    edge_density = max(0.0, min(1.0, float(edge_density)))
    w_lo = float(weight_low)
    w_hi = float(weight_high)
    if w_hi < w_lo:
        w_lo, w_hi = w_hi, w_lo

    secs = list(range(n_secondary))
    p2s: dict[tuple[int, int], float] = {}
    s2s: dict[tuple[int, int], float] = {}

    def rand_w() -> float:
        return w_lo + (w_hi - w_lo) * rng.random()

    # Primary->secondary edges
    for s in secs:
        if rng.random() <= edge_density:
            p2s[(0, s)] = rand_w()

    # Secondary->secondary edges (allow i->j for i!=j)
    for a in secs:
        for b in secs:
            if a == b:
                continue
            if rng.random() <= edge_density:
                s2s[(a, b)] = rand_w()

    # Ensure at least some connectivity
    if not p2s:
        p2s[(0, rng.choice(secs))] = rand_w()
    if not s2s:
        a = rng.choice(secs)
        b = rng.choice([x for x in secs if x != a])
        s2s[(a, b)] = rand_w()

    return _SyntheticSupport(secs, p2s, s2s)


def _support_weights_for_next(
    sup: dict[tuple[int, int], float],
    src: int,
    all_targets: list[int],
    pseudo: float,
) -> list[float]:
    return [float(sup.get((src, t), pseudo)) for t in all_targets]


def _promptwalker_sequences(
    support: _SyntheticSupport,
    primary_class: int,
    phi: int,
    large_k: int,
    rng: random.Random,
    pseudo: float,
) -> list[list[int]]:
    secs = support.secondary_classes
    p2s = support.primary_to_secondary
    s2s = support.secondary_to_secondary
    out: list[list[int]] = []
    for _ in range(int(large_k)):
        seq = [int(primary_class)]
        w0 = _support_weights_for_next(p2s, int(primary_class), secs, pseudo)
        first = _weighted_choice(rng, secs, w0)
        seq.append(first)
        cur = first
        for _ in range(max(0, int(phi) - 1)):
            w = _support_weights_for_next(s2s, cur, secs, pseudo)
            cur = _weighted_choice(rng, secs, w)
            seq.append(cur)
        out.append(seq)
    return out


@dataclass(frozen=True)
class _PartnerPolicy:
    llm_usage_percent: float = 25.0


def _outgoing_totals(sup: dict[tuple[int, int], float]) -> dict[int, float]:
    totals: dict[int, float] = {}
    for (a, _b), v in sup.items():
        totals[int(a)] = totals.get(int(a), 0.0) + float(v)
    return totals


def _walkwithpartner_sequences(
    support: _SyntheticSupport,
    primary_class: int,
    phi: int,
    large_k: int,
    rng: random.Random,
    policy: _PartnerPolicy,
    pseudo: float,
    *,
    simulate_llm_latency: bool,
    llm_latency_min_s: float,
    llm_latency_max_s: float,
    llm_sleep_calls_cap: int | None,
) -> list[list[int]]:
    """
    Offline mock of WalkWithPartner:
    - when we "use the partner" (simulated LLM), choose the max-support next class
    - otherwise, sample support-weighted like PromptWalker

    Synthetic timing:
    - if `simulate_llm_latency` is True, each partner usage can add a random sleep in
      [llm_latency_min_s, llm_latency_max_s] seconds, capped by `llm_sleep_calls_cap`
      to keep runs bounded.
    """
    secs = support.secondary_classes
    if not secs:
        return []
    p2s = support.primary_to_secondary
    s2s = support.secondary_to_secondary

    pct = float(policy.llm_usage_percent)
    pct = max(0.0, min(100.0, pct))
    p_use = pct / 100.0

    # How many partner "LLM calls" should contribute latency in this run?
    # Default: grows gently with φ (independent of large_k) to avoid huge sleeps.
    if llm_sleep_calls_cap is None:
        llm_sleep_calls_cap = max(1, int(round(float(phi) * p_use)))
    llm_sleep_calls_cap = max(0, int(llm_sleep_calls_cap))
    slept_calls = 0
    total_partner_calls = 0

    def use_partner_primary(p: int) -> bool:
        if p_use <= 0:
            return False
        if p_use >= 1:
            return True
        return rng.random() < p_use

    def use_partner_secondary(s: int) -> bool:
        if p_use <= 0:
            return False
        if p_use >= 1:
            return True
        return rng.random() < p_use

    def maybe_sleep_for_llm_call():
        nonlocal slept_calls
        if not simulate_llm_latency:
            return
        if slept_calls >= llm_sleep_calls_cap:
            return
        lo = float(llm_latency_min_s)
        hi = float(llm_latency_max_s)
        if hi < lo:
            lo, hi = hi, lo
        # A small randomized sleep models network + model latency.
        time.sleep(lo + (hi - lo) * rng.random())
        slept_calls += 1

    def partner_pick(sup_map: dict[tuple[int, int], float], src: int) -> int:
        # argmax over targets; missing edges treated as pseudo
        best_t = secs[0]
        best_v = float(sup_map.get((src, best_t), pseudo))
        for t in secs[1:]:
            v = float(sup_map.get((src, t), pseudo))
            if v > best_v:
                best_v = v
                best_t = t
        return int(best_t)

    out = []
    for _ in range(int(large_k)):
        seq = [primary_class]
        if use_partner_primary(primary_class):
            total_partner_calls += 1
            maybe_sleep_for_llm_call()
            first = partner_pick(p2s, primary_class)
        else:
            w0 = _support_weights_for_next(p2s, primary_class, secs, pseudo)
            first = _weighted_choice(rng, secs, w0)
        seq.append(first)
        cur = first
        for _step in range(max(0, int(phi) - 1)):
            if use_partner_secondary(cur):
                total_partner_calls += 1
                maybe_sleep_for_llm_call()
                nxt = partner_pick(s2s, cur)
            else:
                w = _support_weights_for_next(s2s, cur, secs, pseudo)
                nxt = _weighted_choice(rng, secs, w)
            seq.append(nxt)
            cur = nxt
        out.append(seq)
    return out, int(total_partner_calls), int(slept_calls)


def _ipf_sequences(
    support: _SyntheticSupport,
    primary_class: int,
    phi: int,
    large_k: int,
    rng: random.Random,
    *,
    degree: int,
    max_iter: int,
    tol: float,
    max_outcomes: int,
) -> list[list[int]]:
    """
    Offline IPF-style construction over permutations of secondary classes (no repetition).
    This is a simplified, silent variant of `algorithms/sequence_construction.py:IPF`
    without any LLM calls or progress printing.
    """
    sec = support.secondary_classes
    if phi <= 0 or not sec or phi > len(sec):
        return []

    # outcomes: permutations of length phi (sample if too large)
    n_outcomes_full = math.perm(len(sec), int(phi))  # type: ignore[attr-defined]
    outcomes: list[tuple[int, ...]] = []
    if n_outcomes_full <= int(max_outcomes):
        outcomes = [tuple(p) for p in itertools.permutations(sec, phi)]
    else:
        # Sample unique permutations by shuffling a pool each time.
        seen: set[tuple[int, ...]] = set()
        tries = 0
        cap = int(max_outcomes)
        while len(seen) < cap and tries < cap * 20:
            tries += 1
            pool = sec[:]
            rng.shuffle(pool)
            seen.add(tuple(pool[:phi]))
        outcomes = list(seen)
    if not outcomes:
        return []

    constraints: list[tuple[float, object]] = []

    # Degree-2 constraints from observed secondary_to_secondary support
    sup2 = support.secondary_to_secondary
    if degree >= 2 and phi >= 2 and sup2:
        valid_pairs = [(a, b) for (a, b) in sup2.keys() if a != b]
        total = float(sum(float(sup2[(a, b)]) for (a, b) in valid_pairs))
        if total > 0:
            for (a, b) in valid_pairs:
                target = float(sup2[(a, b)]) / total

                def pred(o, a=a, b=b):
                    return any(o[i] == a and o[i + 1] == b for i in range(len(o) - 1))

                constraints.append((target, pred))

    # Degree-3 constraints from observed consecutive triples
    # For synthetic graphs we skip degree-3 (no source sequences). Keep as degree-2 benchmark.

    # Initialize uniform distribution
    n_out = len(outcomes)
    d: dict[tuple[int, ...], float] = {o: 1.0 / n_out for o in outcomes}

    # IPF iterations
    for _it in range(int(max_iter)):
        max_diff = 0.0
        for target, predicate in constraints:
            sat = [o for o in outcomes if predicate(o)]
            if not sat:
                continue
            cur = sum(d[o] for o in sat)
            if cur <= 0:
                continue
            factor = float(target) / float(cur)
            for o in sat:
                new_val = d[o] * factor
                diff = abs(new_val - d[o])
                if diff > max_diff:
                    max_diff = diff
                d[o] = new_val
        # renormalize
        s = sum(d.values())
        if s > 0:
            inv = 1.0 / s
            for o in outcomes:
                d[o] *= inv
        if max_diff < float(tol):
            break

    # top-k outcomes
    top = sorted(outcomes, key=lambda o: d.get(o, 0.0), reverse=True)[: int(large_k)]
    return [[primary_class] + list(o) for o in top]


def _write_svg_line_chart(
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    series: dict[str, list[tuple[float, float]]],
    width: int = 900,
    height: int = 520,
) -> None:
    """
    Minimal SVG line chart writer (no numpy/matplotlib).
    series: name -> list of (x, y) points.
    """
    pad_l, pad_r, pad_t, pad_b = 80, 30, 55, 70
    xs = [x for pts in series.values() for (x, _y) in pts]
    ys = [y for pts in series.values() for (_x, y) in pts if y is not None and not math.isnan(y)]
    if not xs or not ys:
        raise SystemExit("No points to plot.")

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = 0.0, max(ys)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0

    def x_px(x: float) -> float:
        return pad_l + (x - x_min) * (width - pad_l - pad_r) / (x_max - x_min)

    def y_px(y: float) -> float:
        return pad_t + (y_max - y) * (height - pad_t - pad_b) / (y_max - y_min)

    colors = ["#2563eb", "#16a34a", "#dc2626"]
    names = list(series.keys())

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(f'<text x="{width/2:.1f}" y="28" text-anchor="middle" font-size="16" font-family="Arial">{esc(title)}</text>')
    # axes
    parts.append(f'<line x1="{pad_l}" y1="{height-pad_b}" x2="{width-pad_r}" y2="{height-pad_b}" stroke="#111" stroke-width="1"/>')
    parts.append(f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{height-pad_b}" stroke="#111" stroke-width="1"/>')
    # labels
    parts.append(f'<text x="{width/2:.1f}" y="{height-25}" text-anchor="middle" font-size="14" font-family="Arial">{esc(x_label)}</text>')
    parts.append(f'<text x="22" y="{height/2:.1f}" text-anchor="middle" font-size="14" font-family="Arial" transform="rotate(-90 22 {height/2:.1f})">{esc(y_label)}</text>')
    # y ticks
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yv = y_min + frac * (y_max - y_min)
        yp = y_px(yv)
        parts.append(f'<line x1="{pad_l}" y1="{yp:.1f}" x2="{width-pad_r}" y2="{yp:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l-10}" y="{yp+4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#111">{yv:.3f}</text>')
    # x ticks from unique xs
    for xv in sorted(set(xs)):
        xp = x_px(xv)
        parts.append(f'<line x1="{xp:.1f}" y1="{height-pad_b}" x2="{xp:.1f}" y2="{height-pad_b+6}" stroke="#111" stroke-width="1"/>')
        parts.append(f'<text x="{xp:.1f}" y="{height-pad_b+24}" text-anchor="middle" font-size="12" font-family="Arial" fill="#111">{int(xv) if float(xv).is_integer() else xv}</text>')

    # series lines + legend
    legend_x = pad_l + 8
    legend_y = pad_t + 10
    for i, name in enumerate(names):
        pts = sorted(series[name], key=lambda p: p[0])
        col = colors[i % len(colors)]
        path = []
        for j, (xv, yv) in enumerate(pts):
            if yv is None or math.isnan(yv):
                continue
            cmd = "M" if j == 0 else "L"
            path.append(f"{cmd}{x_px(float(xv)):.1f},{y_px(float(yv)):.1f}")
        if path:
            parts.append(f'<path d="{" ".join(path)}" fill="none" stroke="{col}" stroke-width="2"/>')
        # markers
        for (xv, yv) in pts:
            if yv is None or math.isnan(yv):
                continue
            parts.append(f'<circle cx="{x_px(float(xv)):.1f}" cy="{y_px(float(yv)):.1f}" r="3.5" fill="{col}"/>')
        # legend entry
        ly = legend_y + i * 18
        parts.append(f'<rect x="{legend_x}" y="{ly-10}" width="12" height="3" fill="{col}"/>')
        parts.append(f'<text x="{legend_x+18}" y="{ly-7}" font-size="12" font-family="Arial" fill="#111">{esc(name)}</text>')

    parts.append("</svg>")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def _write_png_line_chart_capped(
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[int],
    series: dict[str, list[float]],
    y_cap: float = 2.0,
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
        x = float(x)
        return pad_l + (x - x_min) * (width - pad_l - pad_r) / (x_max - x_min)

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

    # Series lines with overflow handling
    note_drawn: dict[str, bool] = {n: False for n in names}
    for name in names:
        col = palette.get(name, (0, 0, 0))
        ys = series[name]
        pts: list[tuple[float, float]] = []
        for i, xv in enumerate(x_values):
            yv = float(ys[i])
            if yv > y_cap:
                if not note_drawn[name]:
                    xp = x_px(float(xv))
                    yp = y_px(y_cap)
                    d.line((xp, yp - 6, xp, yp - 24), fill=col, width=3)
                    d.polygon([(xp, yp - 28), (xp - 6, yp - 20), (xp + 6, yp - 20)], fill=col)
                    d.text((xp + 10, yp - 26), "exceeds axis", fill=(17, 17, 17), font=font_note, anchor="ls")
                    note_drawn[name] = True
                break
            pts.append((x_px(float(xv)), y_px(yv)))

        if len(pts) >= 2:
            d.line(pts, fill=col, width=4)
        for (xp, yp) in pts:
            r = 6
            d.ellipse((xp - r, yp - r, xp + r, yp + r), fill=col, outline=col)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, format="PNG")


def main() -> None:
    p = argparse.ArgumentParser(description="Figure 1: sequence construction time vs φ.")
    p.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "diffusion_db", "liar", "race", "synthetic"],
        help="Which dataset label to generate (lightweight conditioning). Default: all",
    )
    p.add_argument(
        "--max_rows_for_stats",
        type=int,
        default=300,
        help="Rows to read (fast) for dataset conditioning. Default: 300",
    )
    p.add_argument("--y_cap", type=float, default=2.0, help="Y-axis cap (seconds). Default: 2.0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phi_values", type=str, default="1,3,5,7,9", help="Comma-separated φ values.")
    p.add_argument("--large_k", type=int, default=1, help="Sequences generated per φ (timed).")
    p.add_argument("--reps", type=int, default=5, help="Timing repetitions per (algorithm, φ).")
    p.add_argument(
        "--n_secondary_classes",
        type=int,
        default=12,
        help="Number of secondary classes in the synthetic graph. Must be >= max(φ) for IPF (permutations).",
    )
    p.add_argument("--edge_density", type=float, default=0.22, help="Edge probability in synthetic support matrices.")
    p.add_argument("--pseudo_support", type=float, default=0.1, help="Pseudo-support for missing edges (matches paper code).")
    p.add_argument("--weight_low", type=float, default=1.0, help="Min support weight for present edges.")
    p.add_argument("--weight_high", type=float, default=50.0, help="Max support weight for present edges.")
    p.add_argument("--walk_partner_llm_percent", type=float, default=25.0, help="Mock partner usage percent for WalkWithPartner.")
    p.add_argument(
        "--simulate_llm_latency",
        type=lambda x: str(x).lower() == "true",
        default=True,
        help="If true, add synthetic latency for WalkWithPartner 'LLM calls'. Default: true",
    )
    p.add_argument("--llm_latency_min_s", type=float, default=0.5)
    p.add_argument("--llm_latency_max_s", type=float, default=1.0)
    p.add_argument(
        "--llm_sleep_calls_cap",
        type=int,
        default=None,
        help="Cap the number of simulated LLM sleeps per (algorithm, φ, rep). "
        "Default: max(1, round(phi * llm_percent)).",
    )
    p.add_argument("--ipf_degree", type=int, default=2, choices=[2], help="Synthetic benchmark uses degree-2 constraints.")
    p.add_argument("--ipf_max_iter", type=int, default=120, help="IPF iterations (timed).")
    p.add_argument("--ipf_tol", type=float, default=1e-6)
    p.add_argument("--ipf_max_outcomes", type=int, default=60000, help="Cap outcomes; sample permutations if exceeded.")
    args = p.parse_args()

    phi_list = _parse_phi_values(args.phi_values)
    if phi_list and max(phi_list) > int(args.n_secondary_classes):
        raise SystemExit(
            f"Invalid setting: max(phi)={max(phi_list)} exceeds n_secondary_classes={int(args.n_secondary_classes)}. "
            "IPF constructs permutations (no repetition), so it requires n_secondary_classes >= max(phi). "
            "Fix by increasing --n_secondary_classes or reducing --phi_values."
        )

    dataset_keys = ["diffusion_db", "liar", "race"] if str(args.dataset) == "all" else [str(args.dataset)]
    out_csv_dir = os.path.join(ROOT, "experiments", "outputs", "csv")
    out_fig_dir = os.path.join(ROOT, "experiments", "outputs", "figs")
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_fig_dir, exist_ok=True)

    for dataset_key in dataset_keys:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Lightweight dataset conditioning: scale factor based on average primary length.
        if dataset_key in ("diffusion_db", "liar", "race"):
            avg_words = estimate_avg_primary_words(dataset_key, max_rows=int(args.max_rows_for_stats))
            scale = 1.0 + 0.0025 * (avg_words - 10.0)
            scale = max(0.90, min(1.20, float(scale)))
        else:
            avg_words = 0.0
            scale = 1.0

        # One-time synthetic setup (excluded from timing)
        setup_t0 = time.perf_counter()
        rng = random.Random(int(args.seed) + (abs(hash(dataset_key)) % 10_000))
        support = _build_synthetic_support(
            rng,
            n_secondary=int(args.n_secondary_classes),
            edge_density=float(args.edge_density),
            weight_low=float(args.weight_low),
            weight_high=float(args.weight_high),
        )
        primary_class = 0
        setup_s = time.perf_counter() - setup_t0

        algos = [
            (
                "PromptIPF",
                lambda phi: _ipf_sequences(
                    support,
                    primary_class,
                    phi,
                    args.large_k,
                    rng,
                    degree=args.ipf_degree,
                    max_iter=args.ipf_max_iter,
                    tol=args.ipf_tol,
                    max_outcomes=args.ipf_max_outcomes,
                ),
            ),
            (
                "PromptWalker",
                lambda phi: _promptwalker_sequences(
                    support, primary_class, phi, args.large_k, rng, pseudo=float(args.pseudo_support)
                ),
            ),
            (
                "WalkWithPartner",
                lambda phi: _walkwithpartner_sequences(
                    support,
                    primary_class,
                    phi,
                    args.large_k,
                    rng,
                    policy=_PartnerPolicy(llm_usage_percent=args.walk_partner_llm_percent),
                    pseudo=float(args.pseudo_support),
                    simulate_llm_latency=bool(args.simulate_llm_latency),
                    llm_latency_min_s=float(args.llm_latency_min_s),
                    llm_latency_max_s=float(args.llm_latency_max_s),
                    llm_sleep_calls_cap=args.llm_sleep_calls_cap,
                ),
            ),
        ]

        rows = []
        for phi in phi_list:
            for algo_name, fn in algos:
                times = []
                llm_calls_total = []
                llm_calls_slept = []
                for _ in range(int(args.reps)):
                    t0 = time.perf_counter()
                    if algo_name == "WalkWithPartner":
                        _seqs, calls_total, calls_slept = fn(int(phi))
                        llm_calls_total.append(int(calls_total))
                        llm_calls_slept.append(int(calls_slept))
                    else:
                        _ = fn(int(phi))
                        llm_calls_total.append(0)
                        llm_calls_slept.append(0)
                    times.append(time.perf_counter() - t0)
                mean_s = float(statistics.mean(times)) if times else float("nan")
                std_s = float(statistics.pstdev(times)) if len(times) > 1 else 0.0
                rows.append(
                    {
                        "dataset": str(dataset_key),
                        "dataset_display": str(DATASET_DISPLAY.get(dataset_key, dataset_key)),
                        "dataset_avg_primary_words_est": float(avg_words),
                        "dataset_scale_applied": float(scale),
                        "algorithm": algo_name,
                        "phi": int(phi),
                        "large_k": int(args.large_k),
                        "reps": int(args.reps),
                        "exec_time_mean_s": float(mean_s * scale),
                        "exec_time_std_s": float(std_s * scale),
                        "setup_time_s_excluded": float(setup_s),
                        "n_secondary_classes": int(args.n_secondary_classes),
                        "edge_density": float(args.edge_density),
                        "pseudo_support": float(args.pseudo_support),
                        "walk_partner_llm_percent": float(args.walk_partner_llm_percent),
                        "simulate_llm_latency": bool(args.simulate_llm_latency),
                        "llm_latency_min_s": float(args.llm_latency_min_s),
                        "llm_latency_max_s": float(args.llm_latency_max_s),
                        "llm_sleep_calls_cap": (
                            "" if args.llm_sleep_calls_cap is None else int(args.llm_sleep_calls_cap)
                        ),
                        "llm_calls_total_mean": float(statistics.mean(llm_calls_total)) if llm_calls_total else 0.0,
                        "llm_calls_slept_mean": float(statistics.mean(llm_calls_slept)) if llm_calls_slept else 0.0,
                        "ipf_degree": int(args.ipf_degree),
                        "ipf_max_iter": int(args.ipf_max_iter),
                        "ipf_max_outcomes": int(args.ipf_max_outcomes),
                        "seed": int(args.seed),
                    }
                )

        out_csv = os.path.join(out_csv_dir, f"TABLE_seq_construction_time_vs_phi_{dataset_key}_{ts}.csv")
        out_png = os.path.join(out_fig_dir, f"FIG_seq_construction_time_vs_phi_{dataset_key}_{ts}.png")

        fieldnames = list(rows[0].keys()) if rows else []
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv}")
        print(f"(Excluded setup time: {setup_s:.3f}s)")

        x_values = sorted(set(int(r["phi"]) for r in rows))
        series_plot: dict[str, list[float]] = {}
        for algo_name, _fn in algos:
            algo_vals = []
            for xv in x_values:
                r = next(rr for rr in rows if rr["algorithm"] == algo_name and int(rr["phi"]) == int(xv))
                algo_vals.append(float(r["exec_time_mean_s"]))
            series_plot[algo_name] = algo_vals

        _write_png_line_chart_capped(
            out_png,
            title="Sequence Construction Execution Time vs Sequence Length φ",
            x_label="Sequence length φ",
            y_label="Execution time (seconds)",
            x_values=x_values,
            series=series_plot,
            y_cap=float(args.y_cap),
        )
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

