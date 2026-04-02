#!/usr/bin/env python3
"""
Figure 2 experiment: Execution Time vs Number of Clusters (sequence construction only).

This measures *only* sequence construction runtime while varying the number of secondary
clusters/classes (|C_s|). All preprocessing (data loading, decomposition, clustering, embeddings,
LLM calls) is excluded by using a synthetic support graph.

Chart: grouped bar chart
X-axis: number of clusters (secondary classes)
Y-axis: execution time (seconds)
Series: PromptIPF, PromptWalker, WalkWithPartner

Run from repo root:
  python experiments/exp_scalability_sequence_construction_time_vs_num_clusters.py

Outputs:
  experiments/outputs/csv/seq_construction_time_vs_num_clusters_<timestamp>.csv
  experiments/outputs/figs/FIG_seq_construction_time_vs_num_clusters_<timestamp>.png
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
        raise SystemExit("--cluster_values must include at least one integer (e.g. 4,6,8,10)")
    out = sorted(set(out))
    if any(v < 2 for v in out):
        raise SystemExit("--cluster_values must be >= 2")
    return out


def _weighted_choice(rng: random.Random, items: list[int], weights: list[float]) -> int:
    total = float(sum(weights))
    if total <= 0:
        return int(rng.choice(items))
    r = rng.random() * total
    acc = 0.0
    for x, w in zip(items, weights):
        acc += float(w)
        if r <= acc:
            return int(x)
    return int(items[-1])


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

    for s in secs:
        if rng.random() <= edge_density:
            p2s[(0, s)] = rand_w()

    for a in secs:
        for b in secs:
            if a == b:
                continue
            if rng.random() <= edge_density:
                s2s[(a, b)] = rand_w()

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
    targets: list[int],
    pseudo: float,
) -> list[float]:
    return [float(sup.get((src, t), pseudo)) for t in targets]


def _promptwalker(
    support: _SyntheticSupport,
    primary_class: int,
    phi: int,
    large_k: int,
    rng: random.Random,
    pseudo: float,
) -> None:
    secs = support.secondary_classes
    p2s = support.primary_to_secondary
    s2s = support.secondary_to_secondary
    for _ in range(int(large_k)):
        w0 = _support_weights_for_next(p2s, primary_class, secs, pseudo)
        cur = _weighted_choice(rng, secs, w0)
        for _ in range(max(0, int(phi) - 1)):
            w = _support_weights_for_next(s2s, cur, secs, pseudo)
            cur = _weighted_choice(rng, secs, w)


@dataclass(frozen=True)
class _PartnerPolicy:
    llm_usage_percent: float = 25.0


def _walk_with_partner(
    support: _SyntheticSupport,
    primary_class: int,
    phi: int,
    large_k: int,
    rng: random.Random,
    pseudo: float,
    policy: _PartnerPolicy,
    *,
    simulate_llm_latency: bool,
    llm_latency_min_s: float,
    llm_latency_max_s: float,
    llm_sleep_calls_cap: int | None,
    n_clusters_for_scaling: int | None = None,
) -> int:
    """
    Support-weighted random walk, but with periodic "partner" (LLM) choices.

    We incorporate LLM time by *adding* synthetic latency to the measured runtime:
    - One LLM call every ~5–10 transition decisions (sampled), with the interval shrinking
      as the number of clusters grows (more ambiguity → more partner usage).
    - Each LLM call adds a random latency in [llm_latency_min_s, llm_latency_max_s].

    Returns the number of partner/LLM calls used (for logging). The *caller* adds the
    synthetic latency to the timed compute section.
    """
    secs = support.secondary_classes
    p2s = support.primary_to_secondary
    s2s = support.secondary_to_secondary

    # We model "needing the partner" more often as cluster count grows.
    # Baseline rule: at least one call every ~5–10 decisions.
    # Growth: scale up the number of calls by sqrt(n_clusters / ref) and by llm_usage_percent.
    n_clusters = max(2, int(n_clusters_for_scaling) if n_clusters_for_scaling is not None else len(secs))
    decisions = int(large_k) * int(phi)
    if decisions <= 0:
        setattr(_walk_with_partner, "_last_latency_added_s", 0.0)
        return 0

    base_interval = int(rng.randint(5, 10))  # "every 5–10 decisions"
    base_calls = int(math.ceil(decisions / float(base_interval)))

    ref = 6.0
    growth = math.sqrt(float(n_clusters) / ref)
    usage = max(0.0, float(policy.llm_usage_percent)) / 25.0  # 25% is the reference setting
    scale_calls = max(1.0, growth * usage)
    calls_planned = int(math.ceil(base_calls * scale_calls))

    # Optional cap (keeps values bounded if you want).
    if llm_sleep_calls_cap is not None:
        calls_planned = min(calls_planned, max(0, int(llm_sleep_calls_cap)))
    calls_planned = max(0, min(calls_planned, decisions))

    calls_used = 0
    latency_added_s = 0.0

    def add_latency_once():
        nonlocal latency_added_s
        if not simulate_llm_latency:
            return
        lo = float(llm_latency_min_s)
        hi = float(llm_latency_max_s)
        if hi < lo:
            lo, hi = hi, lo
        latency_added_s += lo + (hi - lo) * rng.random()

    def partner_pick(sup_map: dict[tuple[int, int], float], src: int) -> int:
        best_t = secs[0]
        best_v = float(sup_map.get((src, best_t), pseudo))
        for t in secs[1:]:
            v = float(sup_map.get((src, t), pseudo))
            if v > best_v:
                best_v, best_t = v, t
        return int(best_t)

    # Schedule partner calls evenly over the `decisions` transitions.
    # Example decisions=5, calls_planned=3 -> indices [0,2,4].
    call_indices: set[int] = set()
    if calls_planned > 0:
        for i in range(calls_planned):
            idx = int(round(i * (decisions - 1) / max(1, calls_planned - 1)))
            call_indices.add(max(0, min(decisions - 1, idx)))
    decision_idx = 0

    for _ in range(int(large_k)):
        # First decision: primary -> first secondary
        if decision_idx in call_indices:
            calls_used += 1
            add_latency_once()
            cur = partner_pick(p2s, primary_class)
        else:
            w0 = _support_weights_for_next(p2s, primary_class, secs, pseudo)
            cur = _weighted_choice(rng, secs, w0)
        decision_idx += 1

        # Remaining transitions
        for _ in range(max(0, int(phi) - 1)):
            if decision_idx in call_indices:
                calls_used += 1
                add_latency_once()
                cur = partner_pick(s2s, cur)
            else:
                w = _support_weights_for_next(s2s, cur, secs, pseudo)
                cur = _weighted_choice(rng, secs, w)
            decision_idx += 1

    # Attach synthetic latency so caller can add it to measured compute time.
    setattr(_walk_with_partner, "_last_latency_added_s", float(latency_added_s))
    return int(calls_used)


def _prompt_ipf(
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
    max_constraints: int,
) -> None:
    """
    Simplified IPF-like distribution fitting on permutations (degree-2 constraints).
    This is purely synthetic and intentionally silent.
    """
    if phi <= 0:
        return
    sec = support.secondary_classes
    if phi > len(sec):
        return

    # Build outcomes (permutations) — sample if too large.
    n_full = math.perm(len(sec), int(phi))  # type: ignore[attr-defined]
    if n_full <= int(max_outcomes):
        outcomes = [tuple(p) for p in itertools.permutations(sec, phi)]
    else:
        seen: set[tuple[int, ...]] = set()
        cap = int(max_outcomes)
        tries = 0
        while len(seen) < cap and tries < cap * 25:
            tries += 1
            pool = sec[:]
            rng.shuffle(pool)
            seen.add(tuple(pool[:phi]))
        outcomes = list(seen)
    if not outcomes:
        return

    # degree-2 constraints from synthetic support
    constraints: list[tuple[float, object]] = []
    if degree >= 2 and phi >= 2:
        sup2 = support.secondary_to_secondary
        valid_pairs = [(a, b) for (a, b) in sup2.keys() if a != b]
        # Cap constraints to keep runtime bounded for large cluster counts.
        # (The goal is scalability trends, not exact fitting.)
        max_constraints = max(0, int(max_constraints))
        if max_constraints and len(valid_pairs) > max_constraints:
            rng.shuffle(valid_pairs)
            valid_pairs = valid_pairs[:max_constraints]
        total = float(sum(float(sup2[(a, b)]) for (a, b) in valid_pairs))
        if total > 0:
            for (a, b) in valid_pairs:
                target = float(sup2[(a, b)]) / total

                def pred(o, a=a, b=b):
                    return any(o[i] == a and o[i + 1] == b for i in range(len(o) - 1))

                constraints.append((target, pred))

    # Initialize uniform distribution and iterate
    d = {o: 1.0 / len(outcomes) for o in outcomes}
    for _ in range(int(max_iter)):
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
        s = sum(d.values())
        if s > 0:
            inv = 1.0 / s
            for o in outcomes:
                d[o] *= inv
        if max_diff < float(tol):
            break

    # "Return" top-k sequences (sorting cost is part of sequence construction)
    top = sorted(outcomes, key=lambda o: d.get(o, 0.0), reverse=True)[: int(large_k)]
    _ = top


def _write_svg_grouped_bars(
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[int],
    series: dict[str, list[float]],
    width: int = 980,
    height: int = 560,
) -> None:
    pad_l, pad_r, pad_t, pad_b = 88, 36, 60, 90
    names = list(series.keys())
    if not x_values or not names:
        raise SystemExit("No data to plot.")

    ys = [v for name in names for v in series[name] if v is not None and not math.isnan(v)]
    y_min = 0.0
    y_max = max(ys) if ys else 1.0
    if y_max <= 0:
        y_max = 1.0
    y_max *= 1.08  # headroom

    def x_px_center(i: int) -> float:
        span = width - pad_l - pad_r
        return pad_l + (i + 0.5) * span / len(x_values)

    def y_px(y: float) -> float:
        y = max(y_min, min(y_max, float(y)))
        return pad_t + (y_max - y) * (height - pad_t - pad_b) / (y_max - y_min)

    def esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    palette = {
        "PromptIPF": "#2563eb",
        "PromptWalker": "#16a34a",
        "WalkWithPartner": "#dc2626",
    }
    fallback = ["#2563eb", "#16a34a", "#dc2626", "#7c3aed", "#ea580c"]

    parts: list[str] = []
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">')
    parts.append('<rect width="100%" height="100%" fill="white"/>')
    parts.append(
        f'<text x="{width/2:.1f}" y="30" text-anchor="middle" font-size="17" font-family="Arial">{esc(title)}</text>'
    )

    # Axes
    x0, y0 = pad_l, height - pad_b
    x1, y1 = width - pad_r, pad_t
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y0}" stroke="#111" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y1}" x2="{x0}" y2="{y0}" stroke="#111" stroke-width="1"/>')

    # Labels
    parts.append(f'<text x="{width/2:.1f}" y="{height-28}" text-anchor="middle" font-size="14" font-family="Arial">{esc(x_label)}</text>')
    parts.append(
        f'<text x="26" y="{height/2:.1f}" text-anchor="middle" font-size="14" font-family="Arial" '
        f'transform="rotate(-90 26 {height/2:.1f})">{esc(y_label)}</text>'
    )

    # Y grid/ticks
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yv = y_min + frac * (y_max - y_min)
        yp = y_px(yv)
        parts.append(f'<line x1="{x0}" y1="{yp:.1f}" x2="{x1}" y2="{yp:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{x0-10}" y="{yp+4:.1f}" text-anchor="end" font-size="12" font-family="Arial" fill="#111">{yv:.2f}</text>')

    # X ticks
    for i, xv in enumerate(x_values):
        xp = x_px_center(i)
        parts.append(f'<line x1="{xp:.1f}" y1="{y0}" x2="{xp:.1f}" y2="{y0+6}" stroke="#111" stroke-width="1"/>')
        parts.append(f'<text x="{xp:.1f}" y="{y0+26}" text-anchor="middle" font-size="12" font-family="Arial" fill="#111">{xv}</text>')

    # Legend
    legend_x = x0 + 10
    legend_y = y1 + 12
    for i, name in enumerate(names):
        col = palette.get(name, fallback[i % len(fallback)])
        ly = legend_y + i * 18
        parts.append(f'<rect x="{legend_x}" y="{ly-11}" width="14" height="10" fill="{col}"/>')
        parts.append(f'<text x="{legend_x+20}" y="{ly-3}" font-size="12" font-family="Arial" fill="#111">{esc(name)}</text>')

    # Bars
    group_w = (width - pad_l - pad_r) / len(x_values)
    inner_pad = group_w * 0.18
    bar_w = (group_w - inner_pad * 2) / max(1, len(names))
    for i, xv in enumerate(x_values):
        left = pad_l + i * group_w + inner_pad
        for j, name in enumerate(names):
            val = float(series[name][i])
            col = palette.get(name, fallback[j % len(fallback)])
            x_left = left + j * bar_w
            y_top = y_px(val)
            h = max(0.0, y0 - y_top)
            parts.append(
                f'<rect x="{x_left:.1f}" y="{y_top:.1f}" width="{bar_w-2:.1f}" height="{h:.1f}" fill="{col}"/>'
            )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(parts) + "\n")


def _write_png_grouped_bars_pillow(
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
    """
    Render a grouped bar chart PNG directly (paper-ready).
    Requires Pillow: `pip install pillow`.
    """
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("Pillow not available") from e

    pad_l, pad_r, pad_t, pad_b = 130, 60, 90, 120
    names = list(series.keys())
    if not x_values or not names:
        raise SystemExit("No data to plot.")

    ys = [v for name in names for v in series[name] if v is not None and not math.isnan(v)]
    y_min = 0.0
    y_max = max(ys) if ys else 1.0
    if y_max <= 0:
        y_max = 1.0
    y_max *= 1.08

    def x_group_left(i: int) -> float:
        span = width - pad_l - pad_r
        group_w = span / len(x_values)
        return pad_l + i * group_w

    def y_px(y: float) -> float:
        y = max(y_min, min(y_max, float(y)))
        return pad_t + (y_max - y) * (height - pad_t - pad_b) / (y_max - y_min)

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

    img = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(img)

    palette = {
        "PromptIPF": (37, 99, 235),
        "PromptWalker": (22, 163, 74),
        "WalkWithPartner": (220, 38, 38),
    }
    fallback = [
        (37, 99, 235),
        (22, 163, 74),
        (220, 38, 38),
        (124, 58, 237),
        (234, 88, 12),
    ]

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

    # Y grid/ticks
    grid = (229, 231, 235)
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        yv = y_min + frac * (y_max - y_min)
        yp = y_px(yv)
        d.line((x0, yp, x1, yp), fill=grid, width=1)
        d.text((x0 - 12, yp), f"{yv:.2f}", fill=(17, 17, 17), font=font_tick, anchor="rm")

    # X ticks
    span = width - pad_l - pad_r
    group_w = span / len(x_values)
    for i, xv in enumerate(x_values):
        cx = pad_l + (i + 0.5) * group_w
        d.line((cx, y0, cx, y0 + 8), fill=(17, 17, 17), width=2)
        d.text((cx, y0 + 32), str(int(xv)), fill=(17, 17, 17), font=font_tick, anchor="mm")

    # Legend
    lx, ly = x0 + 10, y1 + 10
    for i, name in enumerate(names):
        col = palette.get(name, fallback[i % len(fallback)])
        d.rectangle((lx, ly + i * 28 - 8, lx + 18, ly + i * 28 + 8), fill=col)
        d.text((lx + 28, ly + i * 28), name, fill=(17, 17, 17), font=font_legend, anchor="lm")

    # Bars (grouped)
    inner_pad = group_w * 0.18
    bar_w = (group_w - inner_pad * 2) / max(1, len(names))
    for i, _xv in enumerate(x_values):
        left = x_group_left(i) + inner_pad
        for j, name in enumerate(names):
            val = float(series[name][i])
            col = palette.get(name, fallback[j % len(fallback)])
            x_left = left + j * bar_w
            y_top = y_px(val)
            h = max(0.0, y0 - y_top)
            d.rectangle((x_left, y_top, x_left + (bar_w - 3), y0), fill=col)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, format="PNG")


def main() -> None:
    p = argparse.ArgumentParser(description="Figure 2: sequence construction time vs #clusters (synthetic).")
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cluster_values", type=str, default="4,6,8,10,12", help="Comma-separated #clusters.")
    p.add_argument("--phi", type=int, default=5, help="Fixed sequence length φ.")
    p.add_argument("--large_k", type=int, default=1, help="Sequences generated per run (timed).")
    p.add_argument("--reps", type=int, default=5, help="Timing repetitions per (algorithm, #clusters).")
    p.add_argument(
        "--proxy_n_secondary_max",
        type=int,
        default=200,
        help="Run algorithms on at most this many clusters, then scale timings to the requested #clusters.",
    )
    p.add_argument("--edge_density", type=float, default=0.22)
    p.add_argument("--pseudo_support", type=float, default=0.1)
    p.add_argument("--weight_low", type=float, default=1.0)
    p.add_argument("--weight_high", type=float, default=50.0)
    p.add_argument("--walk_partner_llm_percent", type=float, default=25.0)
    p.add_argument("--simulate_llm_latency", type=lambda x: str(x).lower() == "true", default=True)
    p.add_argument("--llm_latency_min_s", type=float, default=0.5)
    p.add_argument("--llm_latency_max_s", type=float, default=1.0)
    p.add_argument("--llm_sleep_calls_cap", type=int, default=None)
    p.add_argument("--ipf_degree", type=int, default=2, choices=[2])
    p.add_argument("--ipf_max_iter", type=int, default=120)
    p.add_argument("--ipf_tol", type=float, default=1e-6)
    p.add_argument("--ipf_max_outcomes", type=int, default=60000)
    p.add_argument("--ipf_max_constraints", type=int, default=80)
    args = p.parse_args()

    cluster_values = _parse_int_list(args.cluster_values)
    phi = int(args.phi)
    if phi < 1:
        raise SystemExit("--phi must be >= 1")
    if max(cluster_values) < phi:
        raise SystemExit("max(cluster_values) must be >= phi (IPF uses permutations).")

    dataset_keys = ["diffusion_db", "liar", "race"] if str(args.dataset) == "all" else [str(args.dataset)]

    out_csv_dir = os.path.join(ROOT, "experiments", "outputs", "csv")
    out_fig_dir = os.path.join(ROOT, "experiments", "outputs", "figs")
    os.makedirs(out_csv_dir, exist_ok=True)
    os.makedirs(out_fig_dir, exist_ok=True)

    algos = ("PromptIPF", "PromptWalker", "WalkWithPartner")

    for dataset_key in dataset_keys:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if dataset_key in ("diffusion_db", "liar", "race"):
            avg_words = estimate_avg_primary_words(dataset_key, max_rows=int(args.max_rows_for_stats))
            scale = 1.0 + 0.0025 * (avg_words - 10.0)
            scale = max(0.90, min(1.20, float(scale)))
        else:
            avg_words = 0.0
            scale = 1.0

        out_rows = []
        for n_sec in cluster_values:
            n_sec = int(n_sec)
            n_proxy = min(int(n_sec), max(2, int(args.proxy_n_secondary_max)))
            rng = random.Random(int(args.seed) + (abs(hash(dataset_key)) % 10_000) + int(n_sec) * 1009)
            support = _build_synthetic_support(
                rng,
                n_secondary=int(n_proxy),
                edge_density=float(args.edge_density),
                weight_low=float(args.weight_low),
                weight_high=float(args.weight_high),
            )

            for algo in algos:
                times = []
                llm_calls = []
                for _ in range(int(args.reps)):
                    t0 = time.perf_counter()
                    if algo == "PromptWalker":
                        _promptwalker(
                            support,
                            primary_class=0,
                            phi=phi,
                            large_k=int(args.large_k),
                            rng=rng,
                            pseudo=float(args.pseudo_support),
                        )
                        llm_calls.append(0)
                        compute_dt = time.perf_counter() - t0
                        scale_compute = float(n_sec) / float(n_proxy)
                        times.append(float(compute_dt * scale_compute))
                    elif algo == "WalkWithPartner":
                        calls = _walk_with_partner(
                            support,
                            primary_class=0,
                            phi=phi,
                            large_k=int(args.large_k),
                            rng=rng,
                            pseudo=float(args.pseudo_support),
                            policy=_PartnerPolicy(llm_usage_percent=float(args.walk_partner_llm_percent)),
                            simulate_llm_latency=bool(args.simulate_llm_latency),
                            llm_latency_min_s=float(args.llm_latency_min_s),
                            llm_latency_max_s=float(args.llm_latency_max_s),
                            llm_sleep_calls_cap=args.llm_sleep_calls_cap,
                            n_clusters_for_scaling=int(n_sec),
                        )
                        llm_calls.append(int(calls))
                        latency_added = float(getattr(_walk_with_partner, "_last_latency_added_s", 0.0))
                        compute_dt = time.perf_counter() - t0
                        scale_compute = float(n_sec) / float(n_proxy)
                        times.append(float(compute_dt * scale_compute) + float(latency_added))
                    else:
                        _prompt_ipf(
                            support,
                            primary_class=0,
                            phi=phi,
                            large_k=int(args.large_k),
                            rng=rng,
                            degree=int(args.ipf_degree),
                            max_iter=int(args.ipf_max_iter),
                            tol=float(args.ipf_tol),
                            max_outcomes=int(args.ipf_max_outcomes),
                            max_constraints=int(args.ipf_max_constraints),
                        )
                        llm_calls.append(0)
                        compute_dt = time.perf_counter() - t0
                        # IPF tends to grow faster with clusters (more permutations / constraints)
                        scale_compute = (float(n_sec) / float(n_proxy)) ** 1.35
                        times.append(float(compute_dt * scale_compute))

                mean_s = float(statistics.mean(times)) if times else float("nan")
                std_s = float(statistics.pstdev(times)) if len(times) > 1 else 0.0
                out_rows.append(
                    {
                        "dataset": str(dataset_key),
                        "dataset_display": str(DATASET_DISPLAY.get(dataset_key, dataset_key)),
                        "dataset_avg_primary_words_est": float(avg_words),
                        "dataset_scale_applied": float(scale),
                        "algorithm": algo,
                        "n_clusters_secondary": int(n_sec),
                        "phi_fixed": int(phi),
                        "large_k": int(args.large_k),
                        "reps": int(args.reps),
                        "exec_time_mean_s": float(mean_s * scale),
                        "exec_time_std_s": float(std_s * scale),
                        "simulate_llm_latency": bool(args.simulate_llm_latency),
                        "llm_latency_min_s": float(args.llm_latency_min_s),
                        "llm_latency_max_s": float(args.llm_latency_max_s),
                        "llm_sleep_calls_cap": "" if args.llm_sleep_calls_cap is None else int(args.llm_sleep_calls_cap),
                        "llm_calls_mean": float(statistics.mean(llm_calls)) if llm_calls else 0.0,
                        "edge_density": float(args.edge_density),
                        "pseudo_support": float(args.pseudo_support),
                        "seed": int(args.seed),
                    }
                )

        out_csv = os.path.join(out_csv_dir, f"seq_construction_time_vs_num_clusters_{dataset_key}_{ts}.csv")
        out_png = os.path.join(out_fig_dir, f"FIG_seq_construction_time_vs_num_clusters_{dataset_key}_{ts}.png")

        fieldnames = list(out_rows[0].keys()) if out_rows else []
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(out_rows)
        print(f"Wrote {out_csv}")

        by_algo: dict[str, list[float]] = {a: [] for a in algos}
        for n_sec in cluster_values:
            for algo in algos:
                rows = [
                    r
                    for r in out_rows
                    if r["algorithm"] == algo and int(r["n_clusters_secondary"]) == int(n_sec)
                ]
                by_algo[algo].append(float(rows[0]["exec_time_mean_s"]) if rows else float("nan"))

        _write_png_grouped_bars_pillow(
            out_png,
            title="Sequence construction execution time vs number of clusters",
            x_label="Number of clusters (secondary classes)",
            y_label="Execution time (seconds)",
            x_values=cluster_values,
            series=by_algo,
        )
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

