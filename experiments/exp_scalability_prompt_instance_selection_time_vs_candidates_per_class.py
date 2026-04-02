#!/usr/bin/env python3
"""
Figure 7 (Scalability) — Prompt instance selection execution time vs candidates per class.

Chart type: Line chart
X-axis: Number of candidate prompts per class
Y-axis: Execution time (seconds)
Lines (algorithms):
  - PromptSelector
  - SampledGreedySelector
  - NaiveSelector (computed in CSV, not plotted in the figure by default)

Setup (synthetic, no real LLM calls):
- Fix φ = 1 (choose exactly one prompt instance from a class)
- Pre-generate candidate "prompts" and embeddings per x-value outside the timed section
- During timing, add synthetic latencies (added to runtime without sleeping):
  - LLM latency per "chat" call: sampled uniformly in [0.5, 1.0] seconds
  - Embedding latency per embedding API call: sampled uniformly in [0.05, 0.15] seconds

Run from repo root:
  python experiments/exp_scalability_prompt_instance_selection_time_vs_candidates_per_class.py

Outputs:
  experiments/outputs/csv/scalability_prompt_instance_selection_time_vs_candidates_per_class_<timestamp>.csv
  experiments/outputs/figs/FIG_prompt_instance_selection_time_vs_candidates_per_class_<timestamp>.png

Requires (for PNG):
  pip install pillow
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import os
import random
import statistics
import time
from datetime import datetime
from typing import Literal


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
        raise SystemExit("--candidates_values must include at least one integer (e.g. 10,100,500)")
    out = sorted(set(out))
    if any(v < 1 for v in out):
        raise SystemExit("--candidates_values must be >= 1")
    return out


def _dot(a: list[float], b: list[float]) -> float:
    s = 0.0
    # plain python loop (keeps us dependency-free)
    for i in range(min(len(a), len(b))):
        s += a[i] * b[i]
    return s


def _synthetic_llm_latency_sum(rng: random.Random, n_calls: int, lo: float, hi: float) -> float:
    if n_calls <= 0:
        return 0.0
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return sum(lo + (hi - lo) * rng.random() for _ in range(int(n_calls)))


def _synthetic_embedding_latency_sum(
    rng: random.Random, n_calls: int, lo: float, hi: float
) -> float:
    if n_calls <= 0:
        return 0.0
    lo = float(lo)
    hi = float(hi)
    if hi < lo:
        lo, hi = hi, lo
    return sum(lo + (hi - lo) * rng.random() for _ in range(int(n_calls)))


def _promptselector_pick_one(
    *,
    query_emb: list[float],
    cand_embs: list[list[float]],
    top_l: int,
    rng: random.Random,
) -> int:
    """
    PromptSelector proxy:
    - score all candidates by similarity to a query embedding
    - evaluate only top_l candidates with "LLM"
    - return chosen candidate index (argmax among top_l)
    """
    scores = [(i, _dot(query_emb, e)) for i, e in enumerate(cand_embs)]
    top_l = max(1, min(int(top_l), len(scores)))
    top = heapq.nlargest(top_l, scores, key=lambda x: x[1])
    # "LLM" would pick the best among the shortlisted
    best_i = max(top, key=lambda x: x[1])[0]
    _ = rng.random()  # tiny non-zero overhead
    return int(best_i)


def _sampled_greedy_pick_one(
    *,
    cand_count: int,
    sample_fraction: float,
    rng: random.Random,
) -> tuple[int, int]:
    """
    SampledGreedySelector proxy for φ=1:
    - sample a fraction of candidates, evaluate each with "LLM", pick best
    Returns (chosen_index, llm_calls)
    """
    n = int(cand_count)
    m = max(1, min(n, int(math.ceil(float(sample_fraction) * n))))
    pool = rng.sample(range(n), m) if m < n else list(range(n))
    # choose deterministic best = smallest index (placeholder)
    # NOTE: real SampledGreedySelector does NOT do chat-LLM reranking; it embeds the current prompt
    # and uses cosine similarity against stored candidate embeddings. So in this scalability experiment
    # we account for 1 embedding API call (φ=1) and 0 LLM calls.
    return int(min(pool)), 0


def _naive_pick_one(*, cand_count: int) -> tuple[int, int]:
    """
    NaiveSelector proxy for φ=1:
    - evaluate all candidates with "LLM", pick best
    Returns (chosen_index, llm_calls)
    """
    n = int(cand_count)
    return 0, int(n)


def _write_png_line_chart_pillow(
    out_path: str,
    *,
    title: str,
    x_label: str,
    y_label: str,
    x_values: list[int],
    series: dict[str, list[float]],
    styles: dict[str, dict],
    x_scale: Literal["linear", "log10"] = "log10",
    y_cap: float | None = None,
    width: int = 1400,
    height: int = 800,
) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except Exception as e:
        raise SystemExit("PNG rendering requires Pillow: pip install pillow") from e

    pad_l, pad_r, pad_t, pad_b = 130, 60, 90, 120
    names = [k for k in styles.keys() if k in series]
    if not x_values or not names:
        raise SystemExit("No data to plot.")

    xs = [float(x) for x in x_values]
    if x_scale == "log10":
        tx = [math.log10(max(1.0, x)) for x in xs]
    else:
        tx = xs[:]
    x_min, x_max = min(tx), max(tx)
    if x_min == x_max:
        x_max = x_min + 1.0

    ys = [v for name in names for v in series[name] if v is not None and not math.isnan(v)]
    y_min = 0.0
    if y_cap is not None:
        y_max = float(y_cap)
    else:
        y_max = max(ys) if ys else 1.0
        if y_max <= 0:
            y_max = 1.0
        y_max *= 1.10

    def x_px(x: int) -> float:
        xv = float(x)
        if x_scale == "log10":
            xv = math.log10(max(1.0, xv))
        return pad_l + (xv - x_min) * (width - pad_l - pad_r) / (x_max - x_min)

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
    # Move the y-axis label further left (more gap vs tick labels).
    img.paste(yl_img, (4, int(height / 2 - yl_img.size[1] / 2)), yl_img)

    # Grid/ticks
    grid = (229, 231, 235)
    # If y_max is a small integer, use integer ticks for readability.
    cap_int = int(round(y_max))
    if abs(y_max - cap_int) < 1e-9 and 1 <= cap_int <= 10:
        y_ticks = [float(i) for i in range(0, cap_int + 1)]
    else:
        y_ticks = [y_min + frac * (y_max - y_min) for frac in [0.0, 0.25, 0.5, 0.75, 1.0]]
    for yv in y_ticks:
        yp = y_px(float(yv))
        d.line((x0, yp, x1, yp), fill=grid, width=1)
        lab = f"{int(yv)}" if float(yv).is_integer() else f"{yv:.3f}"
        d.text((x0 - 12, yp), lab, fill=(17, 17, 17), font=font_tick, anchor="rm")

    # X ticks
    for x in x_values:
        xp = x_px(int(x))
        d.line((xp, y0, xp, y0 + 8), fill=(17, 17, 17), width=2)
        # Use commas for readability on large values (e.g., 10,000).
        d.text((xp, y0 + 32), f"{int(x):,}", fill=(17, 17, 17), font=font_tick, anchor="mm")

    # Legend
    lx, ly = x0 + 10, y1 + 10
    for i, name in enumerate(names):
        st = styles[name]
        y_ = ly + i * 28
        # sample line
        d.line((lx, y_, lx + 26, y_), fill=st["color"], width=4)
        # marker
        r = 6
        if st["marker"] == "circle":
            d.ellipse((lx + 13 - r, y_ - r, lx + 13 + r, y_ + r), fill=st["color"], outline=st["color"])
        elif st["marker"] == "square":
            d.rectangle((lx + 13 - r, y_ - r, lx + 13 + r, y_ + r), fill=st["color"], outline=st["color"])
        else:  # triangle
            d.polygon([(lx + 13, y_ - r), (lx + 13 - r, y_ + r), (lx + 13 + r, y_ + r)], fill=st["color"], outline=st["color"])
        d.text((lx + 36, y_), name, fill=(17, 17, 17), font=font_legend, anchor="lm")

    # Lines
    for name in names:
        st = styles[name]
        # If this series exceeds the capped y-axis, stop after the first overflow point.
        pts = []
        for i in range(len(x_values)):
            xv = int(x_values[i])
            yv = float(series[name][i])
            if math.isnan(yv):
                continue
            pts.append((x_px(xv), y_px(min(yv, float(y_cap)) if y_cap is not None else yv)))
        # styled line: solid/dashed/dotted
        if st["line"] == "solid":
            d.line(pts, fill=st["color"], width=4)
        else:
            dash = 18.0 if st["line"] == "dashed" else 4.0
            gap = 10.0 if st["line"] == "dashed" else 8.0
            step = dash + gap
            for i in range(len(pts) - 1):
                x0s, y0s = pts[i]
                x1s, y1s = pts[i + 1]
                dx = x1s - x0s
                dy = y1s - y0s
                seg_len = math.hypot(dx, dy)
                if seg_len <= 1e-6:
                    continue
                u = 0.0
                while u < seg_len:
                    u2 = min(seg_len, u + dash)
                    t0 = u / seg_len
                    t1 = u2 / seg_len
                    p0 = (x0s + dx * t0, y0s + dy * t0)
                    p1 = (x0s + dx * t1, y0s + dy * t1)
                    d.line([p0, p1], fill=st["color"], width=4)
                    u += step
        # markers
        for (xp, yp) in pts:
            r = 6
            if st["marker"] == "circle":
                d.ellipse((xp - r, yp - r, xp + r, yp + r), fill=st["color"], outline=st["color"])
            elif st["marker"] == "square":
                d.rectangle((xp - r, yp - r, xp + r, yp + r), fill=st["color"], outline=st["color"])
            else:
                d.polygon([(xp, yp - r), (xp - r, yp + r), (xp + r, yp + r)], fill=st["color"], outline=st["color"])

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path, format="PNG")


def main() -> None:
    p = argparse.ArgumentParser(description="Figure 7: prompt instance selection scalability (synthetic).")
    p.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "diffusion_db", "liar", "race", "synthetic"],
    )
    p.add_argument("--max_rows_for_stats", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phi", type=int, default=1, help="Fixed φ (default: 1).")
    p.add_argument("--candidates_values", type=str, default="1000,5000,10000,20000,50000")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--top_l", type=int, default=5, help="PromptSelector shortlist size.")
    p.add_argument("--sample_fraction", type=float, default=0.15, help="SampledGreedySelector fraction.")
    p.add_argument("--llm_latency_min_s", type=float, default=0.5)
    p.add_argument("--llm_latency_max_s", type=float, default=1.0)
    p.add_argument("--embedding_latency_min_s", type=float, default=0.05)
    p.add_argument("--embedding_latency_max_s", type=float, default=0.15)
    p.add_argument("--x_scale", type=str, default="log10", choices=["linear", "log10"])
    args = p.parse_args()

    phi = int(args.phi)
    if phi != 1:
        raise SystemExit("This experiment assumes φ=1. Use --phi 1.")

    xs = _parse_int_list(args.candidates_values)
    reps = max(1, int(args.reps))

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

        out_csv = os.path.join(
            out_csv_dir,
            f"scalability_prompt_instance_selection_time_vs_candidates_per_class_{dataset_key}_{ts}.csv",
        )
        out_png = os.path.join(
            out_fig_dir,
            f"FIG_prompt_instance_selection_time_vs_candidates_per_class_{dataset_key}_{ts}.png",
        )

        rng_master = random.Random(int(args.seed) + (abs(hash(dataset_key)) % 10_000))

        # Pre-generate candidates + embeddings per x-value (excluded from timing)
        embed_dim = max(8, int(args.embed_dim))
        per_x = {}
        for c in xs:
            rng_gen = random.Random(rng_master.randint(0, 10**9))
            cand_embs = [[rng_gen.uniform(-1.0, 1.0) for _ in range(embed_dim)] for _ in range(int(c))]
            query_emb = [rng_gen.uniform(-1.0, 1.0) for _ in range(embed_dim)]
            per_x[int(c)] = (query_emb, cand_embs)

        rows = []
        algos = ("PromptSelector", "SampledGreedySelector", "NaiveSelector")

        for c in xs:
            query_emb, cand_embs = per_x[int(c)]
            for algo in algos:
                times = []
                llm_calls = []
                embedding_calls = []
                for r_i in range(reps):
                    rng = random.Random(rng_master.randint(0, 10**9) + r_i * 10007 + int(c) * 97)
                    t0 = time.perf_counter()

                    if algo == "PromptSelector":
                        _ = _promptselector_pick_one(
                            query_emb=query_emb,
                            cand_embs=cand_embs,
                            top_l=int(args.top_l),
                            rng=rng,
                        )
                        calls = int(max(1, min(int(args.top_l), int(c))))
                        emb_calls = 0
                    elif algo == "SampledGreedySelector":
                        _idx, calls = _sampled_greedy_pick_one(
                            cand_count=int(c),
                            sample_fraction=float(args.sample_fraction),
                            rng=rng,
                        )
                        emb_calls = 1  # embed current prompt once for φ=1
                    else:
                        _idx, calls = _naive_pick_one(cand_count=int(c))
                        emb_calls = 0

                    compute_s = time.perf_counter() - t0
                    llm_latency_s = _synthetic_llm_latency_sum(
                        rng,
                        n_calls=int(calls),
                        lo=float(args.llm_latency_min_s),
                        hi=float(args.llm_latency_max_s),
                    )
                    emb_latency_s = _synthetic_embedding_latency_sum(
                        rng,
                        n_calls=int(emb_calls),
                        lo=float(args.embedding_latency_min_s),
                        hi=float(args.embedding_latency_max_s),
                    )
                    times.append(float((compute_s + llm_latency_s + emb_latency_s) * scale))
                    llm_calls.append(int(calls))
                    embedding_calls.append(int(emb_calls))

                rows.append(
                    {
                        "dataset": str(dataset_key),
                        "dataset_display": str(DATASET_DISPLAY.get(dataset_key, dataset_key)),
                        "dataset_avg_primary_words_est": float(avg_words),
                        "dataset_scale_applied": float(scale),
                        "algorithm": algo,
                        "candidates_per_class": int(c),
                        "phi_fixed": 1,
                        "reps": reps,
                        "exec_time_mean_s": float(statistics.mean(times)),
                        "exec_time_std_s": float(statistics.pstdev(times)) if len(times) > 1 else 0.0,
                        "llm_calls_mean": float(statistics.mean(llm_calls)) if llm_calls else 0.0,
                        "embedding_calls_mean": float(statistics.mean(embedding_calls)) if embedding_calls else 0.0,
                        "llm_latency_min_s": float(args.llm_latency_min_s),
                        "llm_latency_max_s": float(args.llm_latency_max_s),
                        "embedding_latency_min_s": float(args.embedding_latency_min_s),
                        "embedding_latency_max_s": float(args.embedding_latency_max_s),
                        "seed": int(args.seed),
                    }
                )

        fieldnames = sorted({k for r in rows for k in r.keys()})
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out_csv}")

        series = {a: [] for a in algos}
        for c in xs:
            for a in algos:
                match = [r for r in rows if r["algorithm"] == a and int(r["candidates_per_class"]) == int(c)]
                series[a].append(float(match[0]["exec_time_mean_s"]) if match else float("nan"))

        styles = {
            "PromptSelector": {"color": (37, 99, 235), "line": "solid", "marker": "square"},
            "SampledGreedySelector": {"color": (22, 163, 74), "line": "dashed", "marker": "circle"},
        }
        _write_png_line_chart_pillow(
            out_png,
            title="Prompt instance selection: execution time vs candidates per class",
            x_label="Number of candidate prompts per class",
            y_label="Execution time (seconds)",
            x_values=xs,
            series=series,
            styles=styles,
            x_scale=str(args.x_scale),  # type: ignore[arg-type]
            y_cap=5.0,
        )
        print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()

