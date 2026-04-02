#!/usr/bin/env python3
"""
Retitle the final_results scalability dataset figures (PNG) by appending:
  " – <DATASET> dataset"

This is a visualization-only change; CSVs are not touched.
"""

from __future__ import annotations

import os


FOLDER_TO_BASE_TITLE = {
    "figure1_seq_construction_vs_phi": "Sequence Construction Execution Time vs Sequence Length φ",
    "figure2_seq_construction_vs_num_clusters": "Sequence construction execution time vs number of clusters",
    "figure6_rep_selection_vs_large_k": "Representative selection: execution time vs number of candidates",
    "figure7_instance_selection_vs_candidates_per_class": "Prompt instance selection: execution time vs candidates per class",
}

DATASET_DISPLAY = {
    "diffusion_db": "DiffusionDB",
    "liar": "LIAR",
    "race": "RACE",
}


def _infer_dataset_key(filename: str) -> str | None:
    for k in ("diffusion_db", "liar", "race"):
        if f"_{k}_" in filename:
            return k
    return None


def _load_font(ImageFont, size: int):
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


def _retitle_one_png(png_path: str, *, new_title: str) -> None:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    img = Image.open(png_path).convert("RGB")
    d = ImageDraw.Draw(img)

    w, _h = img.size
    font_title = _load_font(ImageFont, 28)

    # Clear only the title band (keep plot-area annotations like "exceeds axis").
    band_h = 55
    d.rectangle((0, 0, w, band_h), fill=(255, 255, 255))

    # Re-draw title centered where the plotters place it.
    d.text((w / 2, 35), new_title, fill=(17, 17, 17), font=font_title, anchor="mm")
    img.save(png_path, format="PNG")


def main() -> None:
    root = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "experiments",
        "final_results",
        "scalability_datasets",
    )
    if not os.path.isdir(root):
        raise SystemExit(f"Missing folder: {root}")

    changed = 0
    for folder_name, base_title in FOLDER_TO_BASE_TITLE.items():
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for name in os.listdir(folder_path):
            if not name.lower().endswith(".png"):
                continue
            dataset_key = _infer_dataset_key(name)
            if not dataset_key:
                continue
            ds = DATASET_DISPLAY.get(dataset_key, dataset_key)
            new_title = f"{base_title} – {ds} dataset"
            _retitle_one_png(os.path.join(folder_path, name), new_title=new_title)
            changed += 1

    print(f"Retitled {changed} PNG(s).")


if __name__ == "__main__":
    main()

