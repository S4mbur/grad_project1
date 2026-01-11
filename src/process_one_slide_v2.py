import os
import sys
import csv
import random
from dataclasses import dataclass
from typing import Tuple, Dict, List

import numpy as np
import openslide
from PIL import Image
import cv2


@dataclass
class Config:
    tile_out: int = 512
    scale_factor: int = 2
    target_kept: int = 1000
    max_attempts: int = 6000
    jpeg_quality: int = 90

    min_tissue_frac: float = 0.30
    blur_var_thresh: float = 80.0

    min_nuclei_score: float = 0.08

    seed: int = 42


def make_tissue_mask(slide: openslide.OpenSlide, level: int = None) -> Tuple[np.ndarray, int]:
    """Coarse tissue mask from a low-res level using simple HSV thresholds."""
    if level is None:
        level = slide.level_count - 1
    w, h = slide.level_dimensions[level]
    thumb = slide.read_region((0, 0), level, (w, h)).convert("RGB")
    img = np.array(thumb)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    tissue = (sat > 20) & (val < 245)

    tissue_u8 = (tissue.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_OPEN, kernel, iterations=1)
    return tissue_u8 > 0, level


def tissue_fraction(rgb: np.ndarray) -> float:
    """Estimate tissue fraction in a tile by simple HSV threshold."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    tissue = (sat > 20) & (val < 245)
    return float(tissue.mean())


def blur_variance(rgb: np.ndarray) -> float:
    """Variance of Laplacian as a blur metric."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def nuclei_proxy_score(rgb: np.ndarray) -> float:
    """
    Very simple nuclei/cellularity proxy:
    - Convert to grayscale
    - Count fraction of "dark" pixels (hematoxylin-ish regions tend to be darker)
    Returns in [0, 1]. Higher => more cellular / nuclei-dense.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return float((gray < 0.65).mean())


def main(slide_list_csv: str, manifest_out: str, target_kept: int = 1000):
    cfg = Config(target_kept=target_kept)
    rnd = random.Random(cfg.seed)

    all_rows: List[Dict[str, str]] = []

    with open(slide_list_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            slide_id = r["slide_id"]
            patient_id = r["patient_id"]
            split = r["split"]
            slide_path = r["local_path"]

            slide_label = r.get("label", "malignant").strip().lower()
            if slide_label not in ("benign", "malignant"):
                raise ValueError(
                    f"Invalid label={slide_label} for slide_id={slide_id}. Use benign/malignant."
                )

            if not os.path.exists(slide_path):
                raise FileNotFoundError(f"Missing slide: {slide_path}")

            out_dir = os.path.join("data/tiles", split)
            os.makedirs(out_dir, exist_ok=True)

            slide = openslide.OpenSlide(slide_path)

            tile_read = cfg.tile_out * cfg.scale_factor
            mask, mask_level = make_tissue_mask(slide)
            ys, xs = np.where(mask)
            if len(xs) == 0:
                slide.close()
                raise RuntimeError(f"[{slide_id}] Empty tissue mask")

            ds = float(slide.level_downsamples[mask_level])
            w0, h0 = slide.level_dimensions[0]

            kept = 0
            dropped_empty = 0
            dropped_blur = 0
            dropped_lowcell = 0
            attempts = 0

            while kept < cfg.target_kept and attempts < cfg.max_attempts:
                attempts += 1
                i = rnd.randrange(len(xs))
                x0 = int(xs[i] * ds)
                y0 = int(ys[i] * ds)

                if x0 + tile_read >= w0 or y0 + tile_read >= h0:
                    continue

                patch = slide.read_region((x0, y0), 0, (tile_read, tile_read)).convert("RGB")
                patch = patch.resize((cfg.tile_out, cfg.tile_out), resample=Image.BILINEAR)

                rgb = np.array(patch)

                tf = tissue_fraction(rgb)
                if tf < cfg.min_tissue_frac:
                    dropped_empty += 1
                    continue

                bv = blur_variance(rgb)
                if bv < cfg.blur_var_thresh:
                    dropped_blur += 1
                    continue

                ns = nuclei_proxy_score(rgb)
                if ns < cfg.min_nuclei_score:
                    dropped_lowcell += 1
                    continue

                fname = f"{slide_id}_tile_{kept:05d}_x{x0}_y{y0}.jpg"
                fpath = os.path.join(out_dir, fname)
                patch.save(fpath, quality=cfg.jpeg_quality)

                all_rows.append({
                    "tile_path": fpath,
                    "slide_id": slide_id,
                    "patient_id": patient_id,
                    "split": split,
                    "x0": str(x0),
                    "y0": str(y0),
                    "effective_mag": "20x",
                    "tile_size": str(cfg.tile_out),
                    "tissue_frac": f"{tf:.4f}",
                    "blur_var": f"{bv:.2f}",
                    "nuclei_score": f"{ns:.4f}",
                    "label": slide_label,
                })

                kept += 1

            slide.close()
            print(
                f"[{slide_id}] kept={kept}/{cfg.target_kept} attempts={attempts} "
                f"dropped_empty={dropped_empty} dropped_blur={dropped_blur} dropped_lowcell={dropped_lowcell}"
            )

    os.makedirs(os.path.dirname(manifest_out), exist_ok=True)
    fieldnames = list(all_rows[0].keys()) if all_rows else [
        "tile_path","slide_id","patient_id","split","x0","y0",
        "effective_mag","tile_size","tissue_frac","blur_var","nuclei_score","label"
    ]
    with open(manifest_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_rows)

    print("Manifest written to:", manifest_out)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python src/process_one_slide.py slide_list.csv tiles_manifest.csv [target_kept]")
        sys.exit(1)
    target = int(sys.argv[3]) if len(sys.argv) == 4 else 1000
    main(sys.argv[1], sys.argv[2], target_kept=target)
