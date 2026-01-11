import os
import sys
import random
from typing import Tuple, List

import numpy as np
import openslide
from PIL import Image
import cv2


def make_tissue_mask(slide: openslide.OpenSlide, level: int = None) -> Tuple[np.ndarray, int]:
    """
    Create a coarse tissue mask from a low-resolution level.
    Returns (mask, used_level).
    mask is a boolean array where True indicates tissue.
    """
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

    tissue = tissue_u8 > 0
    return tissue, level


def sample_coords_from_mask(
    mask: np.ndarray,
    slide: openslide.OpenSlide,
    mask_level: int,
    num_samples: int,
    tile_size_level0: int,
) -> List[Tuple[int, int]]:
    """
    Sample level-0 coordinates from a tissue mask computed at mask_level.
    """
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("Tissue mask is empty. Check thresholding or slide content.")

    downsample = float(slide.level_downsamples[mask_level])

    coords = []
    for _ in range(num_samples * 5):
        i = random.randrange(len(xs))
        x_mask, y_mask = int(xs[i]), int(ys[i])

        x0 = int(x_mask * downsample)
        y0 = int(y_mask * downsample)

        w0, h0 = slide.level_dimensions[0]
        if x0 + tile_size_level0 < w0 and y0 + tile_size_level0 < h0:
            coords.append((x0, y0))
        if len(coords) >= num_samples:
            break

    if len(coords) < num_samples:
        raise RuntimeError(f"Could only sample {len(coords)} coords. Try relaxing mask or fewer samples.")

    return coords


def main(path: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    slide = openslide.OpenSlide(path)

    tile_out = 512
    scale_factor = 2
    tile_read = tile_out * scale_factor

    mask, mask_level = make_tissue_mask(slide, level=None)
    coords = sample_coords_from_mask(mask, slide, mask_level, num_samples=10, tile_size_level0=tile_read)

    for idx, (x0, y0) in enumerate(coords):
        patch = slide.read_region((x0, y0), 0, (tile_read, tile_read)).convert("RGB")

        patch = patch.resize((tile_out, tile_out), resample=Image.BILINEAR)

        out_path = os.path.join(out_dir, f"tile_{idx:02d}_x{x0}_y{y0}.jpg")
        patch.save(out_path, quality=90)

    slide.close()
    print(f"Saved 10 demo tiles to: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/extract_demo_tiles.py /path/to/slide.svs output_dir")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
