import os
import sys
import csv
import numpy as np
import cv2
import openslide
from PIL import Image


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def main(slide_path: str, tiles_with_scores_csv: str, slide_id: str, out_png: str, use_level: int = None):
    slide = openslide.OpenSlide(slide_path)

    if use_level is None:
        use_level = slide.level_count - 1
    wL, hL = slide.level_dimensions[use_level]
    ds = float(slide.level_downsamples[use_level])

    thumb = slide.read_region((0, 0), use_level, (wL, hL)).convert("RGB")
    thumb_np = np.array(thumb).astype(np.float32) / 255.0

    heat = np.zeros((hL, wL), dtype=np.float32)

    count = 0
    with open(tiles_with_scores_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("slide_id") != slide_id:
                continue
            x0 = int(r.get("x0") or r.get("x", 0))
            y0 = int(r.get("y0") or r.get("y", 0))
            p = float(r["p_malignant"])

            xL = int(x0 / ds)
            yL = int(y0 / ds)

            tile_out = int(r.get("tile_size", "512"))
            stamp = max(2, int(tile_out / ds * 2))

            x2 = min(wL, xL + stamp)
            y2 = min(hL, yL + stamp)

            if xL < 0 or yL < 0 or xL >= wL or yL >= hL:
                continue

            heat[yL:y2, xL:x2] = np.maximum(heat[yL:y2, xL:x2], p)
            count += 1

    if count == 0:
        raise RuntimeError(f"No tiles found for slide_id={slide_id}. Check slide_id in CSV.")

    heat_blur = cv2.GaussianBlur(heat, (0, 0), sigmaX=3, sigmaY=3)

    hmin, hmax = float(heat_blur.min()), float(heat_blur.max())
    if hmax > hmin:
        heat_norm = (heat_blur - hmin) / (hmax - hmin)
    else:
        heat_norm = heat_blur

    heat_u8 = (heat_norm * 255).astype(np.uint8)
    heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    alpha = 0.45
    overlay = clamp01((1 - alpha) * thumb_np + alpha * heat_color)

    bar_h = max(18, int(hL * 0.04))
    legend = np.zeros((bar_h, wL, 3), dtype=np.float32)

    grad = np.linspace(0, 255, wL, dtype=np.uint8)[None, :]
    grad_color = cv2.applyColorMap(grad, cv2.COLORMAP_JET)
    grad_color = cv2.cvtColor(grad_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    legend[:, :, :] = grad_color

    combined = np.vstack([overlay, legend])
    combined_u8 = (combined * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    Image.fromarray(combined_u8).save(out_png)
    slide.close()

    print(f"Slide heatmap saved: {out_png}")
    print(f"Used level={use_level}, size={wL}x{hL}, downsample={ds}, tiles_used={count}")


if __name__ == "__main__":
    if len(sys.argv) not in (5, 6):
        print("Usage: python src/build_slide_heatmap.py slide.svs tiles_with_scores.csv SLIDE_ID out.png [level]")
        sys.exit(1)
    level = int(sys.argv[5]) if len(sys.argv) == 6 else None
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], use_level=level)
