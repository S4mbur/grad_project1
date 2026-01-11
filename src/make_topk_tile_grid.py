import os
import sys
import csv
from math import ceil, sqrt
from typing import List, Tuple

from PIL import Image, ImageDraw, ImageFont


def read_topk(tile_scores_csv: str, slide_id: str, k: int) -> List[Tuple[float, str]]:
    items = []
    with open(tile_scores_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("slide_id") != slide_id:
                continue
            p = float(row["p_malignant"])
            items.append((p, row["tile_path"]))
    items.sort(key=lambda x: x[0], reverse=True)
    return items[:k]


def safe_font(size: int = 16):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def main(tile_scores_csv: str, slide_id: str, out_png: str, k: int = 16, cols: int = 4, tile_px: int = 256):
    topk = read_topk(tile_scores_csv, slide_id, k)
    if len(topk) == 0:
        raise RuntimeError(f"No tiles for slide_id={slide_id}")

    rows = ceil(len(topk) / cols)
    w = cols * tile_px
    header_h = 40
    h = header_h + rows * tile_px

    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = safe_font(18)

    title = f"Top-{len(topk)} tiles by p_malignant  |  slide={slide_id}"
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)

    font_small = safe_font(14)

    for idx, (p, path) in enumerate(topk):
        r = idx // cols
        c = idx % cols
        x = c * tile_px
        y = header_h + r * tile_px

        img = Image.open(path).convert("RGB")
        img = img.resize((tile_px, tile_px), resample=Image.BILINEAR)
        canvas.paste(img, (x, y))

        label = f"{p:.3f}"
        pad = 6
        box_w, box_h = draw.textbbox((0, 0), label, font=font_small)[2:]
        draw.rectangle([x + pad, y + pad, x + pad + box_w + 8, y + pad + box_h + 6], fill=(255, 255, 255))
        draw.text((x + pad + 4, y + pad + 2), label, fill=(0, 0, 0), font=font_small)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)
    print("Saved:", out_png)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5, 6, 7):
        print("Usage: python src/make_topk_tile_grid.py tiles_with_scores.csv SLIDE_ID out.png [k] [cols] [tile_px]")
        sys.exit(1)
    k = int(sys.argv[4]) if len(sys.argv) >= 5 else 16
    cols = int(sys.argv[5]) if len(sys.argv) >= 6 else 4
    tile_px = int(sys.argv[6]) if len(sys.argv) >= 7 else 256
    main(sys.argv[1], sys.argv[2], sys.argv[3], k=k, cols=cols, tile_px=tile_px)
