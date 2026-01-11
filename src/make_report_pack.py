import os
import sys
from PIL import Image, ImageDraw, ImageFont


def safe_font(size: int = 18):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def main(heatmap_png: str, grid_png: str, out_png: str, title: str = "Phase-1 Explainability Pack"):
    hm = Image.open(heatmap_png).convert("RGB")
    grid = Image.open(grid_png).convert("RGB")

    target_h = max(hm.height, grid.height)
    hm2 = hm.resize((int(hm.width * (target_h / hm.height)), target_h), resample=Image.BILINEAR)
    grid2 = grid.resize((int(grid.width * (target_h / grid.height)), target_h), resample=Image.BILINEAR)

    header_h = 60
    w = hm2.width + grid2.width
    h = header_h + target_h

    canvas = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = safe_font(22)

    draw.text((10, 16), title, fill=(0, 0, 0), font=font)

    canvas.paste(hm2, (0, header_h))
    canvas.paste(grid2, (hm2.width, header_h))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    canvas.save(out_png)
    print("Saved:", out_png)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python src/make_report_pack.py heatmap.png topk_grid.png out.png [title]")
        sys.exit(1)
    title = sys.argv[4] if len(sys.argv) >= 5 else "Phase-1 Explainability Pack"
    main(sys.argv[1], sys.argv[2], sys.argv[3], title=title)
