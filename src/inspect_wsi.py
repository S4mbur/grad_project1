import sys
import openslide

def main(path: str):
    slide = openslide.OpenSlide(path)

    print("=== BASIC INFO ===")
    print("Path:", path)
    print("Vendor:", slide.properties.get("openslide.vendor"))
    print("Level count:", slide.level_count)

    print("\n=== LEVELS ===")
    for i in range(slide.level_count):
        w, h = slide.level_dimensions[i]
        ds = slide.level_downsamples[i]
        print(f"Level {i}: {w}x{h}  downsample={ds}")

    print("\n=== MPP / MAG (if available) ===")
    keys_to_try = [
        "openslide.mpp-x",
        "openslide.mpp-y",
        "aperio.MPP",
        "hamamatsu.XResolution",
        "hamamatsu.YResolution",
        "aperio.AppMag",
    ]
    for k in keys_to_try:
        if k in slide.properties:
            print(f"{k}: {slide.properties[k]}")

    mpp_x = slide.properties.get("openslide.mpp-x")
    mpp_y = slide.properties.get("openslide.mpp-y")
    if mpp_x and mpp_y:
        print(f"\nComputed note: MPP ≈ ({mpp_x}, {mpp_y}) µm/px (level 0)")

    slide.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/inspect_wsi.py /path/to/slide.svs")
        sys.exit(1)
    main(sys.argv[1])

