import sys
import os
import pandas as pd

def strip_ext(x: str) -> str:
    x = str(x)
    base = os.path.basename(x)
    if base.lower().endswith((".tif", ".tiff", ".svs")):
        base = os.path.splitext(base)[0]
    return base

def main(out_csv: str, images_csv: str, labels_csv: str):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    imgs = pd.read_csv(images_csv)
    lab = pd.read_csv(labels_csv)

    if "filename" not in imgs.columns:
        raise ValueError(f"{images_csv} must have a 'filename' column. cols={list(imgs.columns)}")
    if "filename" not in lab.columns or "label" not in lab.columns:
        raise ValueError(f"{labels_csv} must have 'filename' and 'label'. cols={list(lab.columns)}")

    imgs["image_id"] = imgs["filename"].apply(strip_ext)
    lab["image_id"]  = lab["filename"].apply(strip_ext)

    m = imgs.merge(lab[["image_id", "label", "split"]], on="image_id", how="inner")

    def map_label(v):
        return "malignant" if int(v) == 1 else "benign"

    out = pd.DataFrame({
        "slide_id": m["image_id"],
        "patient_id": m["image_id"],
        "split": m["split"],
        "local_path": "data/raw_wsi/" + m["filename"].astype(str),
        "label": m["label"].apply(map_label),
        "source": "cobra"
    })

    out.to_csv(out_csv, index=False)
    print("Wrote:", out_csv, "rows=", len(out))
    print(out["label"].value_counts())
    print(out["split"].value_counts())

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python src/build_cobra_slide_list.py out.csv bcc_images.csv bcc_bcc.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
