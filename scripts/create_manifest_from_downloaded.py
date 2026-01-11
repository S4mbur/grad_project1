#!/usr/bin/env python3
"""
Script: Create manifest from already downloaded slides.
Uses only the slides that are already in data/raw_wsi/
"""

import os
import sys
import csv
import random
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd


def main():
    project_root = Path(__file__).parent.parent
    wsi_dir = project_root / "data" / "raw_wsi"
    bcc_csv = project_root / "data" / "cobra" / "bcc_bcc.csv"
    output_path = project_root / "data" / "manifests" / "slide_manifest.csv"
    
    downloaded = set()
    for f in wsi_dir.glob("*.tif"):
        slide_id = f.stem
        downloaded.add(slide_id)
    
    print(f"Found {len(downloaded)} downloaded slides")
    
    bcc_df = pd.read_csv(bcc_csv)
    bcc_df["slide_id"] = bcc_df["filename"].str.strip()
    
    available = bcc_df[bcc_df["slide_id"].isin(downloaded)].copy()
    print(f"Matched {len(available)} slides with labels")
    
    benign = available[available["label"] == 0]["slide_id"].tolist()
    malignant = available[available["label"] == 1]["slide_id"].tolist()
    
    print(f"  Benign: {len(benign)}")
    print(f"  Malignant: {len(malignant)}")
    
    random.seed(42)
    random.shuffle(benign)
    random.shuffle(malignant)
    
    min_per_class = min(len(benign), len(malignant))
    print(f"  Min per class: {min_per_class}")
    
    train_n = int(min_per_class * 0.5)
    val_n = int(min_per_class * 0.25)
    test_n = min_per_class - train_n - val_n
    
    print(f"  Split: train={train_n}, val={val_n}, test={test_n} per class")
    
    slides = []
    
    for i, slide_id in enumerate(benign[:min_per_class]):
        if i < train_n:
            split = "train"
        elif i < train_n + val_n:
            split = "val"
        else:
            split = "test"
        slides.append({
            "slide_id": slide_id,
            "patient_id": slide_id,
            "local_path": f"data/raw_wsi/{slide_id}.tif",
            "label": "benign",
            "split": split,
        })
    
    for i, slide_id in enumerate(malignant[:min_per_class]):
        if i < train_n:
            split = "train"
        elif i < train_n + val_n:
            split = "val"
        else:
            split = "test"
        slides.append({
            "slide_id": slide_id,
            "patient_id": slide_id,
            "local_path": f"data/raw_wsi/{slide_id}.tif",
            "label": "malignant",
            "split": split,
        })
    
    os.makedirs(output_path.parent, exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "patient_id", "local_path", "label", "split"])
        writer.writeheader()
        for s in slides:
            writer.writerow(s)
    
    print(f"\nManifest saved to: {output_path}")
    print(f"Total slides: {len(slides)}")
    
    from collections import Counter
    splits = Counter(s["split"] for s in slides)
    labels = Counter(s["label"] for s in slides)
    
    print(f"\nBy split:")
    for split in ["train", "val", "test"]:
        split_slides = [s for s in slides if s["split"] == split]
        b = sum(1 for s in split_slides if s["label"] == "benign")
        m = len(split_slides) - b
        print(f"  {split}: {len(split_slides)} (benign={b}, malignant={m})")


if __name__ == "__main__":
    main()
