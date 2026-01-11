"""
Slide dataset management.
Handles slide list loading, filtering, and patient-aware splitting.
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import pandas as pd


@dataclass
class SlideInfo:
    """Information about a single slide."""
    slide_id: str
    patient_id: str
    local_path: str
    label: str
    split: str
    label_int: int = 0
    
    def __post_init__(self):
        self.label_int = 1 if self.label == "malignant" else 0


class SlideDataset:
    """Dataset class for managing slide information."""
    
    def __init__(self, slides: List[SlideInfo]):
        self.slides = slides
        self._build_index()
    
    def _build_index(self):
        """Build indices for quick lookups."""
        self.by_id = {s.slide_id: s for s in self.slides}
        self.by_split = {"train": [], "val": [], "test": []}
        for s in self.slides:
            if s.split in self.by_split:
                self.by_split[s.split].append(s)
    
    def __len__(self) -> int:
        return len(self.slides)
    
    def __getitem__(self, idx: int) -> SlideInfo:
        return self.slides[idx]
    
    def get_split(self, split: str) -> List[SlideInfo]:
        """Get all slides for a specific split."""
        return self.by_split.get(split, [])
    
    def get_by_label(self, label: str) -> List[SlideInfo]:
        """Get all slides with a specific label."""
        return [s for s in self.slides if s.label == label]
    
    def stats(self) -> Dict:
        """Get dataset statistics."""
        stats = {
            "total": len(self.slides),
            "by_split": {},
            "by_label": {"benign": 0, "malignant": 0},
        }
        for split in ["train", "val", "test"]:
            split_slides = self.by_split.get(split, [])
            benign = sum(1 for s in split_slides if s.label == "benign")
            malignant = len(split_slides) - benign
            stats["by_split"][split] = {
                "total": len(split_slides),
                "benign": benign,
                "malignant": malignant,
            }
        stats["by_label"]["benign"] = sum(1 for s in self.slides if s.label == "benign")
        stats["by_label"]["malignant"] = len(self.slides) - stats["by_label"]["benign"]
        return stats
    
    def print_stats(self):
        """Print dataset statistics."""
        stats = self.stats()
        print(f"\n{'='*50}")
        print(f"Dataset Statistics")
        print(f"{'='*50}")
        print(f"Total slides: {stats['total']}")
        print(f"  Benign: {stats['by_label']['benign']}")
        print(f"  Malignant: {stats['by_label']['malignant']}")
        print(f"\nBy split:")
        for split, data in stats["by_split"].items():
            print(f"  {split}: {data['total']} (benign={data['benign']}, malignant={data['malignant']})")
        print(f"{'='*50}\n")


def load_slide_manifest(manifest_path: str) -> SlideDataset:
    """Load slide manifest from CSV file."""
    slides = []
    with open(manifest_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide = SlideInfo(
                slide_id=row["slide_id"],
                patient_id=row.get("patient_id", row["slide_id"]),
                local_path=row["local_path"],
                label=row["label"].strip().lower(),
                split=row["split"].strip().lower(),
            )
            slides.append(slide)
    return SlideDataset(slides)


def create_slide_manifest_from_cobra(
    bcc_csv: str,
    images_csv: str,
    output_path: str,
    wsi_dir: str = "data/raw_wsi",
    train_per_class: int = 400,
    val_per_class: int = 100,
    test_per_class: int = 100,
    seed: int = 42,
    append: bool = False,
) -> SlideDataset:
    """
    Create a balanced slide manifest from COBRA metadata.
    
    Args:
        bcc_csv: Path to bcc_bcc.csv (contains labels and original splits)
        images_csv: Path to bcc_images.csv (contains file metadata)
        output_path: Where to save the manifest
        wsi_dir: Directory where WSI files will be stored
        train_per_class: Number of slides per class for training
        val_per_class: Number of slides per class for validation
        test_per_class: Number of slides per class for testing
        seed: Random seed for reproducibility
        append: If True, add new slides to existing manifest
    """
    random.seed(seed)
    
    existing_ids = set()
    existing_slides = []
    if append and os.path.exists(output_path):
        existing_dataset = load_slide_manifest(output_path)
        existing_slides = existing_dataset.slides
        existing_ids = {s.slide_id for s in existing_slides}
        print(f"Appending to existing manifest with {len(existing_ids)} slides")
    
    bcc_df = pd.read_csv(bcc_csv)
    images_df = pd.read_csv(images_csv)
    
    bcc_df["file_id"] = bcc_df["filename"].str.strip()
    images_df["file_id"] = images_df["filename"].str.replace(".tif", "", regex=False).str.strip()
    
    merged = bcc_df.merge(images_df[["file_id", "filename"]], on="file_id", how="inner")
    
    benign_slides = [s for s in merged[merged["label"] == 0]["file_id"].tolist() if s not in existing_ids]
    malignant_slides = [s for s in merged[merged["label"] == 1]["file_id"].tolist() if s not in existing_ids]
    
    random.shuffle(benign_slides)
    random.shuffle(malignant_slides)
    
    def allocate(slides: List[str], train_n: int, val_n: int, test_n: int):
        total_needed = train_n + val_n + test_n
        if len(slides) < total_needed:
            raise ValueError(f"Not enough slides: need {total_needed}, have {len(slides)}")
        return {
            "train": slides[:train_n],
            "val": slides[train_n:train_n + val_n],
            "test": slides[train_n + val_n:train_n + val_n + test_n],
        }
    
    benign_splits = allocate(benign_slides, train_per_class, val_per_class, test_per_class)
    malignant_splits = allocate(malignant_slides, train_per_class, val_per_class, test_per_class)
    
    slides = list(existing_slides) if append else []
    
    for split in ["train", "val", "test"]:
        for slide_id in benign_splits[split]:
            slides.append(SlideInfo(
                slide_id=slide_id,
                patient_id=slide_id,
                local_path=f"{wsi_dir}/{slide_id}.tif",
                label="benign",
                split=split,
            ))
        for slide_id in malignant_splits[split]:
            slides.append(SlideInfo(
                slide_id=slide_id,
                patient_id=slide_id,
                local_path=f"{wsi_dir}/{slide_id}.tif",
                label="malignant",
                split=split,
            ))
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["slide_id", "patient_id", "local_path", "label", "split"])
        writer.writeheader()
        for s in slides:
            writer.writerow({
                "slide_id": s.slide_id,
                "patient_id": s.patient_id,
                "local_path": s.local_path,
                "label": s.label,
                "split": s.split,
            })
    
    print(f"Created manifest: {output_path}")
    dataset = SlideDataset(slides)
    dataset.print_stats()
    
    return dataset


def get_slides_to_download(manifest_path: str) -> List[Tuple[str, str]]:
    """
    Get list of (slide_id, filename) tuples for downloading.
    Returns slides that need to be downloaded from S3.
    """
    dataset = load_slide_manifest(manifest_path)
    return [(s.slide_id, f"{s.slide_id}.tif") for s in dataset.slides]
