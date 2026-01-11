"""
Tile dataset for training patch classifiers.
"""

import os
import csv
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass
class TileInfo:
    """Information about a single tile."""
    tile_path: str
    slide_id: str
    label: str
    split: str
    label_int: int = 0
    
    def __post_init__(self):
        self.label_int = 1 if self.label == "malignant" else 0


class TileDataset(Dataset):
    """PyTorch Dataset for tile images."""
    
    def __init__(
        self,
        tiles: List[TileInfo],
        transform: Optional[Callable] = None,
    ):
        self.tiles = tiles
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int):
        tile = self.tiles[idx]
        
        image = Image.open(tile.tile_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        label = torch.tensor(tile.label_int, dtype=torch.long)
        
        return image, label
    
    def get_labels(self) -> List[int]:
        """Get all labels for computing class weights."""
        return [t.label_int for t in self.tiles]
    
    def stats(self) -> Dict:
        """Get dataset statistics."""
        benign = sum(1 for t in self.tiles if t.label == "benign")
        malignant = len(self.tiles) - benign
        return {
            "total": len(self.tiles),
            "benign": benign,
            "malignant": malignant,
            "balance_ratio": benign / malignant if malignant > 0 else float('inf'),
        }


def load_tile_manifest(manifest_path: str, split: Optional[str] = None) -> List[TileInfo]:
    """
    Load tile manifest from CSV file.
    
    Args:
        manifest_path: Path to the tile manifest CSV
        split: Optional split filter ("train", "val", "test")
    
    Returns:
        List of TileInfo objects
    """
    tiles = []
    with open(manifest_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_split = row["split"].strip().lower()
            if split and row_split != split.lower():
                continue
            
            tile = TileInfo(
                tile_path=row["tile_path"],
                slide_id=row["slide_id"],
                label=row["label"].strip().lower(),
                split=row_split,
            )
            tiles.append(tile)
    
    return tiles


def create_tile_datasets(
    manifest_path: str,
    train_transform: Optional[Callable] = None,
    eval_transform: Optional[Callable] = None,
) -> Dict[str, TileDataset]:
    """
    Create train, val, and test TileDatasets from manifest.
    
    Returns:
        Dictionary with "train", "val", "test" keys
    """
    datasets = {}
    
    for split in ["train", "val", "test"]:
        tiles = load_tile_manifest(manifest_path, split=split)
        transform = train_transform if split == "train" else eval_transform
        datasets[split] = TileDataset(tiles, transform=transform)
    
    return datasets


def print_tile_stats(manifest_path: str):
    """Print statistics about the tile manifest."""
    print(f"\n{'='*50}")
    print(f"Tile Manifest Statistics: {manifest_path}")
    print(f"{'='*50}")
    
    for split in ["train", "val", "test"]:
        tiles = load_tile_manifest(manifest_path, split=split)
        if tiles:
            dataset = TileDataset(tiles)
            stats = dataset.stats()
            print(f"{split}: {stats['total']} tiles (benign={stats['benign']}, malignant={stats['malignant']})")
    
    all_tiles = load_tile_manifest(manifest_path)
    total_stats = TileDataset(all_tiles).stats()
    print(f"\nTotal: {total_stats['total']} tiles")
    print(f"{'='*50}\n")
