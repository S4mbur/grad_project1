#!/usr/bin/env python3
"""
Script 03: Train patch classifier on extracted tiles.

This script:
1. Loads the tile manifest
2. Creates train/val datasets
3. Trains the patch classifier
4. Saves the best model checkpoint

Usage:
    python scripts/03_train.py [--epochs 20] [--batch-size 32]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.dataset.tile_dataset import TileDataset, load_tile_manifest
from src.models import create_model
from src.training import Trainer, TrainerConfig
from src.utils.transforms import get_train_transforms, get_eval_transforms


def main():
    parser = argparse.ArgumentParser(description="Train patch classifier")
    parser.add_argument("--tile-manifest", type=str, default="data/manifests/tile_manifest.csv",
                        help="Path to tile manifest")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--checkpoint-name", type=str, default="patch_classifier.pt",
                        help="Checkpoint filename")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    project_root = Path(__file__).parent.parent
    tile_manifest = project_root / args.tile_manifest
    
    if not tile_manifest.exists():
        logging.error(f"Tile manifest not found: {tile_manifest}")
        logging.error("Run 02_extract_tiles.py first")
        sys.exit(1)
    
    logging.info("Loading datasets...")
    
    train_tiles = load_tile_manifest(str(tile_manifest), split="train")
    val_tiles = load_tile_manifest(str(tile_manifest), split="val")
    
    logging.info(f"Train tiles: {len(train_tiles)}")
    logging.info(f"Val tiles: {len(val_tiles)}")
    
    train_transform = get_train_transforms(img_size=224, augmentation=True)
    eval_transform = get_eval_transforms(img_size=224)
    
    train_dataset = TileDataset(train_tiles, transform=train_transform)
    val_dataset = TileDataset(val_tiles, transform=eval_transform)
    
    train_stats = train_dataset.stats()
    val_stats = val_dataset.stats()
    logging.info(f"Train balance: benign={train_stats['benign']}, malignant={train_stats['malignant']}")
    logging.info(f"Val balance: benign={val_stats['benign']}, malignant={val_stats['malignant']}")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    
    logging.info("Creating model...")
    model = create_model(
        num_classes=2,
        pretrained=True,
        dropout=0.5,
        architecture="resnet18",
        device=device,
    )
    
    config = TrainerConfig(
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.0001,
        early_stopping_patience=5,
        device=device,
        checkpoint_dir=str(project_root / "logs" / "checkpoints"),
    )
    
    logging.info("=" * 60)
    logging.info("Starting training")
    logging.info("=" * 60)
    
    trainer = Trainer(model, train_loader, val_loader, config)
    results = trainer.train(checkpoint_name=args.checkpoint_name)
    
    logging.info("=" * 60)
    logging.info("Training complete!")
    logging.info(f"Best validation accuracy: {results['best_val_acc']:.4f}")
    logging.info(f"Checkpoint saved to: {results['checkpoint_path']}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
