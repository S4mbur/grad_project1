#!/usr/bin/env python3
"""
Script 02: Extract tiles from downloaded WSI slides.

This script:
1. Reads the slide manifest
2. Extracts tiles from each slide with quality control
3. Creates a tile manifest for training

Usage:
    python scripts/02_extract_tiles.py [--max-tiles 500]
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.dataset.tile_extractor import TileExtractor, TileExtractionConfig
from src.dataset.tile_dataset import print_tile_stats


def main():
    parser = argparse.ArgumentParser(description="Extract tiles from WSI slides")
    parser.add_argument("--slide-manifest", type=str, default="data/manifests/slide_manifest.csv",
                        help="Path to slide manifest")
    parser.add_argument("--output-dir", type=str, default="data/tiles",
                        help="Output directory for tiles")
    parser.add_argument("--tile-manifest", type=str, default="data/manifests/tile_manifest.csv",
                        help="Output tile manifest path")
    parser.add_argument("--max-tiles", type=int, default=500,
                        help="Maximum tiles per slide")
    parser.add_argument("--tile-size", type=int, default=512,
                        help="Tile size in pixels")
    parser.add_argument("--min-tissue", type=float, default=0.3,
                        help="Minimum tissue fraction")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    project_root = Path(__file__).parent.parent
    slide_manifest = project_root / args.slide_manifest
    output_dir = project_root / args.output_dir
    tile_manifest = project_root / args.tile_manifest
    
    if not slide_manifest.exists():
        logging.error(f"Slide manifest not found: {slide_manifest}")
        logging.error("Run 01_prepare_slides.py first")
        sys.exit(1)
    
    config = TileExtractionConfig(
        tile_size=args.tile_size,
        max_tiles_per_slide=args.max_tiles,
        min_tissue_fraction=args.min_tissue,
    )
    
    logging.info("=" * 60)
    logging.info("Starting tile extraction")
    logging.info("=" * 60)
    logging.info(f"Slide manifest: {slide_manifest}")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Tile size: {config.tile_size}")
    logging.info(f"Max tiles per slide: {config.max_tiles_per_slide}")
    
    extractor = TileExtractor(config)
    
    tile_manifest_path = extractor.extract_from_manifest(
        slide_manifest_path=str(slide_manifest),
        output_dir=str(output_dir),
        tile_manifest_path=str(tile_manifest),
    )
    
    print_tile_stats(tile_manifest_path)
    
    logging.info(f"\nTile manifest saved to: {tile_manifest_path}")


if __name__ == "__main__":
    main()
