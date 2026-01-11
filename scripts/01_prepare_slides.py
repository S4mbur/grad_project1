#!/usr/bin/env python3
"""
Script 01: Prepare slide list and download slides from COBRA dataset.

This script:
1. Creates a balanced slide manifest from COBRA metadata
2. Downloads the selected slides from AWS S3

Usage:
    python scripts/01_prepare_slides.py [--download]
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import load_config
from src.dataset.slide_dataset import create_slide_manifest_from_cobra, get_slides_to_download


def download_slides(manifest_path: str, wsi_dir: str, s3_bucket: str = "s3://cobra-pathology/packages/bcc/images/"):
    """Download slides from S3 using AWS CLI."""
    
    slides = get_slides_to_download(manifest_path)
    total = len(slides)
    
    os.makedirs(wsi_dir, exist_ok=True)
    
    print(f"\nDownloading {total} slides to {wsi_dir}...")
    print(f"Source: {s3_bucket}")
    print("-" * 60)
    
    success = 0
    failed = []
    
    for i, (slide_id, filename) in enumerate(slides):
        local_path = os.path.join(wsi_dir, filename)
        
        if os.path.exists(local_path):
            print(f"[{i+1}/{total}] {filename} - already exists, skipping")
            success += 1
            continue
        
        s3_path = f"{s3_bucket}{filename}"
        
        print(f"[{i+1}/{total}] Downloading {filename}...")
        
        try:
            cmd = [
                "aws", "s3", "cp",
                s3_path,
                local_path,
                "--no-sign-request"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                success += 1
                size_mb = os.path.getsize(local_path) / (1024 * 1024)
                print(f"  ✓ Downloaded ({size_mb:.1f} MB)")
            else:
                failed.append(filename)
                print(f"  ✗ Failed: {result.stderr}")
        
        except Exception as e:
            failed.append(filename)
            print(f"  ✗ Error: {e}")
    
    print("-" * 60)
    print(f"Download complete: {success}/{total} successful")
    
    if failed:
        print(f"\nFailed downloads ({len(failed)}):")
        for f in failed[:10]:
            print(f"  - {f}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    
    return success, failed


def main():
    parser = argparse.ArgumentParser(description="Prepare slides for training")
    parser.add_argument("--download", action="store_true", help="Download slides from S3")
    parser.add_argument("--append", action="store_true", help="Append new slides to existing manifest")
    parser.add_argument("--train-per-class", type=int, default=400, help="Train slides per class")
    parser.add_argument("--val-per-class", type=int, default=100, help="Val slides per class")
    parser.add_argument("--test-per-class", type=int, default=100, help="Test slides per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    config = load_config()
    
    project_root = Path(__file__).parent.parent
    bcc_csv = project_root / "data" / "cobra" / "bcc_bcc.csv"
    images_csv = project_root / "data" / "cobra" / "bcc_images.csv"
    manifest_path = project_root / "data" / "manifests" / "slide_manifest.csv"
    wsi_dir = project_root / "data" / "raw_wsi"
    
    print("=" * 60)
    if args.append:
        print("Appending to existing slide manifest...")
    else:
        print("Creating balanced slide manifest...")
    print("=" * 60)
    
    dataset = create_slide_manifest_from_cobra(
        bcc_csv=str(bcc_csv),
        images_csv=str(images_csv),
        output_path=str(manifest_path),
        wsi_dir="data/raw_wsi",
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
        append=args.append,
    )
    
    if args.download:
        success, failed = download_slides(
            manifest_path=str(manifest_path),
            wsi_dir=str(wsi_dir),
        )
        
        if failed:
            print(f"\nWarning: {len(failed)} slides failed to download")
            print("You can re-run the script to retry failed downloads")
    else:
        print("\nTo download slides, run with --download flag:")
        print(f"  python scripts/01_prepare_slides.py --download")
    
    print("\nManifest created at:", manifest_path)


if __name__ == "__main__":
    main()
