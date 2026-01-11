#!/usr/bin/env python3
"""
Script 04: Run slide-level inference.

This script:
1. Loads trained patch classifier
2. Runs tile-level inference on test set
3. Aggregates tile predictions to slide predictions
4. Outputs slide-level results

Usage:
    python scripts/04_inference.py [--checkpoint best_model.pt]
"""

import os
import sys
import argparse
import csv
import logging
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference import TileInference, SlideAggregator, AggregationConfig
from src.inference.slide_aggregator import run_slide_inference
from src.evaluation import print_classification_report


def main():
    parser = argparse.ArgumentParser(description="Run slide-level inference")
    parser.add_argument("--tile-manifest", type=str, default="data/manifests/tile_manifest.csv",
                        help="Path to tile manifest")
    parser.add_argument("--checkpoint", type=str, default="logs/checkpoints/patch_classifier.pt",
                        help="Model checkpoint path")
    parser.add_argument("--output", type=str, default="data/manifests/slide_predictions.csv",
                        help="Output predictions CSV")
    parser.add_argument("--split", type=str, default="test", help="Split to evaluate")
    parser.add_argument("--top-k", type=int, default=50, help="Top K tiles for aggregation")
    parser.add_argument("--percentile", type=float, default=90, help="Percentile for scoring")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    project_root = Path(__file__).parent.parent
    tile_manifest = project_root / args.tile_manifest
    checkpoint = project_root / args.checkpoint
    output_path = project_root / args.output
    
    if not tile_manifest.exists():
        logging.error(f"Tile manifest not found: {tile_manifest}")
        sys.exit(1)
    
    if not checkpoint.exists():
        logging.error(f"Checkpoint not found: {checkpoint}")
        sys.exit(1)
    
    logging.info(f"Loading model from {checkpoint}")
    tile_inference = TileInference(
        checkpoint_path=str(checkpoint),
        device=device,
        batch_size=64,
    )
    
    agg_config = AggregationConfig(
        method="percentile",
        top_k=args.top_k,
        percentile=args.percentile,
        threshold=args.threshold,
        use_ratio_rule=True,
        ratio_threshold=0.15,
        tile_prob_threshold=0.7,
    )
    
    logging.info("=" * 60)
    logging.info(f"Running slide inference on {args.split} split")
    logging.info("=" * 60)
    
    results = run_slide_inference(
        tile_manifest_path=str(tile_manifest),
        tile_inference=tile_inference,
        aggregation_config=agg_config,
        split=args.split,
    )
    
    logging.info(f"Processed {len(results)} slides")
    
    os.makedirs(output_path.parent, exist_ok=True)
    
    fieldnames = [
        "slide_id", "true_label", "prediction", "score",
        "malignant_ratio", "num_tiles", "high_prob_tiles",
        "mean_prob", "max_prob", "top_5_probs"
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for r in results:
            r["top_5_probs"] = str(r["top_5_probs"])
            writer.writerow(r)
    
    logging.info(f"Results saved to: {output_path}")
    
    y_true = [r["true_label"] for r in results]
    y_pred = [r["prediction"] for r in results]
    y_proba = [r["score"] for r in results]
    
    print_classification_report(y_true, y_pred, y_proba, title=f"Slide-Level Results ({args.split})")
    
    logging.info("\nExample predictions:")
    for r in results[:5]:
        logging.info(
            f"  {r['slide_id']}: true={r['true_label']}, pred={r['prediction']}, "
            f"score={r['score']:.3f}, ratio={r['malignant_ratio']:.3f}"
        )


if __name__ == "__main__":
    main()
