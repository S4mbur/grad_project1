#!/usr/bin/env python3
"""
Script 05: Evaluate model and generate report.

This script:
1. Loads slide predictions
2. Computes comprehensive metrics
3. Generates visualizations (confusion matrix, etc.)
4. Outputs evaluation report

Usage:
    python scripts/05_evaluate.py
"""

import os
import sys
import argparse
import csv
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation import compute_metrics, print_classification_report, plot_confusion_matrix


def main():
    parser = argparse.ArgumentParser(description="Evaluate model and generate report")
    parser.add_argument("--predictions", type=str, default="data/manifests/slide_predictions.csv",
                        help="Path to slide predictions CSV")
    parser.add_argument("--output-dir", type=str, default="reports",
                        help="Output directory for reports")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    project_root = Path(__file__).parent.parent
    predictions_path = project_root / args.predictions
    output_dir = project_root / args.output_dir
    
    if not predictions_path.exists():
        logging.error(f"Predictions not found: {predictions_path}")
        logging.error("Run 04_inference.py first")
        sys.exit(1)
    
    with open(predictions_path, 'r') as f:
        reader = csv.DictReader(f)
        results = list(reader)
    
    logging.info(f"Loaded {len(results)} slide predictions")
    
    y_true = [r["true_label"] for r in results]
    y_pred = [r["prediction"] for r in results]
    y_proba = [float(r["score"]) for r in results]
    
    metrics = print_classification_report(y_true, y_pred, y_proba, title="Final Evaluation Report")
    
    os.makedirs(output_dir / "figures", exist_ok=True)
    os.makedirs(output_dir / "tables", exist_ok=True)
    
    plot_confusion_matrix(
        y_true, y_pred,
        save_path=str(output_dir / "figures" / "confusion_matrix.png"),
        title="Slide-Level Confusion Matrix"
    )
    
    metrics_path = output_dir / "tables" / "metrics.csv"
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in metrics.items():
            if key != "confusion_matrix":
                writer.writerow([key, value])
    
    logging.info(f"Metrics saved to: {metrics_path}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Error Analysis")
    logging.info("=" * 60)
    
    fp_cases = [r for r in results if r["true_label"] == "benign" and r["prediction"] == "malignant"]
    logging.info(f"\nFalse Positives: {len(fp_cases)}")
    for r in fp_cases[:5]:
        logging.info(f"  {r['slide_id']}: score={r['score']}, ratio={r['malignant_ratio']}")
    
    fn_cases = [r for r in results if r["true_label"] == "malignant" and r["prediction"] == "benign"]
    logging.info(f"\nFalse Negatives: {len(fn_cases)}")
    for r in fn_cases[:5]:
        logging.info(f"  {r['slide_id']}: score={r['score']}, ratio={r['malignant_ratio']}")
    
    logging.info("\n" + "=" * 60)
    logging.info("Summary")
    logging.info("=" * 60)
    logging.info(f"Total slides: {len(results)}")
    logging.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    logging.info(f"Sensitivity (Recall): {metrics['recall']:.4f}")
    logging.info(f"Specificity: {metrics['specificity']:.4f}")
    if metrics.get('auc'):
        logging.info(f"AUC-ROC: {metrics['auc']:.4f}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
