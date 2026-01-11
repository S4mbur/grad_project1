#!/usr/bin/env python3
"""
Threshold Tuning Script

Test different decision thresholds to find optimal specificity/sensitivity balance.
"""

import os
import sys
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score

sys.path.insert(0, str(Path(__file__).parent))

def load_predictions(csv_path):
    """Load slide predictions from CSV."""
    predictions = {}
    true_labels = {}
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            slide_id = row['slide_id']
            true_label_str = row['true_label'].lower()
            true_label = 0 if true_label_str == 'benign' else 1
            pred_score = float(row['score'])
            
            predictions[slide_id] = pred_score
            true_labels[slide_id] = true_label
    
    return predictions, true_labels

def evaluate_threshold(true_labels, predictions, threshold):
    """Evaluate metrics at a specific threshold."""
    pred_labels = {sid: 1 if score >= threshold else 0 
                   for sid, score in predictions.items()}
    
    true_vals = [true_labels[sid] for sid in predictions.keys()]
    pred_vals = [pred_labels[sid] for sid in predictions.keys()]
    
    acc = accuracy_score(true_vals, pred_vals)
    prec = precision_score(true_vals, pred_vals, zero_division=0)
    recall = recall_score(true_vals, pred_vals, zero_division=0)
    f1 = f1_score(true_vals, pred_vals, zero_division=0)
    
    cm = confusion_matrix(true_vals, pred_vals)
    tn, fp, fn, tp = cm.ravel()
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    balanced_acc = (sensitivity + specificity) / 2
    
    auc = roc_auc_score(true_vals, [predictions[sid] for sid in predictions.keys()])
    
    return {
        'threshold': threshold,
        'accuracy': acc,
        'precision': prec,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'f1': f1,
        'balanced_acc': balanced_acc,
        'auc': auc,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn
    }

def main():
    predictions_path = Path(__file__).parent / "data/manifests/slide_predictions.csv"
    
    if not predictions_path.exists():
        print(f"Error: {predictions_path} not found")
        print("Run 04_inference.py first")
        return
    
    print("Loading predictions...")
    predictions, true_labels = load_predictions(predictions_path)
    print(f"Loaded {len(predictions)} predictions")
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    results = []
    
    print("\n" + "="*100)
    print("THRESHOLD TUNING RESULTS")
    print("="*100)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'Sensitivity':<14} {'Specificity':<13} {'F1-Score':<12} {'Balanced Acc':<14}")
    print("-"*100)
    
    for threshold in thresholds:
        result = evaluate_threshold(true_labels, predictions, threshold)
        results.append(result)
        
        print(f"{result['threshold']:<12.1f} "
              f"{result['accuracy']:<12.1%} "
              f"{result['sensitivity']:<14.1%} "
              f"{result['specificity']:<13.1%} "
              f"{result['f1']:<12.3f} "
              f"{result['balanced_acc']:<14.1%}")
    
    print("="*100)
    
    print("\n" + "="*100)
    print("BEST THRESHOLDS BY METRIC")
    print("="*100)
    
    best_acc = max(results, key=lambda x: x['accuracy'])
    best_spec = max(results, key=lambda x: x['specificity'])
    best_sens = max(results, key=lambda x: x['sensitivity'])
    best_f1 = max(results, key=lambda x: x['f1'])
    best_balanced = max(results, key=lambda x: x['balanced_acc'])
    
    print(f"\n📊 Best Accuracy:     threshold={best_acc['threshold']:.1f} → {best_acc['accuracy']:.1%}")
    print(f"📊 Best Specificity:  threshold={best_spec['threshold']:.1f} → {best_spec['specificity']:.1%}")
    print(f"📊 Best Sensitivity:  threshold={best_sens['threshold']:.1f} → {best_sens['sensitivity']:.1%}")
    print(f"📊 Best F1-Score:     threshold={best_f1['threshold']:.1f} → {best_f1['f1']:.3f}")
    print(f"📊 Best Balanced:     threshold={best_balanced['threshold']:.1f} → {best_balanced['balanced_acc']:.1%}")
    
    print("\n" + "="*100)
    print("RECOMMENDATION FOR 70% SPECIFICITY TARGET")
    print("="*100)
    
    target_spec = 0.70
    closest = min(results, key=lambda x: abs(x['specificity'] - target_spec))
    
    if closest['specificity'] >= target_spec:
        print(f"\n✅ Target specificity {target_spec:.0%} ACHIEVABLE")
        print(f"\n   Recommended Threshold: {closest['threshold']:.1f}")
        print(f"   - Accuracy: {closest['accuracy']:.1%}")
        print(f"   - Sensitivity: {closest['sensitivity']:.1%}")
        print(f"   - Specificity: {closest['specificity']:.1%}")
        print(f"   - F1-Score: {closest['f1']:.3f}")
        print(f"\n   Confusion Matrix:")
        print(f"   TP (Malignant detected): {closest['tp']}")
        print(f"   TN (Benign correctly identified): {closest['tn']}")
        print(f"   FP (Benign misclassified): {closest['fp']}")
        print(f"   FN (Malignant missed): {closest['fn']}")
    else:
        print(f"\n❌ Target specificity {target_spec:.0%} NOT ACHIEVABLE with current model")
        print(f"    Maximum achievable: {max(r['specificity'] for r in results):.1%}")
        print(f"\n    Recommendation: Use aggregation strategy or model fine-tuning")
    
    output_csv = Path(__file__).parent / "reports/threshold_tuning_results.csv"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\n📄 Detailed results saved to: {output_csv}")

if __name__ == "__main__":
    main()
