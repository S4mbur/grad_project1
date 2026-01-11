"""
Evaluate slide-level predictions and generate metrics.
"""
import os
import sys
import csv
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_curve, auc,
    confusion_matrix, classification_report, roc_curve
)


def main(slide_csv: str, output_dir: str, threshold: float = 0.5):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(slide_csv, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    test_rows = [r for r in rows if r["split"] == "test"]
    print(f"Total slides: {len(rows)}, Test slides: {len(test_rows)}")
    
    if len(test_rows) == 0:
        print("No test slides found!")
        return
    
    y_true = np.array([int(r["label_int"]) for r in test_rows])
    y_scores = np.array([float(r["slide_score_topk_mean"]) for r in test_rows])
    y_pred = (y_scores >= threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*50)
    print("SLIDE-LEVEL EVALUATION RESULTS")
    print("="*50)
    print(f"Threshold: {threshold}")
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Benign", "Malignant"]))
    
    metrics = {
        "threshold": threshold,
        "num_test_slides": len(test_rows),
        "accuracy": acc,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "benign_count": int(np.sum(y_true == 0)),
        "malignant_count": int(np.sum(y_true == 1)),
    }
    
    with open(os.path.join(output_dir, "slide_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Slide-Level ROC Curve', fontsize=14)
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slide_roc_curve.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, 'g-', linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Slide-Level Precision-Recall Curve', fontsize=14)
    plt.legend(loc='lower left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slide_pr_curve.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(7, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Slide-Level Confusion Matrix', fontsize=14)
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Benign', 'Malignant'], fontsize=11)
    plt.yticks(tick_marks, ['Benign', 'Malignant'], fontsize=11)
    
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slide_confusion_matrix.png"), dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    benign_scores = y_scores[y_true == 0]
    malignant_scores = y_scores[y_true == 1]
    
    plt.hist(benign_scores, bins=30, alpha=0.6, label='Benign', color='blue')
    plt.hist(malignant_scores, bins=30, alpha=0.6, label='Malignant', color='red')
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold ({threshold})')
    plt.xlabel('Slide Score (Top-K Mean)', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Slide Score Distribution by Class', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "slide_score_distribution.png"), dpi=150)
    plt.close()
    
    print(f"\n✓ Saved figures and metrics to: {output_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python src/evaluate_slides.py slide_predictions.csv output_dir [threshold]")
        sys.exit(1)
    
    slide_csv = sys.argv[1]
    output_dir = sys.argv[2]
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
    
    main(slide_csv, output_dir, threshold)
