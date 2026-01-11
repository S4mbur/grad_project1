"""
Evaluation metrics for classification.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


def compute_metrics(
    y_true: List,
    y_pred: List,
    y_proba: Optional[List] = None,
    pos_label: str = "malignant",
) -> Dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels (strings or ints)
        y_pred: Predicted labels (strings or ints)
        y_proba: Optional probability scores for positive class
        pos_label: Positive class label
    
    Returns:
        Dictionary of metrics
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.dtype == object or isinstance(y_true[0], str):
        labels = ["benign", "malignant"]
        y_true_int = np.array([1 if y == "malignant" else 0 for y in y_true])
        y_pred_int = np.array([1 if y == "malignant" else 0 for y in y_pred])
    else:
        labels = [0, 1]
        y_true_int = y_true.astype(int)
        y_pred_int = y_pred.astype(int)
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    accuracy = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    
    precision = precision_score(y_true_int, y_pred_int, zero_division=0)
    recall = recall_score(y_true_int, y_pred_int, zero_division=0)
    f1 = f1_score(y_true_int, y_pred_int, zero_division=0)
    
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    metrics = {
        "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(specificity, 4),
        "f1": round(f1, 4),
        "confusion_matrix": cm.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
    
    if y_proba is not None:
        try:
            auc = roc_auc_score(y_true_int, y_proba)
            metrics["auc"] = round(auc, 4)
        except:
            metrics["auc"] = None
    
    return metrics


def print_classification_report(
    y_true: List,
    y_pred: List,
    y_proba: Optional[List] = None,
    title: str = "Classification Report",
):
    """Print a formatted classification report."""
    metrics = compute_metrics(y_true, y_pred, y_proba)
    
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"              Benign  Malignant")
    print(f"Actual Benign    {metrics['tn']:4d}     {metrics['fp']:4d}")
    print(f"     Malignant   {metrics['fn']:4d}     {metrics['tp']:4d}")
    
    print(f"\nMetrics:")
    print(f"  Accuracy:          {metrics['accuracy']:.4f}")
    print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"  Precision:         {metrics['precision']:.4f}")
    print(f"  Recall/Sensitivity:{metrics['recall']:.4f}")
    print(f"  Specificity:       {metrics['specificity']:.4f}")
    print(f"  F1 Score:          {metrics['f1']:.4f}")
    
    if metrics.get("auc") is not None:
        print(f"  AUC-ROC:           {metrics['auc']:.4f}")
    
    print(f"{'='*60}\n")
    
    return metrics


def plot_confusion_matrix(
    y_true: List,
    y_pred: List,
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
):
    """Plot confusion matrix."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("matplotlib/seaborn not available for plotting")
        return
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if y_true.dtype == object or isinstance(y_true[0], str):
        labels = ["benign", "malignant"]
    else:
        labels = [0, 1]
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved confusion matrix to {save_path}")
    
    plt.close()
