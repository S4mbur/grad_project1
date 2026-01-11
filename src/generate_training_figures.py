#!/usr/bin/env python3
"""
Generate comprehensive figures for training report.
Creates publication-quality visualizations including:
- Training curves (loss, accuracy)
- Confusion matrix
- ROC curve and AUC
- Sample predictions with confidence
- Attention/GradCAM visualizations
- Model architecture diagram
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, 
    roc_curve, auc, precision_recall_curve
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont


plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
COLORS = {
    'benign': '#2ecc71',
    'malignant': '#e74c3c',
    'primary': '#3498db',
    'secondary': '#9b59b6'
}


class FigureGenerator:
    """Generate all training report figures."""
    
    def __init__(self, 
                 checkpoint_path: str,
                 manifest_csv: str,
                 output_dir: str = "reports/figures",
                 device: str = "cuda"):
        self.checkpoint_path = checkpoint_path
        self.manifest_csv = manifest_csv
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        
        self.model = self._load_model()
        
        self.history = self._load_history()
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        
        architecture = checkpoint.get('architecture', 'convnext_tiny')
        print(f"Architecture: {architecture}")
        
        from torchvision.models import convnext_tiny, convnext_small, convnext_base, resnet18, resnet34, resnet50
        import torch.nn as nn
        
        if architecture == "resnet18":
            model = resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif architecture == "resnet34":
            model = resnet34(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif architecture == "resnet50":
            model = resnet50(weights=None)
            model.fc = nn.Linear(model.fc.in_features, 2)
        elif architecture == "convnext_tiny":
            model = convnext_tiny(weights=None)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
        elif architecture == "convnext_small":
            model = convnext_small(weights=None)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
        elif architecture == "convnext_base":
            model = convnext_base(weights=None)
            model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        model.load_state_dict(checkpoint['model'])
        model.to(self.device)
        model.eval()
        
        self.architecture = architecture
        
        print(f"Loaded {architecture} model")
        return model
    
    def _load_history(self) -> Optional[Dict]:
        """Load training history if exists."""
        history_path = Path(self.checkpoint_path).parent / "training_history.json"
        if history_path.exists():
            with open(history_path, 'r') as f:
                return json.load(f)
        return None
    
    def generate_all(self):
        """Generate all figures for the report."""
        print("\nGenerating report figures...")
        print("="*60)
        
        if self.history:
            print("1. Training curves...")
            self.plot_training_curves()
        
        print("2. Confusion matrix and metrics...")
        self.plot_confusion_matrix()
        
        print("3. ROC curve...")
        self.plot_roc_curve()
        
        print("4. Precision-Recall curve...")
        self.plot_precision_recall()
        
        print("5. Sample predictions grid...")
        self.plot_sample_predictions()
        
        print("6. Model architecture diagram...")
        self.plot_architecture()
        
        print("7. Dataset class distribution...")
        self.plot_class_distribution()
        
        print("="*60)
        print(f"✓ All figures saved to: {self.output_dir}")
    
    def plot_training_curves(self):
        """Plot training and validation loss/accuracy curves."""
        if not self.history:
            print("  ⚠ No training history found, skipping...")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        axes[0].plot(epochs, self.history['train_loss'], 
                    label='Train Loss', linewidth=2, marker='o', markersize=4)
        axes[0].plot(epochs, self.history['val_loss'], 
                    label='Val Loss', linewidth=2, marker='s', markersize=4)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(epochs, self.history['train_acc'], 
                    label='Train Acc', linewidth=2, marker='o', markersize=4)
        axes[1].plot(epochs, self.history['val_acc'], 
                    label='Val Acc', linewidth=2, marker='s', markersize=4)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Accuracy', fontsize=12)
        axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'training_curves.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: training_curves.png/pdf")
    
    def plot_confusion_matrix(self):
        """Plot confusion matrix on test set."""
        y_true, y_pred, _ = self._get_predictions('test')
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Benign', 'Malignant'],
                   yticklabels=['Benign', 'Malignant'],
                   cbar_kws={'label': 'Count'},
                   ax=ax, square=True, annot_kws={'size': 16})
        
        ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
        ax.set_title('Confusion Matrix (Test Set)', fontsize=15, fontweight='bold', pad=20)
        
        accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
        ax.text(0.5, -0.15, f'Overall Accuracy: {accuracy:.1%}', 
               ha='center', transform=ax.transAxes, fontsize=12, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'confusion_matrix.pdf', bbox_inches='tight')
        plt.close()
        
        report = classification_report(y_true, y_pred, 
                                      target_names=['Benign', 'Malignant'],
                                      digits=4)
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write(report)
        
        print(f"  ✓ Saved: confusion_matrix.png/pdf")
        print(f"  ✓ Saved: classification_report.txt")
    
    def plot_roc_curve(self):
        """Plot ROC curve."""
        y_true, _, y_probs = self._get_predictions('test')
        
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(fpr, tpr, color=COLORS['primary'], linewidth=3,
               label=f'ROC curve (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        ax.set_title('ROC Curve - Malignant Detection', fontsize=15, fontweight='bold')
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'roc_curve.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: roc_curve.png/pdf (AUC={roc_auc:.4f})")
    
    def plot_precision_recall(self):
        """Plot Precision-Recall curve."""
        y_true, _, y_probs = self._get_predictions('test')
        
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        pr_auc = auc(recall, precision)
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.plot(recall, precision, color=COLORS['secondary'], linewidth=3,
               label=f'PR curve (AUC = {pr_auc:.4f})')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Recall', fontsize=13, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=13, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=15, fontweight='bold')
        ax.legend(loc="lower left", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'precision_recall.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'precision_recall.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: precision_recall.png/pdf (AUC={pr_auc:.4f})")
    
    def plot_sample_predictions(self, n_samples: int = 16):
        """Plot grid of sample predictions with confidence."""
        from src.train_patch_classifier_v2 import load_rows, split_rows
        
        rows = load_rows(self.manifest_csv)
        _, _, test_rows = split_rows(rows)
        
        benign = [r for r in test_rows if r['label'].lower() == 'benign']
        malignant = [r for r in test_rows if r['label'].lower() == 'malignant']
        
        np.random.seed(42)
        samples = (
            np.random.choice(benign, n_samples // 2, replace=False).tolist() +
            np.random.choice(malignant, n_samples // 2, replace=False).tolist()
        )
        np.random.shuffle(samples)
        
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        
        for idx, row in enumerate(samples):
            img = Image.open(row['tile_path']).convert('RGB')
            x = tfm(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            
            pred_class = int(probs[1] > 0.5)
            true_class = 0 if row['label'].lower() == 'benign' else 1
            confidence = probs[pred_class]
            
            ax = axes[idx]
            ax.imshow(img)
            ax.axis('off')
            
            color = 'green' if pred_class == true_class else 'red'
            pred_label = 'Malignant' if pred_class == 1 else 'Benign'
            true_label = 'Malignant' if true_class == 1 else 'Benign'
            
            title = f"Pred: {pred_label} ({confidence:.2%})\nTrue: {true_label}"
            ax.set_title(title, fontsize=9, color=color, fontweight='bold')
        
        plt.suptitle('Sample Predictions on Test Set', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'sample_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: sample_predictions.png")
    
    def plot_architecture(self):
        """Create model architecture diagram."""
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')
        
        arch_name = getattr(self, 'architecture', 'ConvNeXt-Tiny')
        
        stages = [
            {"name": "Input\nTile", "x": 1, "color": "#ecf0f1"},
            {"name": "Backbone\n" + arch_name.upper(), "x": 3, "color": "#3498db"},
            {"name": "Global\nPooling", "x": 5, "color": "#9b59b6"},
            {"name": "Classifier\nHead", "x": 7, "color": "#e67e22"},
            {"name": "Output\nLogits", "x": 9, "color": "#e74c3c"}
        ]
        
        for stage in stages:
            rect = plt.Rectangle((stage['x']-0.4, 4), 0.8, 2, 
                                facecolor=stage['color'], 
                                edgecolor='black', linewidth=2)
            ax.add_patch(rect)
            
            ax.text(stage['x'], 5, stage['name'], 
                   ha='center', va='center', fontsize=11, 
                   fontweight='bold', color='white')
            
            if stage['x'] < 9:
                ax.annotate('', xy=(stage['x']+0.5, 5), 
                          xytext=(stage['x']+0.4, 5),
                          arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        ax.text(1, 3, '224×224×3', ha='center', fontsize=9, style='italic')
        ax.text(9, 3, '2 classes', ha='center', fontsize=9, style='italic')
        
        ax.set_title(f'Model Architecture Pipeline: {arch_name.title()}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'architecture.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: architecture.png")
    
    def plot_class_distribution(self):
        """Plot class distribution across splits."""
        from src.train_patch_classifier_v2 import load_rows, split_rows, counts
        
        rows = load_rows(self.manifest_csv)
        train_rows, val_rows, test_rows = split_rows(rows)
        
        splits_data = {
            'Train': counts(train_rows),
            'Val': counts(val_rows),
            'Test': counts(test_rows)
        }
        
        splits = list(splits_data.keys())
        benign_counts = [splits_data[s][1] for s in splits]
        malignant_counts = [splits_data[s][2] for s in splits]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x = np.arange(len(splits))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, benign_counts, width, 
                      label='Benign', color=COLORS['benign'], alpha=0.8)
        bars2 = ax.bar(x + width/2, malignant_counts, width, 
                      label='Malignant', color=COLORS['malignant'], alpha=0.8)
        
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Dataset Split', fontsize=13, fontweight='bold')
        ax.set_ylabel('Number of Tiles', fontsize=13, fontweight='bold')
        ax.set_title('Class Distribution Across Dataset Splits', 
                    fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(splits)
        ax.legend(fontsize=12)
        ax.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved: class_distribution.png")
    
    def _get_predictions(self, split: str = 'test') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions for a dataset split."""
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from src.train_patch_classifier_v2 import (
            load_rows, split_rows, TileDataset, label_to_int
        )
        
        rows = load_rows(self.manifest_csv)
        train_rows, val_rows, test_rows = split_rows(rows)
        
        if split == 'train':
            split_rows_data = train_rows
        elif split == 'val':
            split_rows_data = val_rows
        else:
            split_rows_data = test_rows
        
        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        dataset = TileDataset(split_rows_data, tfm)
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        
        y_true_list = []
        y_pred_list = []
        y_probs_list = []
        
        self.model.eval()
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                logits = self.model(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()
                
                y_true_list.append(yb.numpy())
                y_pred_list.append(preds)
                y_probs_list.append(probs)
        
        y_true = np.concatenate(y_true_list)
        y_pred = np.concatenate(y_pred_list)
        y_probs = np.concatenate(y_probs_list)
        
        return y_true, y_pred, y_probs


def main():
    parser = argparse.ArgumentParser(description='Generate training report figures')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--manifest', type=str, required=True,
                       help='Path to tile manifest CSV')
    parser.add_argument('--output-dir', type=str, default='reports/figures',
                       help='Output directory for figures')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    generator = FigureGenerator(
        checkpoint_path=args.checkpoint,
        manifest_csv=args.manifest,
        output_dir=args.output_dir,
        device=args.device
    )
    
    generator.generate_all()
    
    print("\n✓ Figure generation complete!")


if __name__ == '__main__':
    main()
