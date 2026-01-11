"""
Training loop for patch classifier.
"""

import os
import logging
from typing import Dict, Optional, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..models import PatchClassifier, save_checkpoint


@dataclass
class TrainerConfig:
    """Training configuration."""
    epochs: int = 20
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    early_stopping_patience: int = 5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir: str = "logs/checkpoints"
    log_interval: int = 50


class Trainer:
    """Trainer class for patch classifier."""
    
    def __init__(
        self,
        model: PatchClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Optional[TrainerConfig] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainerConfig()
        
        self.device = self.config.device
        self.model = self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.epochs,
            eta_min=1e-6,
        )
        
        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            if batch_idx % self.config.log_interval == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{100. * correct / total:.2f}%",
                })
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return {"loss": avg_loss, "accuracy": accuracy}
    
    def train(self, checkpoint_name: str = "best_model.pt") -> Dict:
        """Full training loop with early stopping."""
        logging.info(f"Starting training for {self.config.epochs} epochs")
        logging.info(f"Device: {self.device}")
        logging.info(f"Train samples: {len(self.train_loader.dataset)}")
        logging.info(f"Val samples: {len(self.val_loader.dataset)}")
        
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.config.checkpoint_dir, checkpoint_name)
        
        for epoch in range(self.config.epochs):
            logging.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            train_metrics = self.train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            
            val_metrics = self.validate()
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])
            
            self.scheduler.step()
            
            logging.info(
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.4f}"
            )
            logging.info(
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )
            
            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.patience_counter = 0
                
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    metrics=val_metrics,
                    path=checkpoint_path,
                )
                logging.info(f"Saved best model (val_acc: {self.best_val_acc:.4f})")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.config.early_stopping_patience:
                logging.info(f"Early stopping at epoch {epoch + 1}")
                break
        
        logging.info(f"\nTraining complete. Best val accuracy: {self.best_val_acc:.4f}")
        
        return {
            "best_val_acc": self.best_val_acc,
            "history": self.history,
            "checkpoint_path": checkpoint_path,
        }


def train_model(
    model: PatchClassifier,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Optional[TrainerConfig] = None,
    checkpoint_name: str = "best_model.pt",
) -> Dict:
    """Convenience function to train a model."""
    trainer = Trainer(model, train_loader, val_loader, config)
    return trainer.train(checkpoint_name)
