"""
Patch (tile) classifier model.
Binary classifier for WSI tiles supporting multiple architectures:
- ResNet (18, 34, 50)
- ConvNeXt (tiny, small, base)
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models


class PatchClassifier(nn.Module):
    """
    Patch classifier for WSI tiles.
    Supports multiple backbone architectures with pretrained weights.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.5,
        architecture: str = "resnet18",
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.architecture = architecture
        
        if architecture == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )
            
        elif architecture == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet34(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )
            
        elif architecture == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(num_features, num_classes),
            )
            
        elif architecture == "convnext_tiny":
            weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_tiny(weights=weights)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(num_features, num_classes)
            
        elif architecture == "convnext_small":
            weights = models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_small(weights=weights)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(num_features, num_classes)
            
        elif architecture == "convnext_base":
            weights = models.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.convnext_base(weights=weights)
            num_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Linear(num_features, num_classes)
            
        else:
            raise ValueError(f"Unknown architecture: {architecture}. "
                           f"Supported: resnet18, resnet34, resnet50, "
                           f"convnext_tiny, convnext_small, convnext_base")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns logits."""
        return self.backbone(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features before classification head."""
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)


def create_model(
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.5,
    architecture: str = "resnet18",
    device: Optional[str] = None,
) -> PatchClassifier:
    """Create a new patch classifier model."""
    model = PatchClassifier(
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        architecture=architecture,
    )
    
    if device:
        model = model.to(device)
    
    return model


def save_checkpoint(
    model: PatchClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: Dict,
    path: str,
):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "architecture": model.architecture,
        "num_classes": model.num_classes,
    }
    
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    device: Optional[str] = None,
    load_optimizer: bool = False,
) -> Tuple[PatchClassifier, Optional[Dict], Dict]:
    """
    Load model from checkpoint.
    
    Returns:
        Tuple of (model, optimizer_state, checkpoint_info)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(path, map_location=device)
    
    model = PatchClassifier(
        num_classes=checkpoint.get("num_classes", 2),
        pretrained=False,
        architecture=checkpoint.get("architecture", "resnet18"),
    )
    
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    
    optimizer_state = checkpoint.get("optimizer") if load_optimizer else None
    
    info = {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }
    
    return model, optimizer_state, info
