"""
Tile-level inference.
"""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from ..models import PatchClassifier, load_checkpoint
from ..utils.transforms import get_eval_transforms


class TileInferenceDataset(Dataset):
    """Simple dataset for tile inference."""
    
    def __init__(self, tile_paths: List[str], transform=None):
        self.tile_paths = tile_paths
        self.transform = transform or get_eval_transforms()
    
    def __len__(self):
        return len(self.tile_paths)
    
    def __getitem__(self, idx):
        path = self.tile_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, path


class TileInference:
    """Tile-level inference using trained patch classifier."""
    
    def __init__(
        self,
        model: Optional[PatchClassifier] = None,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 64,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        
        if model is not None:
            self.model = model.to(self.device)
        elif checkpoint_path is not None:
            self.model, _, _ = load_checkpoint(checkpoint_path, self.device)
        else:
            raise ValueError("Either model or checkpoint_path must be provided")
        
        self.model.eval()
        self.softmax = nn.Softmax(dim=1)
        self.transform = get_eval_transforms()
    
    @torch.no_grad()
    def predict_tiles(self, tile_paths: List[str]) -> List[Dict]:
        """
        Predict probabilities for a list of tile paths.
        
        Returns:
            List of dicts with tile_path, prob_benign, prob_malignant
        """
        dataset = TileInferenceDataset(tile_paths, self.transform)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        
        results = []
        
        for images, paths in tqdm(loader, desc="Tile inference", leave=False):
            images = images.to(self.device)
            
            logits = self.model(images)
            probs = self.softmax(logits)
            
            for i, path in enumerate(paths):
                results.append({
                    "tile_path": path,
                    "prob_benign": probs[i, 0].item(),
                    "prob_malignant": probs[i, 1].item(),
                })
        
        return results
    
    @torch.no_grad()
    def predict_single(self, tile_path: str) -> Tuple[float, float]:
        """
        Predict probability for a single tile.
        
        Returns:
            Tuple of (prob_benign, prob_malignant)
        """
        image = Image.open(tile_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        logits = self.model(image)
        probs = self.softmax(logits)
        
        return probs[0, 0].item(), probs[0, 1].item()
