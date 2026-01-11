"""
Slide-level aggregation from tile predictions.
"""

import csv
from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class AggregationConfig:
    """Configuration for slide aggregation."""
    method: str = "percentile"
    top_k: int = 50
    percentile: float = 90.0
    threshold: float = 0.5
    use_ratio_rule: bool = True
    ratio_threshold: float = 0.15
    tile_prob_threshold: float = 0.7


class SlideAggregator:
    """Aggregate tile predictions to slide-level prediction."""
    
    def __init__(self, config: Optional[AggregationConfig] = None):
        self.config = config or AggregationConfig()
    
    def aggregate(self, tile_probs: List[float]) -> Dict:
        """
        Aggregate tile probabilities to slide prediction.
        
        Args:
            tile_probs: List of P(malignant) for each tile
        
        Returns:
            Dict with score, prediction, and debug info
        """
        if not tile_probs:
            return {
                "score": 0.0,
                "prediction": "benign",
                "prediction_int": 0,
                "num_tiles": 0,
            }
        
        probs = np.array(tile_probs)
        n_tiles = len(probs)
        
        sorted_probs = np.sort(probs)[::-1]
        
        if self.config.method == "mean":
            score = float(np.mean(probs))
        elif self.config.method == "max":
            score = float(np.max(probs))
        elif self.config.method == "percentile":
            k = min(self.config.top_k, n_tiles)
            top_k_probs = sorted_probs[:k]
            score = float(np.percentile(top_k_probs, self.config.percentile))
        elif self.config.method == "topk":
            k = min(self.config.top_k, n_tiles)
            score = float(np.mean(sorted_probs[:k]))
        else:
            raise ValueError(f"Unknown aggregation method: {self.config.method}")
        
        high_prob_count = np.sum(probs >= self.config.tile_prob_threshold)
        malignant_ratio = high_prob_count / n_tiles if n_tiles > 0 else 0.0
        
        if self.config.use_ratio_rule:
            is_malignant = (
                score >= self.config.threshold and 
                malignant_ratio >= self.config.ratio_threshold
            )
        else:
            is_malignant = score >= self.config.threshold
        
        prediction = "malignant" if is_malignant else "benign"
        prediction_int = 1 if is_malignant else 0
        
        result = {
            "score": round(score, 4),
            "prediction": prediction,
            "prediction_int": prediction_int,
            "num_tiles": n_tiles,
            "malignant_ratio": round(malignant_ratio, 4),
            "high_prob_tiles": int(high_prob_count),
            "mean_prob": round(float(np.mean(probs)), 4),
            "max_prob": round(float(np.max(probs)), 4),
            "min_prob": round(float(np.min(probs)), 4),
            "top_5_probs": [round(p, 4) for p in sorted_probs[:5].tolist()],
        }
        
        return result


def aggregate_slide(
    tile_probs: List[float],
    config: Optional[AggregationConfig] = None,
) -> Dict:
    """Convenience function for slide aggregation."""
    aggregator = SlideAggregator(config)
    return aggregator.aggregate(tile_probs)


def run_slide_inference(
    tile_manifest_path: str,
    tile_inference,
    aggregation_config: Optional[AggregationConfig] = None,
    split: str = "test",
) -> List[Dict]:
    """
    Run full slide inference pipeline.
    
    Args:
        tile_manifest_path: Path to tile manifest CSV
        tile_inference: TileInference instance for tile predictions
        aggregation_config: Aggregation configuration
        split: Which split to run inference on
    
    Returns:
        List of slide-level results
    """
    from collections import defaultdict
    
    with open(tile_manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = [r for r in reader if r["split"] == split]
    
    slide_tiles = defaultdict(list)
    slide_labels = {}
    
    for row in rows:
        slide_id = row["slide_id"]
        slide_tiles[slide_id].append(row["tile_path"])
        slide_labels[slide_id] = row["label"]
    
    aggregator = SlideAggregator(aggregation_config)
    results = []
    
    for slide_id, tile_paths in slide_tiles.items():
        tile_results = tile_inference.predict_tiles(tile_paths)
        tile_probs = [r["prob_malignant"] for r in tile_results]
        
        agg_result = aggregator.aggregate(tile_probs)
        
        agg_result["slide_id"] = slide_id
        agg_result["true_label"] = slide_labels[slide_id]
        agg_result["true_label_int"] = 1 if slide_labels[slide_id] == "malignant" else 0
        
        results.append(agg_result)
    
    return results
