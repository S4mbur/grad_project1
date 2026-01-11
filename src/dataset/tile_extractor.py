"""
Tile extraction from Whole Slide Images (WSI).
Extracts tiles with quality control filtering.
"""

import os
import csv
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

import numpy as np
import cv2
from PIL import Image

try:
    import openslide
except ImportError:
    openslide = None
    logging.warning("openslide not available. Install with: pip install openslide-python")


@dataclass
class TileExtractionConfig:
    """Configuration for tile extraction."""
    tile_size: int = 512
    target_mpp: float = 0.5
    max_tiles_per_slide: int = 500
    min_tissue_fraction: float = 0.3
    blur_threshold: float = 80.0
    min_nuclei_score: float = 0.08
    jpeg_quality: int = 90
    seed: int = 42


class TileExtractor:
    """Extract tiles from WSI with quality control."""
    
    def __init__(self, config: Optional[TileExtractionConfig] = None):
        self.config = config or TileExtractionConfig()
        self.rng = random.Random(self.config.seed)
    
    def _make_tissue_mask(self, slide: "openslide.OpenSlide") -> Tuple[np.ndarray, int]:
        """Create tissue mask from low-resolution thumbnail."""
        level = slide.level_count - 1
        w, h = slide.level_dimensions[level]
        
        thumb = slide.read_region((0, 0), level, (w, h)).convert("RGB")
        img = np.array(thumb)
        
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        
        tissue = (sat > 20) & (val < 245)
        
        tissue_u8 = (tissue.astype(np.uint8) * 255)
        kernel = np.ones((5, 5), np.uint8)
        tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
        tissue_u8 = cv2.morphologyEx(tissue_u8, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return tissue_u8 > 0, level
    
    def _tissue_fraction(self, rgb: np.ndarray) -> float:
        """Estimate tissue fraction in a tile."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        tissue = (sat > 20) & (val < 245)
        return float(tissue.mean())
    
    def _blur_score(self, rgb: np.ndarray) -> float:
        """Laplacian variance as blur metric (higher = sharper)."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())
    
    def _nuclei_score(self, rgb: np.ndarray) -> float:
        """Simple nuclei/cellularity proxy (fraction of dark pixels)."""
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        return float((gray < 0.65).mean())
    
    def _get_best_level(self, slide: "openslide.OpenSlide") -> Tuple[int, float]:
        """Find the best level for target MPP."""
        try:
            mpp = float(slide.properties.get('openslide.mpp-x', 0.485))
        except:
            mpp = 0.485
        
        target_mpp = self.config.target_mpp
        best_level = 0
        best_diff = abs(mpp - target_mpp)
        
        for level in range(slide.level_count):
            downsample = slide.level_downsamples[level]
            level_mpp = mpp * downsample
            diff = abs(level_mpp - target_mpp)
            if diff < best_diff:
                best_diff = diff
                best_level = level
        
        actual_mpp = mpp * slide.level_downsamples[best_level]
        return best_level, actual_mpp
    
    def extract_tiles(
        self,
        slide_path: str,
        output_dir: str,
        slide_id: str,
        label: str,
        split: str,
    ) -> List[Dict]:
        """
        Extract tiles from a single WSI.
        
        Returns:
            List of tile metadata dictionaries
        """
        if openslide is None:
            raise RuntimeError("openslide not installed")
        
        slide = openslide.OpenSlide(slide_path)
        
        level, actual_mpp = self._get_best_level(slide)
        level_dims = slide.level_dimensions[level]
        downsample = slide.level_downsamples[level]
        
        tissue_mask, mask_level = self._make_tissue_mask(slide)
        mask_downsample = slide.level_downsamples[mask_level]
        
        tile_size_l0 = int(self.config.tile_size * downsample)
        
        w0, h0 = slide.dimensions
        positions = []
        
        step = tile_size_l0
        for y in range(0, h0 - tile_size_l0 + 1, step):
            for x in range(0, w0 - tile_size_l0 + 1, step):
                mask_x = int(x / mask_downsample)
                mask_y = int(y / mask_downsample)
                mask_tile_size = max(1, int(tile_size_l0 / mask_downsample))
                
                if mask_x + mask_tile_size > tissue_mask.shape[1]:
                    continue
                if mask_y + mask_tile_size > tissue_mask.shape[0]:
                    continue
                
                mask_region = tissue_mask[mask_y:mask_y + mask_tile_size, 
                                         mask_x:mask_x + mask_tile_size]
                
                if mask_region.mean() > 0.5:
                    positions.append((x, y))
        
        self.rng.shuffle(positions)
        max_attempts = min(len(positions), self.config.max_tiles_per_slide * 3)
        
        tiles_metadata = []
        tile_output_dir = os.path.join(output_dir, slide_id)
        os.makedirs(tile_output_dir, exist_ok=True)
        
        tile_count = 0
        for x, y in positions[:max_attempts]:
            if tile_count >= self.config.max_tiles_per_slide:
                break
            
            tile = slide.read_region((x, y), level, 
                                     (self.config.tile_size, self.config.tile_size))
            tile_rgb = np.array(tile.convert("RGB"))
            
            tissue_frac = self._tissue_fraction(tile_rgb)
            if tissue_frac < self.config.min_tissue_fraction:
                continue
            
            blur = self._blur_score(tile_rgb)
            if blur < self.config.blur_threshold:
                continue
            
            nuclei = self._nuclei_score(tile_rgb)
            if nuclei < self.config.min_nuclei_score:
                continue
            
            tile_filename = f"{slide_id}_tile_{tile_count:04d}.jpg"
            tile_path = os.path.join(tile_output_dir, tile_filename)
            
            pil_tile = Image.fromarray(tile_rgb)
            pil_tile.save(tile_path, "JPEG", quality=self.config.jpeg_quality)
            
            tiles_metadata.append({
                "tile_path": tile_path,
                "slide_id": slide_id,
                "label": label,
                "split": split,
                "x": x,
                "y": y,
                "level": level,
                "tissue_fraction": round(tissue_frac, 3),
                "blur_score": round(blur, 1),
                "nuclei_score": round(nuclei, 3),
            })
            
            tile_count += 1
        
        slide.close()
        return tiles_metadata
    
    def extract_from_manifest(
        self,
        slide_manifest_path: str,
        output_dir: str,
        tile_manifest_path: str,
    ) -> str:
        """
        Extract tiles from all slides in manifest.
        
        Returns:
            Path to generated tile manifest
        """
        from .slide_dataset import load_slide_manifest
        
        dataset = load_slide_manifest(slide_manifest_path)
        all_tiles = []
        
        logging.info(f"Extracting tiles from {len(dataset)} slides...")
        
        for i, slide in enumerate(dataset.slides):
            if not os.path.exists(slide.local_path):
                logging.warning(f"Slide not found: {slide.local_path}")
                continue
            
            logging.info(f"[{i+1}/{len(dataset)}] Processing {slide.slide_id}...")
            
            try:
                tiles = self.extract_tiles(
                    slide_path=slide.local_path,
                    output_dir=output_dir,
                    slide_id=slide.slide_id,
                    label=slide.label,
                    split=slide.split,
                )
                all_tiles.extend(tiles)
                logging.info(f"  Extracted {len(tiles)} tiles")
            except Exception as e:
                logging.error(f"  Error: {e}")
                continue
        
        os.makedirs(os.path.dirname(tile_manifest_path), exist_ok=True)
        
        fieldnames = ["tile_path", "slide_id", "label", "split", "x", "y", 
                     "level", "tissue_fraction", "blur_score", "nuclei_score"]
        
        with open(tile_manifest_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for tile in all_tiles:
                writer.writerow(tile)
        
        logging.info(f"Saved tile manifest: {tile_manifest_path}")
        logging.info(f"Total tiles: {len(all_tiles)}")
        
        return tile_manifest_path
