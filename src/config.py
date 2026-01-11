"""
Configuration management for the WSI classification pipeline.
Loads YAML configs and provides easy access to parameters.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import yaml


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Load a YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent


def get_default_config() -> Dict[str, Any]:
    """Load the default configuration."""
    config_path = get_project_root() / "configs" / "default.yaml"
    return load_yaml_config(str(config_path))


@dataclass
class PathConfig:
    """Path configuration."""
    data_root: str = "data"
    raw_wsi_dir: str = "data/raw_wsi"
    tiles_dir: str = "data/tiles"
    manifests_dir: str = "data/manifests"
    logs_dir: str = "logs"
    checkpoints_dir: str = "logs/checkpoints"
    
    def __post_init__(self):
        """Convert to absolute paths and create directories."""
        root = get_project_root()
        self.data_root = str(root / self.data_root)
        self.raw_wsi_dir = str(root / self.raw_wsi_dir)
        self.tiles_dir = str(root / self.tiles_dir)
        self.manifests_dir = str(root / self.manifests_dir)
        self.logs_dir = str(root / self.logs_dir)
        self.checkpoints_dir = str(root / self.checkpoints_dir)
    
    def ensure_dirs(self):
        """Create all directories if they don't exist."""
        for path in [self.data_root, self.raw_wsi_dir, self.tiles_dir, 
                     self.manifests_dir, self.logs_dir, self.checkpoints_dir]:
            os.makedirs(path, exist_ok=True)


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "cobra"
    source: str = "s3://cobra-pathology/packages/bcc/"
    train_per_class: int = 400
    val_per_class: int = 100
    test_per_class: int = 100
    labels: Dict[int, str] = field(default_factory=lambda: {0: "benign", 1: "malignant"})
    classes: List[str] = field(default_factory=lambda: ["benign", "malignant"])
    num_classes: int = 2


@dataclass
class TileConfig:
    """Tile extraction configuration."""
    tile_size: int = 512
    target_mpp: float = 0.5
    max_tiles_per_slide: int = 500
    min_tissue_fraction: float = 0.3
    blur_threshold: float = 80.0
    jpeg_quality: int = 90
    seed: int = 42


@dataclass 
class ModelConfig:
    """Model configuration."""
    architecture: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    dropout: float = 0.5


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    num_workers: int = 4
    learning_rate: float = 0.0001
    weight_decay: float = 0.0001
    epochs: int = 20
    early_stopping_patience: int = 5
    seed: int = 42


@dataclass
class InferenceConfig:
    """Inference/aggregation configuration."""
    batch_size: int = 64
    top_k: int = 50
    percentile: float = 90.0
    threshold: float = 0.5
    use_ratio_rule: bool = True
    ratio_threshold: float = 0.15
    tile_prob_threshold: float = 0.7


@dataclass
class Config:
    """Main configuration class."""
    paths: PathConfig = field(default_factory=PathConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    tile: TileConfig = field(default_factory=TileConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Load configuration from YAML file."""
        cfg_dict = load_yaml_config(config_path)
        
        paths = PathConfig(**cfg_dict.get("paths", {}))
        
        dataset_dict = cfg_dict.get("dataset", {})
        subset = dataset_dict.pop("subset", {})
        dataset = DatasetConfig(
            name=dataset_dict.get("name", "cobra"),
            source=dataset_dict.get("source", ""),
            train_per_class=subset.get("train_per_class", 400),
            val_per_class=subset.get("val_per_class", 100),
            test_per_class=subset.get("test_per_class", 100),
            labels=dataset_dict.get("labels", {0: "benign", 1: "malignant"}),
            classes=dataset_dict.get("classes", ["benign", "malignant"]),
            num_classes=dataset_dict.get("num_classes", 2),
        )
        
        tile = TileConfig(**cfg_dict.get("tile_extraction", {}))
        
        model_dict = cfg_dict.get("model", {})
        model = ModelConfig(**model_dict)
        
        train_dict = cfg_dict.get("training", {})
        train_dict.pop("scheduler", None)
        training = TrainingConfig(**train_dict, seed=cfg_dict.get("seed", 42))
        
        inf_dict = cfg_dict.get("inference", {}).get("aggregation", {})
        inf_dict.pop("method", None)
        inference = InferenceConfig(
            batch_size=cfg_dict.get("inference", {}).get("batch_size", 64),
            top_k=inf_dict.get("top_k", 50),
            percentile=inf_dict.get("percentile", 90.0),
            threshold=inf_dict.get("threshold", 0.5),
            use_ratio_rule=inf_dict.get("use_ratio_rule", True),
            ratio_threshold=inf_dict.get("ratio_threshold", 0.15),
            tile_prob_threshold=inf_dict.get("tile_prob_threshold", 0.7),
        )
        
        return cls(
            paths=paths,
            dataset=dataset,
            tile=tile,
            model=model,
            training=training,
            inference=inference,
            seed=cfg_dict.get("seed", 42),
        )
    
    @classmethod
    def default(cls) -> "Config":
        """Load default configuration."""
        config_path = get_project_root() / "configs" / "default.yaml"
        return cls.from_yaml(str(config_path))


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file or use default."""
    if config_path:
        return Config.from_yaml(config_path)
    return Config.default()
