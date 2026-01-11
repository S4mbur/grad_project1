## Overview

End-to-end system for classifying dermatopathology slides as benign or malignant using a tile-based deep learning approach:

1. **Tile Extraction**: WSIs segmented into 512×512 tiles with quality filtering
2. **Patch Classification**: ConvNeXt-Tiny model pretrained on ImageNet
3. **Slide-Level Aggregation**: Conservative voting strategy
4. **Visualization**: Probability heatmaps

## Project Structure

```
├── configs/              # Configuration files
├── data/
│   ├── cobra/           # COBRA dataset metadata
│   └── manifests/       # Train/val/test splits
├── logs/                 # Training logs
├── reports/figures/      # Evaluation results and figures
├── scripts/              # Pipeline scripts
└── src/                  # Source code
    ├── dataset/         # Data loading
    ├── models/          # Model architectures
    ├── training/        # Training pipeline
    ├── inference/       # Inference utilities
    └── evaluation/      # Metrics
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- OpenSlide
- CUDA (optional, for GPU)

```bash
pip install -r requirements_freeze.txt
```

## Usage

```bash
python scripts/01_prepare_slides.py
python scripts/02_extract_tiles.py
python scripts/03_train.py
python scripts/04_inference.py
python scripts/05_evaluate.py
```

## Results

| Metric | Patch-Level | Slide-Level |
|--------|-------------|-------------|
| Accuracy | 70.8% | 93.4% |
| Sensitivity | - | 96.6% |
| Specificity | - | 90.6% |
| AUC-ROC | 0.79 | 0.99 |

*400-slide subset of COBRA dataset (BCC detection)*

## Author

Burak Yalçın - Istanbul Technical University
