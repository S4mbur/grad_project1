## Overview

This project implements an end-to-end system for classifying dermatopathology slides as benign or malignant using a tile-based deep learning approach. The system processes gigapixel-scale whole slide images by:

1. **Tile Extraction**: Segmenting WSIs into 512×512 pixel tiles with quality filtering
2. **Patch Classification**: Classifying each tile using a ConvNeXt-Tiny model pretrained on ImageNet
3. **Slide-Level Aggregation**: Combining tile predictions using a conservative voting strategy
4. **Visualization**: Generating probability heatmaps for interpretability

## Project Structure

```
├── configs/              # Configuration files
├── data/                 # Data directory (not tracked)
│   ├── cobra/           # COBRA dataset metadata
│   ├── manifests/       # Train/val/test splits
│   └── tiles/           # Extracted tiles
├── logs/                 # Training logs and checkpoints
├── reports/              # Evaluation results and figures
├── scripts/              # Pipeline execution scripts
│   ├── 01_prepare_slides.py
│   ├── 02_extract_tiles.py
│   ├── 03_train.py
│   ├── 04_inference.py
│   └── 05_evaluate.py
└── src/                  # Source code
    ├── dataset/         # Data loading utilities
    ├── models/          # Model architectures
    ├── training/        # Training pipeline
    ├── inference/       # Inference utilities
    └── evaluation/      # Metrics and visualization
```

## Requirements

- Python 3.10+
- PyTorch 2.0+
- OpenSlide (for WSI processing)
- CUDA toolkit (for GPU acceleration)

Install dependencies:
```bash
pip install -r requirements_freeze.txt
```

## Usage

### 1. Prepare Slide List
```bash
python scripts/01_prepare_slides.py
```

### 2. Extract Tiles
```bash
python scripts/02_extract_tiles.py
```

### 3. Train Model
```bash
python scripts/03_train.py
```

### 4. Run Inference
```bash
python scripts/04_inference.py <slide_path> <checkpoint_path>
```

### 5. Evaluate Results
```bash
python scripts/05_evaluate.py
```

## Model Architecture

- **Backbone**: ConvNeXt-Tiny pretrained on ImageNet
- **Input**: 224×224 RGB tiles (resized from 512×512)
- **Output**: Binary classification (benign/malignant)

## Results (Demo Dataset)

| Metric | Patch-Level | Slide-Level |
|--------|-------------|-------------|
| Accuracy | 70.8% | 93.4% |
| Sensitivity | - | 96.6% |
| Specificity | - | 90.6% |
| AUC-ROC | 0.79 | 0.99 |

*Results on 400-slide demo subset of COBRA dataset*

## Dataset

This project uses the [COBRA dataset](https://portal.gdc.cancer.gov/), a publicly available collection of dermatopathology whole slide images for basal cell carcinoma (BCC) detection.

## License

This project is developed as part of a graduation project at Istanbul Technical University.

## Author

Burak Yalçın - Istanbul Technical University, Computer Engineering
