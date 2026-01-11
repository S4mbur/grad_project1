import os
import sys
import csv
from typing import List

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from torchvision import transforms
from torchvision.models import resnet18, convnext_tiny, convnext_small, convnext_base

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(ckpt_path: str, device: str):
    """Load model from checkpoint - supports both ResNet and ConvNeXt"""
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    architecture = state.get("architecture", "resnet18")
    
    print(f"Loading {architecture} model for GradCAM...")
    
    if architecture == "resnet18":
        model = resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 2)
        target_layer_name = "layer4"
    elif architecture == "convnext_tiny":
        model = convnext_tiny(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
        target_layer_name = "features.7"
    elif architecture == "convnext_small":
        model = convnext_small(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
        target_layer_name = "features.7"
    elif architecture == "convnext_base":
        model = convnext_base(weights=None)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
        target_layer_name = "features.7"
    else:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    model.load_state_dict(state["model"])
    model.eval()
    model.to(device)
    return model, architecture


def top_tiles(tile_scores_csv: str, n: int = 3) -> List[str]:
    rows = []
    with open(tile_scores_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((float(r["p_malignant"]), r["tile_path"]))
    rows.sort(reverse=True, key=lambda x: x[0])
    return [p for _, p in rows[:n]]


def main(tile_scores_csv: str, ckpt_path: str, out_dir: str, n: int = 3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(out_dir, exist_ok=True)

    model, architecture = load_model(ckpt_path, device)

    if architecture == "resnet18":
        target_layers = [model.layer4[-1]]
    else:
        target_layers = [model.features[7][-1]]

    cam = GradCAM(model=model, target_layers=target_layers)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    
    tfm_vis = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    paths = top_tiles(tile_scores_csv, n=n)
    print("Selected tiles:")
    for p in paths:
        print(" -", p)

    for i, path in enumerate(paths):
        img_pil = Image.open(path).convert("RGB")
        img_resized = img_pil.resize((224, 224), resample=Image.BILINEAR)

        img_np = np.array(img_resized).astype(np.float32) / 255.0
        input_tensor = tfm(img_pil).unsqueeze(0).to(device)

        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0, :]

        cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

        out_path = os.path.join(out_dir, f"gradcam_{i:02d}.png")
        Image.fromarray(cam_image).save(out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Usage: python src/gradcam_demo.py tiles_with_scores.csv ckpt.pt out_dir [n]")
        sys.exit(1)
    n = int(sys.argv[4]) if len(sys.argv) == 5 else 3
    main(sys.argv[1], sys.argv[2], sys.argv[3], n=n)
