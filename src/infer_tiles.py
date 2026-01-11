import os
import sys
import csv
from collections import defaultdict
from typing import Dict, List

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def load_model(ckpt_path: str, device: str):
    """Load model from checkpoint - supports both ResNet and ConvNeXt"""
    from torchvision.models import resnet18, convnext_tiny, convnext_small, convnext_base
    
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    architecture = state.get("architecture", "resnet18")
    
    print(f"Loading {architecture} model...")
    
    if architecture == "resnet18":
        model = resnet18(weights=None)
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
    
    model.load_state_dict(state["model"])
    model.eval()
    model.to(device)
    return model


def main(manifest_csv: str, ckpt_path: str, out_csv: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    model = load_model(ckpt_path, device)
    
    with open(manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Processing {len(rows)} tiles...")

    rows_out: List[Dict[str, str]] = []
    
    for r in tqdm(rows, desc="Inference"):
        img = Image.open(r["tile_path"]).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            p_malignant = float(probs[1])

        r2 = dict(r)
        r2["p_malignant"] = f"{p_malignant:.6f}"
        rows_out.append(r2)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = list(rows_out[0].keys()) if rows_out else []
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows_out)

    print(f"✓ Wrote tile predictions: {out_csv}")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python src/infer_tiles.py tiles_manifest.csv logs/patch_model_demo.pt output_tiles_with_scores.csv")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
