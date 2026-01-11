import csv, sys
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def load_model(ckpt_path, device):
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model

def main(manifest_csv, ckpt_path, out_csv, slide_id_filter=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = list(csv.DictReader(open(manifest_csv)))
    rows = [r for r in rows if r["split"] == "test"]

    if slide_id_filter:
        slide_id_filter = set(slide_id_filter.split(","))
        rows = [r for r in rows if r["slide_id"] in slide_id_filter]

    tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])
    model = load_model(ckpt_path, device)
    sm = nn.Softmax(dim=1)

    out = []
    for r in tqdm(rows, desc="tile probs"):
        img = Image.open(r["tile_path"]).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)
        with torch.no_grad():
            p_mal = float(sm(model(x))[0,1].item())
        out.append({
            "slide_id": r["slide_id"],
            "tile_path": r["tile_path"],
            "true_label": r["label"],
            "p_malignant": f"{p_mal:.6f}",
        })

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    print("Wrote:", out_csv, "rows=", len(out))

if __name__ == "__main__":
    manifest, ckpt, out = sys.argv[1], sys.argv[2], sys.argv[3]
    filt = sys.argv[4] if len(sys.argv) >= 5 else None
    main(manifest, ckpt, out, slide_id_filter=filt)
