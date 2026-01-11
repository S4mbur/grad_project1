import os
import sys
import csv
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


def label_to_int(s: str) -> int:
    s = (s or "").strip().lower()
    if s == "benign":
        return 0
    if s == "malignant":
        return 1
    raise ValueError(f"Invalid label='{s}' expected benign/malignant")


class TileDataset(Dataset):
    def __init__(self, rows: List[Dict[str, str]], tfm):
        self.rows = rows
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img = Image.open(r["tile_path"]).convert("RGB")
        x = self.tfm(img)
        y = label_to_int(r["label"])
        return x, y, r


def load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def load_model(ckpt_path: str, device: str):
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def main(manifest_csv: str, ckpt_path: str, out_csv: str, topk_per_class: int = 500):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = load_rows(manifest_csv)
    train_rows = [r for r in rows if r["split"].strip().lower() == "train"]
    if not train_rows:
        raise RuntimeError("No train rows found.")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    ds = TileDataset(train_rows, tfm)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    model = load_model(ckpt_path, device)
    softmax = nn.Softmax(dim=1)

    hard_fp: List[Tuple[float, Dict[str, str]]] = []
    hard_fn: List[Tuple[float, Dict[str, str]]] = []

    for xb, yb, meta in tqdm(dl, desc="mining(train)"):
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = softmax(logits).detach().cpu()

        for i in range(len(yb)):
            true_y = int(yb[i])
            p_m = float(probs[i, 1])

            r = {k: meta[k][i] for k in meta}

            if true_y == 0:
                hard_fp.append((p_m, r))
            else:
                hard_fn.append((1.0 - p_m, r))

    hard_fp.sort(key=lambda x: x[0], reverse=True)
    hard_fn.sort(key=lambda x: x[0], reverse=True)

    sel = []
    sel += [("hard_fp", score, r) for score, r in hard_fp[:topk_per_class]]
    sel += [("hard_fn", score, r) for score, r in hard_fn[:topk_per_class]]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = ["kind", "hard_score"] + list(train_rows[0].keys())
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for kind, score, r in sel:
            row = {"kind": kind, "hard_score": f"{score:.6f}"}
            row.update(r)
            w.writerow(row)

    print("Wrote:", out_csv)
    print("Selected:", len(sel), f"(hard_fp={min(topk_per_class,len(hard_fp))}, hard_fn={min(topk_per_class,len(hard_fn))})")


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Usage: python src/mine_hard_tiles.py <manifest.csv> <ckpt.pt> <out.csv> [topk_per_class]")
        sys.exit(1)
    topk = int(sys.argv[4]) if len(sys.argv) == 5 else 500
    main(sys.argv[1], sys.argv[2], sys.argv[3], topk_per_class=topk)
