import sys
import csv
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from PIL import Image
from tqdm import tqdm


def label_to_int(label: str) -> int:
    s = str(label).strip().lower()
    if s == "benign":
        return 0
    if s == "malignant":
        return 1
    raise ValueError(f"Unknown label: {label!r}")


class TileDataset(Dataset):
    def __init__(self, rows, tfm):
        self.rows = rows
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r["tile_path"]).convert("RGB")
        x = self.tfm(img)
        y = torch.tensor(label_to_int(r["label"]), dtype=torch.long)
        return x, y, r["tile_path"]


def load_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def main(manifest_csv, ckpt_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rows = load_rows(manifest_csv)

    test_rows = [r for r in rows if str(r["split"]).strip().lower() == "test"]
    if not test_rows:
        raise RuntimeError("No test rows found in manifest.")

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    ds = TileDataset(test_rows, tfm)
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model = model.to(device)
    model.eval()

    cm = [[0, 0],
          [0, 0]]

    examples = defaultdict(list)
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        for xb, yb, paths in tqdm(dl, desc="eval(test)"):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            logits = model(xb)
            probs = softmax(logits)
            pred = probs.argmax(dim=1)

            for i in range(len(paths)):
                t = int(yb[i].item())
                p = int(pred[i].item())
                cm[t][p] += 1

                conf = float(probs[i, p].item())
                rec = (conf, paths[i], t, p)
                if t == p:
                    examples["correct"].append(rec)
                else:
                    examples["wrong"].append(rec)

    total = sum(sum(r) for r in cm)
    acc = (cm[0][0] + cm[1][1]) / max(total, 1)

    print("\n=== TEST RESULTS ===")
    print(f"Test tiles: {len(ds)}")
    print(f"Accuracy: {acc:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred)")
    print("          pred_benign  pred_malignant")
    print(f"true_benign     {cm[0][0]:4d}         {cm[0][1]:4d}")
    print(f"true_malignant  {cm[1][0]:4d}         {cm[1][1]:4d}")

    for k in ["wrong", "correct"]:
        examples[k].sort(key=lambda x: x[0], reverse=True)
        print(f"\nTop-5 {k} (confidence, path, true, pred):")
        for rec in examples[k][:5]:
            print(rec)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python src/eval_patch_classifier.py <tiles_manifest.csv> <ckpt.pt>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
