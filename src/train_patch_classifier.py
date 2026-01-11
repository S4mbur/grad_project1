import os
import sys
import csv
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


@dataclass
class TrainConfig:
    manifest_csv: str
    batch_size: int = 16
    num_workers: int = 2
    lr: float = 1e-4
    epochs: int = 3
    img_size: int = 224
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_ckpt: str = "logs/patch_model_cobra.pt"


def label_to_int(label: str) -> int:
    """
    Map tile-level string labels to integer classes.
      benign -> 0
      malignant -> 1
    """
    s = str(label).strip().lower()
    if s == "benign":
        return 0
    if s == "malignant":
        return 1
    raise ValueError(f"Unknown label value: {label!r} (expected 'benign' or 'malignant')")


class TileDataset(Dataset):
    def __init__(self, rows: List[dict], tfm):
        self.rows = rows
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img = Image.open(r["tile_path"]).convert("RGB")
        x = self.tfm(img)

        y_int = label_to_int(r["label"])
        y = torch.tensor(y_int, dtype=torch.long)
        return x, y


def load_rows(manifest_csv: str) -> List[dict]:
    with open(manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required = {"tile_path", "label", "split"}
    missing = required - set(rows[0].keys()) if rows else required
    if missing:
        raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")
    return rows


def filter_by_split(rows: List[dict], split_name: str) -> List[dict]:
    s = split_name.strip().lower()
    out = [r for r in rows if str(r["split"]).strip().lower() == s]
    return out


def print_class_balance(name: str, rows: List[dict]):
    c0 = sum(1 for r in rows if str(r["label"]).strip().lower() == "benign")
    c1 = sum(1 for r in rows if str(r["label"]).strip().lower() == "malignant")
    print(f"{name}: n={len(rows)}  benign={c0}  malignant={c1}")


def main(cfg: TrainConfig):
    os.makedirs("logs", exist_ok=True)

    rows = load_rows(cfg.manifest_csv)
    if len(rows) < 50:
        raise RuntimeError("Not enough tiles in manifest. Generate more tiles first.")

    train_rows = filter_by_split(rows, "train")
    val_rows = filter_by_split(rows, "val")
    test_rows = filter_by_split(rows, "test")

    if len(train_rows) == 0 or len(val_rows) == 0:
        raise RuntimeError("Train/val splits are empty. Check 'split' values in the manifest.")

    print_class_balance("train", train_rows)
    print_class_balance("val", val_rows)
    if len(test_rows) > 0:
        print_class_balance("test", test_rows)

    tfm_train = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])
    tfm_eval = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
    ])

    train_ds = TileDataset(train_rows, tfm_train)
    val_ds = TileDataset(val_rows, tfm_eval)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True
    )

    from torchvision.models import resnet18
    model = resnet18(weights="DEFAULT")
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(cfg.device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    crit = nn.CrossEntropyLoss()

    print("Device:", cfg.device)
    if cfg.device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    best_val_acc = -1.0

    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0

        for xb, yb in tqdm(train_loader, desc=f"train epoch {epoch+1}/{cfg.epochs}"):
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()

            running += float(loss.item()) * xb.size(0)

        train_loss = running / len(train_ds)

        model.eval()
        correct = 0
        total = 0
        val_loss_sum = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(cfg.device, non_blocking=True)
                yb = yb.to(cfg.device, non_blocking=True)
                logits = model(xb)
                loss = crit(logits, yb)

                val_loss_sum += float(loss.item()) * xb.size(0)
                pred = logits.argmax(dim=1)
                correct += int((pred == yb).sum().item())
                total += int(yb.numel())

        val_loss = val_loss_sum / len(val_ds)
        val_acc = correct / max(total, 1)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict()}, cfg.out_ckpt)
            print(f"Saved (best so far): {cfg.out_ckpt}")

    print("Best val_acc:", best_val_acc)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/train_patch_classifier.py <tiles_manifest.csv>")
        sys.exit(1)
    cfg = TrainConfig(manifest_csv=sys.argv[1])
    main(cfg)
