import os
import sys
import csv
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


@dataclass
class TrainConfig:
    manifest_csv: str
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 3e-4
    weight_decay: float = 1e-4
    epochs: int = 20
    img_size: int = 224
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_ckpt: str = "logs/patch_model_cobra_v2.pt"
    patience: int = 5
    use_weighted_sampler: bool = False
    label_smoothing: float = 0.05
    architecture: str = "convnext_tiny"


def label_to_int(s: str) -> int:
    s = (s or "").strip().lower()
    if s == "benign":
        return 0
    if s == "malignant":
        return 1
    raise ValueError(f"Invalid label='{s}'. Expected benign/malignant.")


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
        y = torch.tensor(label_to_int(r["label"]), dtype=torch.long)
        return x, y


def load_rows(manifest_csv: str) -> List[Dict[str, str]]:
    with open(manifest_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    need_cols = {"tile_path", "split", "label"}
    missing = need_cols - set(rows[0].keys()) if rows else need_cols
    if missing:
        raise RuntimeError(f"Manifest missing columns: {sorted(missing)}")

    return rows


def split_rows(rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    train = [r for r in rows if r["split"].strip().lower() == "train"]
    val = [r for r in rows if r["split"].strip().lower() == "val"]
    test = [r for r in rows if r["split"].strip().lower() == "test"]
    if not train or not val or not test:
        raise RuntimeError("Expected non-empty train/val/test splits in manifest.")
    return train, val, test


def counts(rows: List[Dict[str, str]]) -> Tuple[int, int, int]:
    n = len(rows)
    b = sum(1 for r in rows if r["label"].strip().lower() == "benign")
    m = sum(1 for r in rows if r["label"].strip().lower() == "malignant")
    return n, b, m


def make_weighted_sampler(rows: List[Dict[str, str]]) -> WeightedRandomSampler:
    ys = [label_to_int(r["label"]) for r in rows]
    c0 = sum(1 for y in ys if y == 0)
    c1 = sum(1 for y in ys if y == 1)
    if c0 == 0 or c1 == 0:
        raise RuntimeError("Sampler cannot be built: one class is empty.")
    w0 = 1.0 / c0
    w1 = 1.0 / c1
    weights = [w0 if y == 0 else w1 for y in ys]
    return WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)


def main(cfg: TrainConfig):
    os.makedirs("logs", exist_ok=True)

    rows = load_rows(cfg.manifest_csv)
    train_rows, val_rows, test_rows = split_rows(rows)

    ntr, btr, mtr = counts(train_rows)
    nva, bva, mva = counts(val_rows)
    nte, bte, mte = counts(test_rows)
    print(f"train: n={ntr}  benign={btr}  malignant={mtr}")
    print(f"val:   n={nva}  benign={bva}  malignant={mva}")
    print(f"test:  n={nte}  benign={bte}  malignant={mte}")

    tfm_train = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),

        transforms.ToTensor(),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.08), ratio=(0.3, 3.3), value=0),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    tfm_eval = transforms.Compose([
        transforms.Resize((cfg.img_size, cfg.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


    train_ds = TileDataset(train_rows, tfm_train)
    val_ds = TileDataset(val_rows, tfm_eval)
    test_ds = TileDataset(test_rows, tfm_eval)

    sampler = make_weighted_sampler(train_rows) if cfg.use_weighted_sampler else None

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=(cfg.num_workers > 0),
    )

    from torchvision.models import resnet18, convnext_tiny, convnext_small, convnext_base
    
    print(f"Creating model: {cfg.architecture}")
    if cfg.architecture == "resnet18":
        model = resnet18(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif cfg.architecture == "resnet34":
        from torchvision.models import resnet34
        model = resnet34(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif cfg.architecture == "resnet50":
        from torchvision.models import resnet50
        model = resnet50(weights="DEFAULT")
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif cfg.architecture == "convnext_tiny":
        model = convnext_tiny(weights="DEFAULT")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    elif cfg.architecture == "convnext_small":
        model = convnext_small(weights="DEFAULT")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    elif cfg.architecture == "convnext_base":
        model = convnext_base(weights="DEFAULT")
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, 2)
    else:
        raise ValueError(f"Unknown architecture: {cfg.architecture}")
    
    model = model.to(cfg.device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=cfg.epochs)

    crit = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    print("Device:", cfg.device)
    if cfg.device == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
    
    print(f"\n{'='*60}")
    print(f"Starting training with {cfg.architecture}")
    print(f"Epochs: {cfg.epochs} | Batch size: {cfg.batch_size} | LR: {cfg.lr}")
    print(f"{'='*60}\n")

    best_val_acc = -1.0
    best_path = cfg.out_ckpt
    bad_epochs = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        pbar = tqdm(train_loader, 
                    total=len(train_loader), 
                    desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]",
                    dynamic_ncols=True,
                    leave=False,
                    file=sys.stdout)
        
        for xb, yb in pbar:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            optim.step()

            running_loss += float(loss.item()) * xb.size(0)
            pred = logits.argmax(dim=1)
            correct_train += int((pred == yb).sum().item())
            total_train += int(yb.numel())
            
            curr_loss = running_loss / total_train
            curr_acc = correct_train / total_train
            pbar.set_postfix(loss=f"{curr_loss:.4f}", acc=f"{curr_acc:.3f}")
        
        pbar.close()
        
        sched.step()
        train_loss = running_loss / len(train_ds)
        train_acc = correct_train / max(total_train, 1)

        model.eval()
        correct_val = 0
        total_val = 0
        val_loss_sum = 0.0
        
        pbar_val = tqdm(val_loader,
                        total=len(val_loader),
                        desc=f"Epoch {epoch+1}/{cfg.epochs} [Val]",
                        dynamic_ncols=True,
                        leave=False,
                        file=sys.stdout)
        
        with torch.no_grad():
            for xb, yb in pbar_val:
                xb = xb.to(cfg.device, non_blocking=True)
                yb = yb.to(cfg.device, non_blocking=True)
                logits = model(xb)
                loss = crit(logits, yb)

                val_loss_sum += float(loss.item()) * xb.size(0)
                pred = logits.argmax(dim=1)
                correct_val += int((pred == yb).sum().item())
                total_val += int(yb.numel())
                
                curr_vloss = val_loss_sum / total_val
                curr_vacc = correct_val / total_val
                pbar_val.set_postfix(loss=f"{curr_vloss:.4f}", acc=f"{curr_vacc:.3f}")
        
        pbar_val.close()

        val_loss = val_loss_sum / len(val_ds)
        val_acc = correct_val / max(total_val, 1)
        lr_now = optim.param_groups[0]["lr"]
        epoch_time = time.time() - epoch_start
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Epoch {epoch+1:02d}/{cfg.epochs} | "
              f"Time: {epoch_time:.0f}s | "
              f"LR: {lr_now:.2e} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}")
        sys.stdout.flush()

        if val_acc > best_val_acc + 1e-6:
            best_val_acc = val_acc
            bad_epochs = 0
            torch.save({"model": model.state_dict(), "architecture": cfg.architecture}, best_path)
            print(f"  ✓ Saved best model: {best_path} (val_acc={best_val_acc:.4f})")
            sys.stdout.flush()
        else:
            bad_epochs += 1
            print(f"  ⚠ No improvement ({bad_epochs}/{cfg.patience})")
            sys.stdout.flush()
            if bad_epochs >= cfg.patience:
                print(f"\n*** EARLY STOPPING: No improvement for {cfg.patience} epochs ***")
                sys.stdout.flush()
                break

    print(f"\n{'='*60}")
    print(f"Training completed! Best val_acc: {best_val_acc:.4f}")
    print(f"{'='*60}\n")
    
    import json
    history_path = os.path.join(os.path.dirname(best_path), "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to: {history_path}")

    print("\nEvaluating on test set...")
    state = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(state["model"])
    model.eval()

    correct = 0
    total = 0
    
    pbar_test = tqdm(test_loader,
                     total=len(test_loader),
                     desc="Test",
                     dynamic_ncols=True,
                     file=sys.stdout)
    
    with torch.no_grad():
        for xb, yb in pbar_test:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)
            logits = model(xb)
            pred = logits.argmax(dim=1)
            correct += int((pred == yb).sum().item())
            total += int(yb.numel())
            
            curr_acc = correct / total
            pbar_test.set_postfix(acc=f"{curr_acc:.3f}")
    
    pbar_test.close()
    
    test_acc = correct / max(total, 1)
    print(f"\n{'='*60}")
    print(f"TEST ACCURACY (best checkpoint): {test_acc:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python src/train_patch_classifier_v2.py <tiles_manifest.csv>")
        sys.exit(1)

    cfg = TrainConfig(manifest_csv=sys.argv[1])
    main(cfg)
