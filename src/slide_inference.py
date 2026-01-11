import argparse
import csv
from collections import defaultdict
from pathlib import Path
from transforms_common import build_tfm
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def load_model(ckpt_path: str, device: str):
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])

    model.to(device)
    model.eval()
    return model


def compute_metrics(y_true, y_pred):
    labels = ["benign", "malignant"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label="malignant", zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label="malignant", zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label="malignant", zero_division=0)

    tn, fp = cm[0, 0], cm[0, 1]
    spec = (tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    bal_acc = 0.5 * (rec + spec)

    return {
        "labels": labels,
        "cm": cm,
        "acc": acc,
        "precision": prec,
        "recall": rec,
        "specificity": spec,
        "f1": f1,
        "balanced_acc": bal_acc,
    }


def main(
    manifest_csv: str,
    ckpt_path: str,
    k: int = 10,
    percentile: float = 50.0,
    score_thr: float = 0.7,
    tile_thr: float = 0.9,
    ratio_thr: float = 0.15,
    out_csv: str | None = None,
    max_tiles_per_slide: int | None = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    rows = list(csv.DictReader(open(manifest_csv)))
    rows = [r for r in rows if r["split"] == "test"]

    slide_tiles = defaultdict(list)
    slide_labels = {}

    for r in rows:
        slide_tiles[r["slide_id"]].append(r["tile_path"])
        slide_labels[r["slide_id"]] = r["label"].strip().lower()

    print("Slides in test:", len(slide_tiles))

    tfm = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    model = load_model(ckpt_path, device)
    sm = nn.Softmax(dim=1)

    y_true, y_pred = [], []
    per_slide_rows = []

    for sid, tiles in tqdm(slide_tiles.items(), desc="slide inference"):
        if max_tiles_per_slide is not None and len(tiles) > max_tiles_per_slide:
            tiles = tiles[:max_tiles_per_slide]

        probs = []
        for p in tiles:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            with torch.no_grad():
                prob_malignant = sm(model(x))[0, 1].item()
            probs.append(prob_malignant)

        probs = np.array(probs, dtype=np.float32)

        kk = min(k, len(probs))
        topk = np.sort(probs)[-kk:]
        score = float(np.percentile(topk, percentile))
        malignant_ratio = float((probs >= tile_thr).mean())

        pred = "malignant" if (score >= score_thr and malignant_ratio >= ratio_thr) else "benign"
        true = slide_labels[sid]

        y_true.append(true)
        y_pred.append(pred)

        top5 = np.sort(probs)[-5:][::-1]
        per_slide_rows.append({
            "slide_id": sid,
            "true_label": true,
            "pred_label": pred,
            "n_tiles_used": len(probs),
            "k": kk,
            "percentile": percentile,
            "score": f"{score:.6f}",
            "score_thr": score_thr,
            "tile_thr": tile_thr,
            "malignant_ratio": f"{malignant_ratio:.6f}",
            "ratio_thr": ratio_thr,
            "top5_probs": ";".join([f"{x:.6f}" for x in top5]),
        })

    m = compute_metrics(y_true, y_pred)
    cm = m["cm"]

    print("\n=== SLIDE LEVEL RESULTS (ROBUST) ===")
    print(f"K={k} perc={percentile} score_thr={score_thr} tile_thr={tile_thr} ratio_thr={ratio_thr}")
    print(f"Accuracy:        {m['acc']:.4f}")
    print(f"Balanced Acc:    {m['balanced_acc']:.4f}")
    print(f"Precision (mal): {m['precision']:.4f}")
    print(f"Recall (mal):    {m['recall']:.4f}")
    print(f"Specificity:     {m['specificity']:.4f}")
    print(f"F1 (mal):        {m['f1']:.4f}")

    print("\nConfusion Matrix (rows=true, cols=pred)")
    print("              pred_benign  pred_malignant")
    print(f"true_benign        {cm[0,0]:4d}          {cm[0,1]:4d}")
    print(f"true_malignant     {cm[1,0]:4d}          {cm[1,1]:4d}")

    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(per_slide_rows[0].keys()))
            w.writeheader()
            w.writerows(per_slide_rows)
        print("\nWrote per-slide results to:", out_csv)

        fps = [r for r in per_slide_rows if r["true_label"] == "benign" and r["pred_label"] == "malignant"]
        fns = [r for r in per_slide_rows if r["true_label"] == "malignant" and r["pred_label"] == "benign"]
        print(f"FP slides: {len(fps)}   FN slides: {len(fns)}")
        if fps:
            print("Example FP slide:", fps[0]["slide_id"], "score=", fps[0]["score"], "ratio=", fps[0]["malignant_ratio"])
        if fns:
            print("Example FN slide:", fns[0]["slide_id"], "score=", fns[0]["score"], "ratio=", fns[0]["malignant_ratio"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_csv")
    ap.add_argument("ckpt_path")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--percentile", type=float, default=50.0)
    ap.add_argument("--score_thr", type=float, default=0.7)
    ap.add_argument("--tile_thr", type=float, default=0.9)
    ap.add_argument("--ratio_thr", type=float, default=0.15)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--max_tiles_per_slide", type=int, default=None)
    ap.add_argument("--save_csv", type=str, default=None)

    args = ap.parse_args()

    main(
        args.manifest_csv,
        args.ckpt_path,
        k=args.k,
        percentile=args.percentile,
        score_thr=args.score_thr,
        tile_thr=args.tile_thr,
        ratio_thr=args.ratio_thr,
        out_csv=(args.save_csv or args.out_csv),
        max_tiles_per_slide=args.max_tiles_per_slide,
    )

