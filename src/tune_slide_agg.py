import argparse
import csv
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

def load_model(ckpt_path, device):
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model

def label_to_int(s: str) -> int:
    s = (s or "").strip().lower()
    if s == "benign": return 0
    if s == "malignant": return 1
    raise ValueError(f"bad label: {s}")

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    acc = (tp + tn) / max((tp+tn+fp+fn), 1)
    recall = tp / max((tp+fn), 1)
    spec = tn / max((tn+fp), 1)
    bal = 0.5 * (recall + spec)
    prec = tp / max((tp+fp), 1)
    f1 = 2*prec*recall / max((prec+recall), 1e-9)
    return {
        "acc": acc, "bal": bal, "prec": prec, "recall": recall, "spec": spec, "f1": f1,
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }

@torch.no_grad()
def slide_predict(
    slide_tiles, slide_label,
    model, device, tfm,
    k, percentile, score_thr, tile_thr, ratio_thr,
    mode="ratio", high_thr=0.95, high_n=10
):
    """
    mode:
      - ratio: malignant if (score>=score_thr) and (ratio>=ratio_thr)
      - highcount: malignant if (score>=score_thr) and (count(probs>high_thr) >= high_n)
    """
    sm = nn.Softmax(dim=1)
    y_true, y_pred = [], []

    for sid, tiles in slide_tiles.items():
        probs = []
        for p in tiles:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            pr = sm(model(x))[0,1].item()
            probs.append(pr)

        probs = np.array(probs, dtype=np.float32)

        topk = np.sort(probs)[-k:] if k <= len(probs) else np.sort(probs)
        score = np.percentile(topk, percentile)

        if mode == "ratio":
            ratio = float((probs > tile_thr).mean())
            pred = 1 if (score >= score_thr and ratio >= ratio_thr) else 0
        else:
            cnt = int((probs > high_thr).sum())
            pred = 1 if (score >= score_thr and cnt >= high_n) else 0

        y_true.append(label_to_int(slide_label[sid]))
        y_pred.append(pred)

    return compute_metrics(y_true, y_pred)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_csv")
    ap.add_argument("ckpt_path")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--normalize", action="store_true", help="use ImageNet normalization (should match training)")
    ap.add_argument("--mode", choices=["ratio","highcount"], default="ratio")

    ap.add_argument("--k_list", default="5,10,20,30")
    ap.add_argument("--perc_list", default="50,80")
    ap.add_argument("--score_thr_list", default="0.6,0.7,0.8")
    ap.add_argument("--tile_thr_list", default="0.85,0.9,0.95")
    ap.add_argument("--ratio_thr_list", default="0.10,0.15,0.20,0.25,0.30")

    ap.add_argument("--high_thr_list", default="0.90,0.95,0.97")
    ap.add_argument("--high_n_list", default="5,10,15,20")
    ap.add_argument("--split", choices=["train","val","test"], default="val")

    ap.add_argument("--metric", choices=["bal","f1","acc","recall","spec"], default="bal")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.manifest_csv)))
    rows = [r for r in rows if r["split"].strip().lower() == args.split]

    slide_tiles = defaultdict(list)
    slide_label = {}
    for r in rows:
        sid = r["slide_id"]
        slide_tiles[sid].append(r["tile_path"])
        slide_label[sid] = r["label"]

    print(f"Slides in {args.split}:", len(slide_tiles))

    t = [transforms.Resize((args.img_size, args.img_size)), transforms.ToTensor()]
    if args.normalize:
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD  = (0.229, 0.224, 0.225)
        t.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    tfm = transforms.Compose(t)

    model = load_model(args.ckpt_path, args.device)

    k_list = [int(x) for x in args.k_list.split(",")]
    perc_list = [float(x) for x in args.perc_list.split(",")]
    score_thr_list = [float(x) for x in args.score_thr_list.split(",")]
    tile_thr_list = [float(x) for x in args.tile_thr_list.split(",")]
    ratio_thr_list = [float(x) for x in args.ratio_thr_list.split(",")]
    high_thr_list = [float(x) for x in args.high_thr_list.split(",")]
    high_n_list = [int(x) for x in args.high_n_list.split(",")]

    best = None
    best_cfg = None

    total = 0
    if args.mode == "ratio":
        for k in k_list:
            for perc in perc_list:
                for sthr in score_thr_list:
                    for tthr in tile_thr_list:
                        for rthr in ratio_thr_list:
                            total += 1
        pbar = tqdm(total=total, desc="grid")
        for k in k_list:
            for perc in perc_list:
                for sthr in score_thr_list:
                    for tthr in tile_thr_list:
                        for rthr in ratio_thr_list:
                            m = slide_predict(
                                slide_tiles, slide_label,
                                model, args.device, tfm,
                                k, perc, sthr, tthr, rthr,
                                mode="ratio"
                            )
                            score = m[args.metric]
                            if best is None or score > best:
                                best = score
                                best_cfg = ("ratio", k, perc, sthr, tthr, rthr, m)
                            pbar.update(1)
        pbar.close()
    else:
        for k in k_list:
            for perc in perc_list:
                for sthr in score_thr_list:
                    for hthr in high_thr_list:
                        for hn in high_n_list:
                            total += 1
        pbar = tqdm(total=total, desc="grid")
        for k in k_list:
            for perc in perc_list:
                for sthr in score_thr_list:
                    for hthr in high_thr_list:
                        for hn in high_n_list:
                            m = slide_predict(
                                slide_tiles, slide_label,
                                model, args.device, tfm,
                                k, perc, sthr, tile_thr=0.0, ratio_thr=0.0,
                                mode="highcount", high_thr=hthr, high_n=hn
                            )
                            score = m[args.metric]
                            if best is None or score > best:
                                best = score
                                best_cfg = ("highcount", k, perc, sthr, hthr, hn, m)
                            pbar.update(1)
        pbar.close()

    print("\n=== BEST CONFIG ===")
    if best_cfg[0] == "ratio":
        _, k, perc, sthr, tthr, rthr, m = best_cfg
        print(f"mode=ratio  k={k} perc={perc} score_thr={sthr} tile_thr={tthr} ratio_thr={rthr}")
    else:
        _, k, perc, sthr, hthr, hn, m = best_cfg
        print(f"mode=highcount  k={k} perc={perc} score_thr={sthr} high_thr={hthr} high_n={hn}")

    print(f"metric={args.metric} best={best:.4f}")
    print("acc={acc:.3f} bal={bal:.3f} f1={f1:.3f} prec={prec:.3f} recall={recall:.3f} spec={spec:.3f}".format(**m))
    print(f"confusion: tn={m['tn']} fp={m['fp']} fn={m['fn']} tp={m['tp']}")

if __name__ == "__main__":
    main()
