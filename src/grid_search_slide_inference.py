import argparse, csv
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

IMAGENET_MEAN=(0.485,0.456,0.406)
IMAGENET_STD=(0.229,0.224,0.225)

def load_model(ckpt_path, device):
    from torchvision.models import resnet18
    m = resnet18(weights=None)
    m.fc = nn.Linear(m.fc.in_features, 2)
    state = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(state["model"])
    m.to(device).eval()
    return m

def label_to_int(s):
    s=(s or "").strip().lower()
    if s=="benign": return 0
    if s=="malignant": return 1
    raise ValueError(s)

def metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()
    acc = (tp+tn)/max(tp+tn+fp+fn,1)
    rec = tp/max(tp+fn,1)
    spec = tn/max(tn+fp,1)
    bal = 0.5*(rec+spec)
    prec = tp/max(tp+fp,1)
    f1 = (2*prec*rec)/max(prec+rec,1e-9)
    return dict(acc=acc, bal=bal, prec=prec, rec=rec, spec=spec, f1=f1, tn=tn, fp=fp, fn=fn, tp=tp)

@torch.no_grad()
def eval_cfg(slide_tiles, slide_label, model, device, tfm, k, perc, score_thr, tile_thr, ratio_thr):
    sm = nn.Softmax(dim=1)
    yt, yp = [], []
    for sid, tiles in slide_tiles.items():
        probs=[]
        for p in tiles:
            img = Image.open(p).convert("RGB")
            x = tfm(img).unsqueeze(0).to(device)
            probs.append(float(sm(model(x))[0,1].item()))
        probs=np.array(probs, dtype=np.float32)

        kk=min(k, len(probs))
        topk=np.sort(probs)[-kk:]
        score=float(np.percentile(topk, perc))
        ratio=float((probs >= tile_thr).mean())
        pred = 1 if (score >= score_thr and ratio >= ratio_thr) else 0

        yt.append(label_to_int(slide_label[sid]))
        yp.append(pred)

    return metrics(yt, yp)

def parse_list(s, tp=float):
    return [tp(x) for x in s.split(",") if x.strip()!=""]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("manifest_csv")
    ap.add_argument("ckpt_path")
    ap.add_argument("--split", default="val", choices=["train","val","test"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--k_list", default="5,10,20,30")
    ap.add_argument("--perc_list", default="50,80")
    ap.add_argument("--score_thr_list", default="0.6,0.7,0.8")
    ap.add_argument("--tile_thr_list", default="0.85,0.9,0.95")
    ap.add_argument("--ratio_thr_list", default="0.10,0.15,0.20,0.25,0.30")

    ap.add_argument("--out_csv", default="data/manifests/grid_results.csv")
    ap.add_argument("--metric", default="bal", choices=["bal","f1","acc","rec","spec"])
    ap.add_argument("--min_spec", type=float, default=0.30)
    args=ap.parse_args()

    rows=list(csv.DictReader(open(args.manifest_csv)))
    rows=[r for r in rows if r["split"].strip().lower()==args.split]

    slide_tiles=defaultdict(list)
    slide_label={}
    for r in rows:
        sid=r["slide_id"]
        slide_tiles[sid].append(r["tile_path"])
        slide_label[sid]=r["label"]

    print(f"Slides in {args.split}:", len(slide_tiles))

    t=[transforms.Resize((args.img_size,args.img_size)), transforms.ToTensor()]
    if args.normalize:
        t.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    tfm=transforms.Compose(t)

    model=load_model(args.ckpt_path, args.device)

    k_list=parse_list(args.k_list, int)
    perc_list=parse_list(args.perc_list, float)
    score_thr_list=parse_list(args.score_thr_list, float)
    tile_thr_list=parse_list(args.tile_thr_list, float)
    ratio_thr_list=parse_list(args.ratio_thr_list, float)

    out_rows=[]
    total=len(k_list)*len(perc_list)*len(score_thr_list)*len(tile_thr_list)*len(ratio_thr_list)
    pbar=tqdm(total=total, desc="grid")
    for k in k_list:
        for perc in perc_list:
            for sthr in score_thr_list:
                for tthr in tile_thr_list:
                    for rthr in ratio_thr_list:
                        m=eval_cfg(slide_tiles, slide_label, model, args.device, tfm, k, perc, sthr, tthr, rthr)
                        out_rows.append(dict(
                            split=args.split, k=k, perc=perc, score_thr=sthr, tile_thr=tthr, ratio_thr=rthr,
                            **m
                        ))
                        pbar.update(1)
    pbar.close()

    filt=[r for r in out_rows if float(r["spec"]) >= args.min_spec]
    pool=filt if len(filt)>0 else out_rows

    pool=sorted(pool, key=lambda r: (float(r[args.metric]), float(r["spec"])), reverse=True)

    import os
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=list(pool[0].keys()))
        w.writeheader()
        w.writerows(pool)

    best=pool[0]
    print("\n=== TOP-1 (after min_spec filter) ===")
    print(best)
    print("Wrote:", args.out_csv, "rows=", len(pool))

if __name__=="__main__":
    main()
