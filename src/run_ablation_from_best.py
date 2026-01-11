import argparse
import pandas as pd
import numpy as np
import subprocess
import shlex


def run_once(manifest, ckpt, k, perc, sthr, tthr, rthr):
    """Run slide inference with given parameters and return metrics."""
    out_csv = f"data/manifests/ablation_tmp_k{k}_p{perc}_s{sthr}_t{tthr}_r{rthr}.csv"
    cmd = (
        f"python src/slide_inference.py {shlex.quote(manifest)} {shlex.quote(ckpt)} "
        f"--k {k} --percentile {perc} --score_thr {sthr} --tile_thr {tthr} --ratio_thr {rthr} "
        f"--save_csv {shlex.quote(out_csv)}"
    )
    subprocess.check_call(cmd, shell=True)
    df = pd.read_csv(out_csv)
    yt = (df["true_label"] == "malignant").astype(int).values
    yp = (df["pred_label"] == "malignant").astype(int).values
    tp = int(((yt == 1) & (yp == 1)).sum())
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    acc = (tp + tn) / max((tp + tn + fp + fn), 1)
    tpr = tp / max((tp + fn), 1)
    tnr = tn / max((tn + fp), 1)
    bal = 0.5 * (tpr + tnr)
    prec = tp / max((tp + fp), 1)
    f1 = (2 * prec * tpr) / max((prec + tpr), 1e-12)
    return dict(acc=acc, bal_acc=bal, recall_mal=tpr, specificity=tnr, precision_mal=prec, f1_mal=f1, tp=tp, tn=tn, fp=fp, fn=fn)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest_csv")
    ap.add_argument("ckpt_path")
    ap.add_argument("--best_k", type=int, required=True)
    ap.add_argument("--best_percentile", type=float, required=True)
    ap.add_argument("--best_score_thr", type=float, required=True)
    ap.add_argument("--best_tile_thr", type=float, required=True)
    ap.add_argument("--best_ratio_thr", type=float, required=True)
    ap.add_argument("--out_csv", default="data/manifests/ablation_results.csv")
    args = ap.parse_args()

    os = __import__("os")
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    base = dict(k=args.best_k, percentile=args.best_percentile,
                score_thr=args.best_score_thr, tile_thr=args.best_tile_thr, ratio_thr=args.best_ratio_thr)

    ablations = []

    k_vals = sorted(set([max(1, base["k"] - 5), base["k"], base["k"] + 10]))
    p_vals = sorted(set([max(1, base["percentile"] - 20), base["percentile"], min(99, base["percentile"] + 20)]))
    s_vals = sorted(set([max(0.1, base["score_thr"] - 0.1), base["score_thr"], min(0.99, base["score_thr"] + 0.1)]))
    t_vals = sorted(set([max(0.5, base["tile_thr"] - 0.05), base["tile_thr"], min(0.99, base["tile_thr"] + 0.05)]))
    r_vals = sorted(set([max(0.0, base["ratio_thr"] - 0.05), base["ratio_thr"], min(1.0, base["ratio_thr"] + 0.05)]))

    def add_runs(param, values):
        for v in values:
            cfg = dict(base)
            cfg[param] = v
            m = run_once(args.manifest_csv, args.ckpt_path,
                         int(cfg["k"]), float(cfg["percentile"]),
                         float(cfg["score_thr"]), float(cfg["tile_thr"]), float(cfg["ratio_thr"]))
            ablations.append({"varied": param, "value": v, **cfg, **m})

    add_runs("k", k_vals)
    add_runs("percentile", p_vals)
    add_runs("score_thr", s_vals)
    add_runs("tile_thr", t_vals)
    add_runs("ratio_thr", r_vals)

    df = pd.DataFrame(ablations)
    df.to_csv(args.out_csv, index=False)
    print("Wrote:", args.out_csv)
    print(df.sort_values(["bal_acc", "acc"], ascending=False).head(20).to_string(index=False))


if __name__ == "__main__":
    main()
