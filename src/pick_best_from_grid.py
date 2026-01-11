import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("grid_csv")
    ap.add_argument("--metric", default="bal_acc", choices=["bal_acc", "acc", "f1_mal", "recall_mal", "specificity"])
    args = ap.parse_args()

    df = pd.read_csv(args.grid_csv)
    if df.empty:
        raise RuntimeError("Grid CSV empty.")

    df = df.sort_values([args.metric, "acc"], ascending=False)

    best = df.iloc[0].to_dict()
    print("=== BEST CONFIG ===")
    for k in ["k", "percentile", "score_thr", "tile_thr", "ratio_thr"]:
        print(f"{k}: {best[k]}")
    print("\n=== METRICS ===")
    for k in ["acc", "bal_acc", "precision_mal", "recall_mal", "specificity", "f1_mal", "tp", "tn", "fp", "fn"]:
        print(f"{k}: {best[k]}")

    cmd = (
        "python src/slide_inference.py "
        "<manifest.csv> <ckpt.pt> "
        f"--k {int(best['k'])} "
        f"--percentile {float(best['percentile'])} "
        f"--score_thr {float(best['score_thr'])} "
        f"--tile_thr {float(best['tile_thr'])} "
        f"--ratio_thr {float(best['ratio_thr'])}"
    )
    print("\nRun command:")
    print(cmd)


if __name__ == "__main__":
    main()
