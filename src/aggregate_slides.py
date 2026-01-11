import os
import sys
import csv
from collections import defaultdict
from typing import List, Dict


def topk_mean(scores: List[float], k: int) -> float:
    if len(scores) == 0:
        return float("nan")
    k = max(1, min(k, len(scores)))
    scores_sorted = sorted(scores, reverse=True)
    return sum(scores_sorted[:k]) / k


def main(tile_scores_csv: str, out_csv: str, k: int = 50):
    by_slide = defaultdict(list)
    meta = {}

    with open(tile_scores_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            slide_id = r["slide_id"]
            by_slide[slide_id].append(float(r["p_malignant"]))
            meta[slide_id] = {
                "patient_id": r.get("patient_id", ""),
                "split": r.get("split", ""),
                "label": r.get("label", ""),
            }

    rows_out: List[Dict[str, str]] = []
    for slide_id, scores in by_slide.items():
        score = topk_mean(scores, k=k)
        label = meta[slide_id]["label"]
        rows_out.append({
            "slide_id": slide_id,
            "patient_id": meta[slide_id]["patient_id"],
            "label": label,
            "label_int": "1" if label == "malignant" else "0",
            "split": meta[slide_id]["split"],
            "num_tiles": str(len(scores)),
            "topk": str(k),
            "slide_score_topk_mean": f"{score:.6f}",
        })

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)

    print("Wrote slide predictions:", out_csv)


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python src/aggregate_slides.py tiles_with_scores.csv slide_predictions.csv [topk]")
        sys.exit(1)
    k = int(sys.argv[3]) if len(sys.argv) == 4 else 50
    main(sys.argv[1], sys.argv[2], k=k)
