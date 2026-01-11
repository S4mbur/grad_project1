import os
import sys
import csv
import subprocess
from typing import Dict, List


def download_gdc(uuid: str, out_path: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    url = f"https://api.gdc.cancer.gov/data/{uuid}"
    print(f"[download] {uuid} -> {out_path}")
    subprocess.run(["curl", "-L", "-o", out_path, url], check=True)


def append_rows(out_csv: str, rows: List[Dict[str, str]]):
    if not rows:
        return
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    exists = os.path.exists(out_csv)

    fieldnames = list(rows[0].keys())
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerows(rows)


def process_one(slide_list_csv_one: str, tmp_manifest: str, target_kept: int):
    subprocess.run([
        sys.executable, "src/process_one_slide_v2.py",
        slide_list_csv_one, tmp_manifest, str(target_kept)
    ], check=True)


def read_manifest_rows(manifest_csv: str) -> List[Dict[str, str]]:
    with open(manifest_csv, "r", newline="") as f:
        return list(csv.DictReader(f))


def main(slide_list_split_csv: str, global_manifest_out: str, target_kept: int = 1000, keep_wsi: bool = False):
    with open(slide_list_split_csv, "r", newline="") as f:
        slides = list(csv.DictReader(f))

    os.makedirs("data/raw_wsi", exist_ok=True)
    os.makedirs("data/tmp", exist_ok=True)

    for s in slides:
        uuid = s["slide_id"]
        patient_id = s["patient_id"]
        split = s["split"]

        wsi_path = os.path.join("data/raw_wsi", f"{uuid}.svs")
        one_csv = os.path.join("data/tmp", f"one_{uuid}.csv")
        tmp_manifest = os.path.join("data/tmp", f"tmp_manifest_{uuid}.csv")

        with open(one_csv, "w", newline="") as f1:
            w = csv.DictWriter(f1, fieldnames=["slide_id","patient_id","split","local_path","label"])
            w.writeheader()
            w.writerow({
                "slide_id": uuid,
                "patient_id": patient_id,
                "split": split,
                "local_path": wsi_path,
                "label": s.get("label", "malignant")
            })


        if not os.path.exists(wsi_path):
            download_gdc(uuid, wsi_path)

        if os.path.exists(tmp_manifest):
            os.remove(tmp_manifest)
        process_one(one_csv, tmp_manifest, target_kept=target_kept)

        rows = read_manifest_rows(tmp_manifest)
        append_rows(global_manifest_out, rows)

        if not keep_wsi:
            try:
                os.remove(wsi_path)
            except FileNotFoundError:
                pass

        for p in (one_csv, tmp_manifest):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

        print(f"[done] {uuid} ({split}) appended {len(rows)} tiles -> {global_manifest_out}")

    print("All slides processed.")


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4, 5):
        print("Usage: python src/run_streaming_pipeline.py slide_list_split.csv global_tiles_manifest.csv [target_kept] [keep_wsi0or1]")
        sys.exit(1)
    target = int(sys.argv[3]) if len(sys.argv) >= 4 else 1000
    keep = bool(int(sys.argv[4])) if len(sys.argv) == 5 else False
    main(sys.argv[1], sys.argv[2], target_kept=target, keep_wsi=keep)
