import os
import sys
import csv
import subprocess

def s3_download(source: str, slide_id: str, out_path: str):
    if source in ("cobra", "cobra_bcc", "cobra_nonmalignant"):
        s3_key = f"s3://cobra-pathology/packages/bcc/images/{slide_id}.tif"
    else:
        raise ValueError(f"Unknown source={source}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    print(f"[download] {s3_key} -> {out_path}")
    subprocess.run(["aws", "s3", "cp", "--no-sign-request", s3_key, out_path], check=True)


def append_rows(out_csv, rows):
    if not rows:
        return
    exists = os.path.exists(out_csv)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerows(rows)

def read_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))

def main(slide_list_csv, global_manifest_out, target_kept=300, keep_wsi=0):
    with open(slide_list_csv, "r", newline="") as f:
        slides = list(csv.DictReader(f))

    os.makedirs("data/raw_wsi", exist_ok=True)
    os.makedirs("data/tmp", exist_ok=True)

    for s in slides:
        slide_id = s["slide_id"]
        patient_id = s["patient_id"]
        split = s["split"]
        label = s["label"]
        source = s.get("source", "cobra_bcc")
        local_path = s["local_path"]

        if not os.path.exists(local_path):
            s3_download(source, slide_id, local_path)

        one_csv = os.path.join("data/tmp", f"one_{slide_id}.csv")
        tmp_manifest = os.path.join("data/tmp", f"tmp_manifest_{slide_id}.csv")
        with open(one_csv, "w", newline="") as f1:
            w = csv.DictWriter(f1, fieldnames=["slide_id","patient_id","split","local_path","label"])
            w.writeheader()
            w.writerow({
                "slide_id": slide_id,
                "patient_id": patient_id,
                "split": split,
                "local_path": local_path,
                "label": label
            })

        if os.path.exists(tmp_manifest):
            os.remove(tmp_manifest)
        subprocess.run([sys.executable, "src/process_one_slide_v2.py", one_csv, tmp_manifest, str(target_kept)], check=True)

        rows = read_rows(tmp_manifest)
        append_rows(global_manifest_out, rows)

        if int(keep_wsi) == 0:
            try:
                os.remove(local_path)
            except FileNotFoundError:
                pass
        for p in (one_csv, tmp_manifest):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass

        print(f"[done] {slide_id} label={label} appended {len(rows)} tiles")

    print("All COBRA slides processed.")

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4, 5):
        print("Usage: python src/run_cobra_streaming_pipeline.py slide_list.csv global_manifest.csv [target_kept] [keep_wsi0or1]")
        sys.exit(1)
    target = int(sys.argv[3]) if len(sys.argv) >= 4 else 300
    keep = int(sys.argv[4]) if len(sys.argv) == 5 else 0
    main(sys.argv[1], sys.argv[2], target_kept=target, keep_wsi=keep)
