import sys, json, csv, os

def load_hits(path):
    with open(path, "r") as f:
        return json.load(f)["data"]["hits"]

def extract_rows(hits, label):
    rows=[]
    for h in hits:
        file_id = h["file_id"]
        patient_id = h["cases"][0]["submitter_id"]
        rows.append({
            "slide_id": file_id,
            "patient_id": patient_id,
            "split": "train",
            "local_path": f"data/raw_wsi/{file_id}.svs",
            "slide_label": label
        })
    return rows

def main(out_csv, tumor_json, normal_json):
    tumor_hits = load_hits(tumor_json)
    normal_hits = load_hits(normal_json)

    rows = []
    rows += extract_rows(tumor_hits, "malignant")
    rows += extract_rows(normal_hits, "benign")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["slide_id","patient_id","split","local_path","slide_label"])
        w.writeheader()
        w.writerows(rows)

    print(f"Wrote {len(rows)} rows -> {out_csv} (malignant={len(tumor_hits)} benign={len(normal_hits)})")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python src/build_slide_list_from_gdc.py out.csv tumor.json normal.json")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
