import sys
import csv
import random
from collections import defaultdict

def main(in_csv: str, out_csv: str, seed: int = 42, train: float = 0.7, val: float = 0.15, test: float = 0.15):
    assert abs((train + val + test) - 1.0) < 1e-9

    with open(in_csv, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    by_patient = defaultdict(list)
    for r in rows:
        pid = r["patient_id"]
        if not pid:
            raise ValueError("patient_id is empty for some rows. Fix your slide list first.")
        by_patient[pid].append(r)

    patients = sorted(by_patient.keys())
    rnd = random.Random(seed)
    rnd.shuffle(patients)

    n = len(patients)
    n_train = int(n * train)
    n_val = int(n * val)
    train_p = set(patients[:n_train])
    val_p = set(patients[n_train:n_train+n_val])
    test_p = set(patients[n_train+n_val:])

    out_rows = []
    for pid, items in by_patient.items():
        if pid in train_p:
            sp = "train"
        elif pid in val_p:
            sp = "val"
        else:
            sp = "test"
        for r in items:
            r2 = dict(r)
            r2["split"] = sp
            out_rows.append(r2)

    out_rows.sort(key=lambda x: (x["split"], x["patient_id"], x["slide_id"]))

    fieldnames = list(out_rows[0].keys()) if out_rows else ["slide_id","patient_id","split","local_path"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Patients: total={n} train={len(train_p)} val={len(val_p)} test={len(test_p)}")
    print(f"Wrote: {out_csv}")

if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        print("Usage: python src/make_patient_split.py input.csv output.csv [seed]")
        sys.exit(1)
    seed = int(sys.argv[3]) if len(sys.argv) == 4 else 42
    main(sys.argv[1], sys.argv[2], seed=seed)
