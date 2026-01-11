import sys
import csv
from typing import List, Dict


def load_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: str, rows: List[Dict[str, str]], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main(base_manifest: str, hard_csv: str, out_manifest: str, repeat: int = 3):
    base = load_csv(base_manifest)
    hard = load_csv(hard_csv)

    base_fields = list(base[0].keys())
    hard_rows = []
    for r in hard:
        rr = {k: r[k] for k in base_fields}
        rr["split"] = "train"
        hard_rows.append(rr)

    out = list(base)
    for _ in range(repeat):
        out.extend(hard_rows)

    write_csv(out_manifest, out, base_fields)
    print("Base rows:", len(base))
    print("Hard rows:", len(hard_rows), "repeat:", repeat, "=> added:", len(hard_rows) * repeat)
    print("Wrote:", out_manifest)


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5):
        print("Usage: python src/make_manifest_with_hard_oversample.py <base.csv> <hard.csv> <out.csv> [repeat]")
        sys.exit(1)
    repeat = int(sys.argv[4]) if len(sys.argv) == 5 else 3
    main(sys.argv[1], sys.argv[2], sys.argv[3], repeat)
