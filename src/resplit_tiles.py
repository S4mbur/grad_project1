#!/usr/bin/env python3
"""
Re-split tile manifest with better train/val/test ratios.
Target: 70% train, 15% val, 15% test (patient-level split)
"""

import csv
import random
from pathlib import Path
from collections import defaultdict

def main():
    manifest_in = "data/manifests/tile_manifest.csv"
    manifest_out = "data/manifests/tile_manifest_70_15_15.csv"
    
    print("Reading manifest...")
    with open(manifest_in, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"Total tiles: {len(rows)}")
    
    patient_tiles = defaultdict(list)
    for row in rows:
        patient_id = row['slide_id']
        patient_tiles[patient_id].append(row)
    
    print(f"Total patients/slides: {len(patient_tiles)}")
    
    patients = list(patient_tiles.keys())
    random.seed(42)
    random.shuffle(patients)
    
    n_patients = len(patients)
    n_train = int(n_patients * 0.70)
    n_val = int(n_patients * 0.15)
    
    train_patients = set(patients[:n_train])
    val_patients = set(patients[n_train:n_train + n_val])
    test_patients = set(patients[n_train + n_val:])
    
    print(f"Patient split: {len(train_patients)} train, {len(val_patients)} val, {len(test_patients)} test")
    
    train_count = 0
    val_count = 0
    test_count = 0
    
    for row in rows:
        patient_id = row['slide_id']
        if patient_id in train_patients:
            row['split'] = 'train'
            train_count += 1
        elif patient_id in val_patients:
            row['split'] = 'val'
            val_count += 1
        else:
            row['split'] = 'test'
            test_count += 1
    
    total = train_count + val_count + test_count
    print(f"\nTile split:")
    print(f"  Train: {train_count:6d} ({100*train_count/total:.1f}%)")
    print(f"  Val:   {val_count:6d} ({100*val_count/total:.1f}%)")
    print(f"  Test:  {test_count:6d} ({100*test_count/total:.1f}%)")
    print(f"  Total: {total:6d}")
    
    print(f"\nWriting to {manifest_out}...")
    with open(manifest_out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print("✓ Done!")

if __name__ == '__main__':
    main()
