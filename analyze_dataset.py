import json
import glob
import os
from collections import Counter
import numpy as np

print("=" * 60)
print("  DEEP DATASET + MODEL ANALYSIS")
print("=" * 60)

# ─── 1. Dataset files
dataset_dir = "dataset"
json_files = glob.glob(os.path.join(dataset_dir, "*.json"))
print(f"\n[1] Dataset files found: {len(json_files)}")
for f in json_files:
    print(f"    - {os.path.basename(f)}  ({os.path.getsize(f)//1024} KB)")

# ─── 2. Labels in dataset
all_labels = []
total_samples = 0
for json_path in json_files:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    labels = [item.get("label") for item in data if item.get("label")]
    all_labels.extend(labels)
    total_samples += len(data)
    counts = Counter(labels)
    print(f"\n    File: {os.path.basename(json_path)}")
    print(f"    Total samples: {len(data)}")
    print(f"    Labels found:")
    for lbl, cnt in sorted(counts.items()):
        print(f"      {lbl}: {cnt} samples")

print(f"\n[2] TOTAL across all files: {total_samples} samples")
print(f"    Unique labels in dataset: {sorted(set(all_labels))}")

# ─── 3. What training script CLASS_LABELS expects
SCRIPT_CLASS_LABELS = [
    "aliph", "bay", "pay", "ray", "seen",
    "laam", "meem", "noon", "kaaf", "hey", "none"
]
print(f"\n[3] train_mediapipe_classifier.py CLASS_LABELS (hardcoded):")
print(f"    {SCRIPT_CLASS_LABELS}")

dataset_labels = set(all_labels)
script_labels  = set(SCRIPT_CLASS_LABELS)

missing_from_dataset = script_labels - dataset_labels - {"none"}
extra_in_dataset     = dataset_labels - script_labels

print(f"\n[4] ANALYSIS:")
print(f"    Labels in CLASS_LABELS but NOT in dataset: {sorted(missing_from_dataset)}")
print(f"    Labels in dataset but NOT in CLASS_LABELS: {sorted(extra_in_dataset)}")

print(f"\n[5] ROOT CAUSE:")
if missing_from_dataset:
    print(f"    *** PROBLEM FOUND ***")
    print(f"    These classes are in the model output layer but have NO training data:")
    for lbl in sorted(missing_from_dataset):
        print(f"      - '{lbl}'  <-- model guesses this randomly!")
    print()
    print(f"    The model has {len(SCRIPT_CLASS_LABELS)} output neurons (one per class).")
    print(f"    Classes with no data still get output neurons.")
    print(f"    When input is ambiguous, the model may predict them.")

# ─── 4. Check saved labels.json
labels_path = "frontend/model/urdu10_labels.json"
if os.path.exists(labels_path):
    with open(labels_path) as f:
        saved_labels = json.load(f)
    print(f"\n[6] Saved urdu10_labels.json has {len(saved_labels)} classes:")
    print(f"    {saved_labels}")
else:
    print(f"\n[6] urdu10_labels.json not found!")

print("\n" + "=" * 60)
print("  SOLUTION")
print("=" * 60)
print("  Remove 'seen', 'kaaf' (and other untrained labels)")
print("  from CLASS_LABELS in scripts/train_mediapipe_classifier.py")
print("  so only labels WITH actual data are used for training.")
print("=" * 60)
