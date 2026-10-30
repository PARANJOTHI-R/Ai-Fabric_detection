"""
Production Dataset Preparation Runner (Revised V1)
===================================================
Creates `data_prepared_v1_revised/` with corrected group-level splitting:
1. Fixes the 0-validation issue for 'lines' by grouping by capture session (date + hour window) rather than date alone.
2. Fixes clean negative leakage by grouping AITEX clean samples by fabric structure code.
3. Guarantees every evaluation class has representative validation and test samples.
4. Preserves data_prepared_v1 unchanged.
"""

import os
import sys
import glob
import shutil
import hashlib
import json
import random
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(r"d:\AI-Fabric-Defect-Detection\AI-Fabric-Defect-Detection")
OUT_DIR = REPO_ROOT / "data_prepared_v1_revised"

SEED = 42
random.seed(SEED)


def get_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def extract_group_key_refined(filename: str) -> str:
    """
    Extracts true session / fabric roll ID.
    - Roboflow TILDA: '001_jpg.rf.xxxx' -> '001'
    - AITEX line-scan: '0010_000_02.png' -> 'aitex_02' (grouped by fabric structure!)
    - Continuous camera runs: 'line_2018-10-10 09_37_25...' -> 'line_2018-10-10_09' (grouped by inspection hour run!)
    - Timestamped photos: '20180531_135032.jpg' -> '20180531_13'
    - AITEX experiments: 'exp1_num_...' -> 'exp1'
    """
    stem = Path(filename).stem.lower()

    # Roboflow export prefix
    if "_jpg.rf." in stem or "_png.rf." in stem:
        return "tilda_" + stem.split("_")[0]

    # AITEX full strips / patches: nnnn_ddd_ff -> group by fabric structure ff
    parts = stem.split("_")
    if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit() and parts[2].isdigit():
        return f"aitex_fabric_{parts[2]}"

    # Continuous camera lines: line_YYYY-MM-DD HH_MM_SS
    if stem.startswith("line_"):
        # e.g. line_2018-10-10 09_37_... -> line_2018-10-10_09
        tokens = stem.split(" ")
        if len(tokens) >= 2:
            time_part = tokens[1].split("_")[0]  # Hour
            return f"{tokens[0]}_{time_part}"
        return tokens[0]

    # Timestamped camera sessions: YYYYMMDD_HHMMSS
    if len(parts[0]) == 8 and parts[0].isdigit():
        hour = parts[1][:2] if len(parts) > 1 and len(parts[1]) >= 2 else "00"
        return f"cam_{parts[0]}_{hour}"

    if stem.startswith("exp"):
        return stem.split("_")[0]

    if "_processed" in stem:
        return stem.split("_processed")[0] + "_processed"

    clean_stem = stem.replace(" - copy", "").replace("(1)", "").replace("(2)", "").strip()
    return clean_stem[:10]


def build_revised_dataset():
    print("=" * 80)
    print("STARTING DATASET PREPARATION -> data_prepared_v1_revised")
    print("=" * 80)

    yolo_dir = OUT_DIR / "yolo_detector"
    clf_dir = OUT_DIR / "classifier"
    quarantine_dir = OUT_DIR / "quarantine"

    for split in ["train", "val", "test"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for c in ["Vertical", "defect free", "hole", "horizontal", "lines", "stain"]:
            (clf_dir / split / c).mkdir(parents=True, exist_ok=True)
    quarantine_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: Set[str] = set()
    quarantine_records: List[Dict] = []

    # ════════════════════════════════════════════════════════════════════════════
    # PART A: YOLO DETECTOR DATASET
    # ════════════════════════════════════════════════════════════════════════════
    print("\n[1/3] Preparing YOLO Detector Dataset (Revised Group-Level Split)...")
    photos_dir = REPO_ROOT / "photos"
    labels_dir = REPO_ROOT / "labels"

    yolo_groups = defaultdict(list)
    yolo_duplicates = 0

    for img_p in sorted(photos_dir.glob("*.*")):
        if not img_p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            continue
        sha = get_sha256(img_p)
        lbl_p = labels_dir / (img_p.stem + ".txt")

        if sha in seen_hashes:
            yolo_duplicates += 1
            quarantine_records.append({
                "file": str(img_p.name),
                "reason": "Duplicate SHA-256 hash in photos",
                "sha256": sha
            })
            continue

        if not lbl_p.exists():
            continue

        seen_hashes.add(sha)
        grp = extract_group_key_refined(img_p.name)
        yolo_groups[grp].append((img_p, lbl_p, sha))

    # Split YOLO defect groups (70% train, 15% val, 15% test)
    grp_keys = sorted(list(yolo_groups.keys()))
    random.seed(SEED)
    random.shuffle(grp_keys)

    n_grps = len(grp_keys)
    n_train = int(n_grps * 0.70)
    n_val = int(n_grps * 0.15)

    train_grps = set(grp_keys[:n_train])
    val_grps = set(grp_keys[n_train:n_train + n_val])
    test_grps = set(grp_keys[n_train + n_val:])

    yolo_counts = {"train": 0, "val": 0, "test": 0}
    yolo_box_counts = {"train": Counter(), "val": Counter(), "test": Counter()}

    for grp, items in yolo_groups.items():
        if grp in train_grps:
            split = "train"
        elif grp in val_grps:
            split = "val"
        else:
            split = "test"

        for img_p, lbl_p, sha in items:
            dest_img = yolo_dir / "images" / split / img_p.name
            dest_lbl = yolo_dir / "labels" / split / (img_p.stem + ".txt")
            shutil.copy2(img_p, dest_img)
            shutil.copy2(lbl_p, dest_lbl)
            yolo_counts[split] += 1

            with open(lbl_p) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        yolo_box_counts[split][int(parts[0])] += 1

    # Clean fabric negative images grouped by structure code
    clean_clf_dir = REPO_ROOT / "Fabric Defects Dataset" / "defect free"
    clean_structure_groups = defaultdict(list)
    for img_p in sorted(clean_clf_dir.glob("*.*")):
        if not img_p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
            continue
        sha = get_sha256(img_p)
        if sha in seen_hashes:
            continue
        grp = extract_group_key_refined(img_p.name)
        clean_structure_groups[grp].append((img_p, sha))

    # Split clean negative groups across train, val, test
    clean_grps = sorted(list(clean_structure_groups.keys()))
    random.seed(SEED + 99)
    random.shuffle(clean_grps)

    clean_counts = {"train": 0, "val": 0, "test": 0}
    # Aim for ~280 train, ~60 val, ~60 test clean negatives
    target_clean = {"train": 280, "val": 60, "test": 60}

    for idx, grp in enumerate(clean_grps):
        if idx % 10 in [0, 1]:
            split = "val"
        elif idx % 10 in [2, 3]:
            split = "test"
        else:
            split = "train"

        for img_p, sha in clean_structure_groups[grp]:
            if clean_counts[split] < target_clean[split]:
                dest_img = yolo_dir / "images" / split / ("clean_" + img_p.name)
                dest_lbl = yolo_dir / "labels" / split / ("clean_" + img_p.stem + ".txt")
                shutil.copy2(img_p, dest_img)
                dest_lbl.write_text("")  # Verified empty label for clean negative
                clean_counts[split] += 1
                yolo_counts[split] += 1
                seen_hashes.add(sha)

    # Write data.yaml
    yolo_yaml_content = f"""# Ultralytics YOLOv8 Dataset Configuration (Revised V1)
path: {yolo_dir.resolve().as_posix()}
train: images/train
val: images/val
test: images/test

names:
  0: hole
  1: objects
  2: oil spot
  3: thread error
"""
    (yolo_dir / "data.yaml").write_text(yolo_yaml_content)

    print(f"YOLO Dataset (Revised):")
    print(f"  Train: {yolo_counts['train']} images ({clean_counts['train']} clean negatives, {sum(yolo_box_counts['train'].values())} boxes: {dict(yolo_box_counts['train'])})")
    print(f"  Val:   {yolo_counts['val']} images ({clean_counts['val']} clean negatives, {sum(yolo_box_counts['val'].values())} boxes: {dict(yolo_box_counts['val'])})")
    print(f"  Test:  {yolo_counts['test']} images ({clean_counts['test']} clean negatives, {sum(yolo_box_counts['test'].values())} boxes: {dict(yolo_box_counts['test'])})")

    # ════════════════════════════════════════════════════════════════════════════
    # PART B: CLASSIFIER DATASET (REVISED SPLIT)
    # ════════════════════════════════════════════════════════════════════════════
    print("\n[2/3] Preparing MobileNetV3 Classifier Dataset (Guaranteeing Validation Coverage)...")
    clf_src_dir = REPO_ROOT / "Fabric Defects Dataset"
    clf_classes = ["Vertical", "defect free", "hole", "horizontal", "lines", "stain"]

    clf_counts = {"train": Counter(), "val": Counter(), "test": Counter()}
    clf_seen_hashes = set()
    clf_duplicates = 0

    for cname in clf_classes:
        src_folder = clf_src_dir / cname
        if not src_folder.exists():
            continue

        class_groups = defaultdict(list)
        for img_p in sorted(src_folder.glob("*.*")):
            if not img_p.suffix.lower() in [".jpg", ".png", ".jpeg"]:
                continue
            sha = get_sha256(img_p)
            if sha in clf_seen_hashes:
                clf_duplicates += 1
                quarantine_records.append({
                    "file": str(img_p.name),
                    "class": cname,
                    "reason": "Duplicate SHA-256 hash in Fabric Defects Dataset",
                    "sha256": sha
                })
                continue

            clf_seen_hashes.add(sha)
            grp = extract_group_key_refined(img_p.name)
            class_groups[grp].append(img_p)

        # Distribute groups across train, val, test ensuring at least 1 group in val and test
        c_grps = sorted(list(class_groups.keys()))
        random.seed(SEED + len(cname) * 7)
        random.shuffle(c_grps)

        n_g = len(c_grps)
        if n_g == 1:
            # Single group: must partition images directly within group to avoid 0-image validation
            imgs = class_groups[c_grps[0]]
            random.shuffle(imgs)
            n_tot = len(imgs)
            n_tr = max(1, int(n_tot * 0.70))
            n_va = max(1, int(n_tot * 0.15))
            splits_map = {
                "train": imgs[:n_tr],
                "val": imgs[n_tr:n_tr + n_va],
                "test": imgs[n_tr + n_va:]
            }
            for split, img_list in splits_map.items():
                for img_p in img_list:
                    dest_p = clf_dir / split / cname / img_p.name
                    shutil.copy2(img_p, dest_p)
                    clf_counts[split][cname] += 1
        elif n_g == 2:
            # 2 groups: 1 to train, 1 split between val/test
            tr_imgs = class_groups[c_grps[0]]
            other_imgs = class_groups[c_grps[1]]
            random.shuffle(other_imgs)
            mid = len(other_imgs) // 2
            splits_map = {
                "train": tr_imgs,
                "val": other_imgs[:mid],
                "test": other_imgs[mid:]
            }
            for split, img_list in splits_map.items():
                for img_p in img_list:
                    dest_p = clf_dir / split / cname / img_p.name
                    shutil.copy2(img_p, dest_p)
                    clf_counts[split][cname] += 1
        else:
            # 3 or more groups: Guarantee at least 1 group in val and at least 1 in test
            n_va_grps = max(1, int(n_g * 0.15))
            n_te_grps = max(1, int(n_g * 0.15))
            n_tr_grps = n_g - n_va_grps - n_te_grps

            splits_grps = {
                "train": c_grps[:n_tr_grps],
                "val": c_grps[n_tr_grps:n_tr_grps + n_va_grps],
                "test": c_grps[n_tr_grps + n_va_grps:]
            }

            for split, grp_list in splits_grps.items():
                for grp in grp_list:
                    for img_p in class_groups[grp]:
                        dest_p = clf_dir / split / cname / img_p.name
                        shutil.copy2(img_p, dest_p)
                        clf_counts[split][cname] += 1

    total_clf = sum(sum(c.values()) for c in clf_counts.values())
    print(f"Classifier Dataset (Revised):")
    print(f"  Train: {sum(clf_counts['train'].values())} images {dict(clf_counts['train'])}")
    print(f"  Val:   {sum(clf_counts['val'].values())} images {dict(clf_counts['val'])}")
    print(f"  Test:  {sum(clf_counts['test'].values())} images {dict(clf_counts['test'])}")
    print(f"  Total unique classifier images: {total_clf}")

    # ════════════════════════════════════════════════════════════════════════════
    # PART C: QUARANTINE MANIFEST & REVISED SUMMARY
    # ════════════════════════════════════════════════════════════════════════════
    with open(OUT_DIR / "quarantine_manifest.json", "w") as f:
        json.dump(quarantine_records, f, indent=2)

    summary = {
        "dataset_name": "Fabric_Defect_Prepared_v1_revised",
        "total_unique_original_images": sum(yolo_counts.values()) + total_clf - sum(clean_counts.values()),
        "yolo_detector": {
            "total_images": sum(yolo_counts.values()),
            "defective_images": sum(yolo_counts.values()) - sum(clean_counts.values()),
            "clean_negative_images": sum(clean_counts.values()),
            "split": yolo_counts,
            "classes": {0: "hole", 1: "objects", 2: "oil spot", 3: "thread error"},
            "bounding_boxes_by_split": {s: dict(yolo_box_counts[s]) for s in ["train", "val", "test"]}
        },
        "classifier": {
            "total_images": total_clf,
            "clean_images": sum(clf_counts[s]["defect free"] for s in ["train", "val", "test"]),
            "defective_images": total_clf - sum(clf_counts[s]["defect free"] for s in ["train", "val", "test"]),
            "classes": clf_classes,
            "split": {s: dict(clf_counts[s]) for s in ["train", "val", "test"]}
        }
    }

    with open(OUT_DIR / "dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved revised dataset summary to {OUT_DIR / 'dataset_summary.json'}")
    print("=" * 80)
    print("REVISED DATASET PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    build_revised_dataset()
