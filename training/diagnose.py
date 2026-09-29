"""
Full diagnostic script for the AI Fabric Defect Detection pipeline.
Run from: d:\PROJECTS\AI-Fabric-Defect-Detection
"""
import os
import sys
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── 1. Dataset analysis ───────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: DATASET ANALYSIS")
print("=" * 60)

labels_dir = "labels"
photos_dir = "photos"

CLASS_NAMES = {0: "hole", 1: "objects", 2: "oil spot", 3: "thread error"}

label_files = [f for f in os.listdir(labels_dir) if f.endswith(".txt")]
image_files = [f for f in os.listdir(photos_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

print(f"Total label files : {len(label_files)}")
print(f"Total image files : {len(image_files)}")

# Count annotations per class
class_counts = defaultdict(int)
images_per_class = defaultdict(list)
empty_labels = []

for lf in label_files:
    with open(os.path.join(labels_dir, lf)) as f:
        lines = [l.strip() for l in f if l.strip()]
    
    if not lines:
        empty_labels.append(lf)
    
    for line in lines:
        parts = line.split()
        if parts:
            cls = int(parts[0])
            class_counts[cls] += 1
            base = lf.replace(".txt", "")
            if base not in images_per_class[cls]:
                images_per_class[cls].append(base)

print(f"\nEmpty label files (background/good): {len(empty_labels)}")
print("\nAnnotation counts per class:")
for cls_id, name in CLASS_NAMES.items():
    count = class_counts[cls_id]
    imgs = len(images_per_class[cls_id])
    print(f"  {cls_id}: {name:15s} → {count:4d} annotations in {imgs:3d} images")

# Find 3 thread-error images
thread_error_imgs = images_per_class[3][:3]
print(f"\nSample thread error image bases: {thread_error_imgs}")

# ─── 2. Direct model inference ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: DIRECT MODEL INFERENCE")
print("=" * 60)

try:
    from ultralytics import YOLO

    model_path = os.path.join("backend", "models", "best.pt")
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    print(f"Model classes: {model.names}")

    # Test on thread error images
    test_images = []
    for base in thread_error_imgs:
        for ext in [".jpg", ".jpeg", ".png"]:
            img_path = os.path.join(photos_dir, base + ext)
            if os.path.exists(img_path):
                test_images.append(img_path)
                break

    # Also pick one hole, one oil-spot
    for cls_id, name in [(0, "hole"), (2, "oil spot")]:
        bases = images_per_class[cls_id][:1]
        for base in bases:
            for ext in [".jpg", ".jpeg", ".png"]:
                img_path = os.path.join(photos_dir, base + ext)
                if os.path.exists(img_path):
                    test_images.append(img_path)
                    break

    print(f"\nTest images: {test_images}\n")

    conf_thresholds = [0.25, 0.15, 0.10, 0.05]

    for img_path in test_images:
        label_path = os.path.join(labels_dir, Path(img_path).stem + ".txt")
        with open(label_path) as f:
            gt_lines = [l.strip() for l in f if l.strip()]
        gt_classes = [CLASS_NAMES.get(int(l.split()[0]), "?") for l in gt_lines]

        print(f"─── Image: {os.path.basename(img_path)}")
        print(f"    Ground truth: {gt_classes}")
        
        for conf in conf_thresholds:
            results = model(img_path, conf=conf, verbose=False)[0]
            dets = []
            for box in results.boxes:
                cid = int(box.cls[0].item())
                cv = float(box.conf[0].item())
                dets.append(f"{CLASS_NAMES.get(cid,'?')}({cv:.3f})")
            print(f"    conf={conf:.2f}: {dets if dets else '(no detections)'}")
        print()

    # ─── 3. Check model's native confidence threshold ──────────────────────────
    print("=" * 60)
    print("STEP 3: MODEL DEFAULT PARAMETERS")
    print("=" * 60)
    print(f"Model task      : {model.task}")
    print(f"Model overrides : {model.overrides}")

    # Try with very low threshold to see ALL detections
    print("\nRunning with conf=0.01 on first test image to see all raw scores:")
    if test_images:
        results = model(test_images[0], conf=0.01, verbose=False)[0]
        print(f"Raw detections at conf=0.01: {len(results.boxes)}")
        for box in results.boxes[:10]:
            cid = int(box.cls[0].item())
            cv = float(box.conf[0].item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            print(f"  {CLASS_NAMES.get(cid,'?'):15s} conf={cv:.4f}  bbox=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")

    # ─── 4. Check existing training results ────────────────────────────────────
    print("\n" + "=" * 60)
    print("STEP 4: TRAINING METRICS")
    print("=" * 60)
    metrics_path = os.path.join("results", "evaluation", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(json.dumps(metrics, indent=2))
    else:
        print("No metrics.json found at", metrics_path)

    # Check results.csv for training history
    csv_path = os.path.join("results", "yolov8_fabric_fast", "results.csv")
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            lines = f.readlines()
        print(f"\nTraining results.csv (last 3 rows):")
        print(lines[0].strip())  # header
        for row in lines[-3:]:
            print(row.strip())

except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)
