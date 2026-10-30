#!/usr/bin/env python3
"""
Detector Evaluation Utility (Queue V1)

Evaluates YOLO detector checkpoints on the revised fabric detector dataset.

Features:
- Separate evaluation for 'val' and 'test' splits.
- Strict test set preservation: requires --confirm-test-split to prevent inadvertent data leakage.
- Metrics reported:
  - Overall Precision, Recall, mAP@0.50, mAP@0.50:0.95.
  - Per-class AP@0.50 and AP@0.50:0.95.
  - Detection counts per class.
  - False positive count & rate on clean negative samples.
  - False negative count & rate on defective samples.
  - Mean inference latency per frame.
- Exports machine-readable metrics.json and structured markdown report.
"""

import os
import sys
import json
import time
import argparse
import datetime
import pathlib
import yaml
import torch
import numpy as np

# PyTorch 2.6+ weights_only compatibility patch
_orig_torch_load = torch.load
torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Detector Checkpoint")
    parser.add_argument("--weights", type=str,
                        default=str(REPO_ROOT / "results" / "yolov8s_fabric_full" / "weights" / "best.pt"),
                        help="Path to YOLO checkpoint (.pt)")
    parser.add_argument("--data", type=str,
                        default=str(REPO_ROOT / "data_prepared_v1_revised" / "yolo_detector" / "data.yaml"),
                        help="Path to data.yaml dataset specification")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="val",
                        help="Dataset split to evaluate ('val' or 'test')")
    parser.add_argument("--confirm-test-split", action="store_true",
                        help="Safety confirmation required to evaluate against the held-out test split")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image resolution")
    parser.add_argument("--conf", type=float, default=0.08,
                        help="Confidence threshold for evaluation")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="NMS IoU threshold")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference ('cpu' or '0')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save evaluation report and metrics.json")
    return parser.parse_args()

def main():
    args = parse_args()

    if args.split == "test" and not args.confirm_test_split:
        print("\n" + "=" * 65)
        print(" [SAFETY BLOCK] HELD-OUT TEST SPLIT EVALUATION BLOCKED")
        print("=" * 65)
        print("Safety Rule #7 & Job 6 require preserving the held-out test set.")
        print("The test split is reserved exclusively for final model benchmarking.")
        print("To run final evaluation on a completed candidate model, pass --confirm-test-split.")
        print("=" * 65)
        sys.exit(1)

    weights_path = pathlib.Path(args.weights).resolve()
    data_yaml_path = pathlib.Path(args.data).resolve()

    if not weights_path.exists():
        print(f"Error: Weights file does not exist: {weights_path}")
        sys.exit(1)
    if not data_yaml_path.exists():
        print(f"Error: data.yaml does not exist: {data_yaml_path}")
        sys.exit(1)

    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).resolve()
    else:
        out_dir = REPO_ROOT / "results" / "evaluations" / f"detector_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f" YOLO Fabric Detector Evaluation (Split: {args.split.upper()})")
    print("=" * 65)
    print(f"Weights     : {weights_path}")
    print(f"Dataset YAML: {data_yaml_path}")
    print(f"Split       : {args.split}")
    print(f"Device      : {args.device}")
    print(f"Output Dir  : {out_dir}")
    print("=" * 65)

    from ultralytics import YOLO
    model = YOLO(str(weights_path))

    # Run validation via Ultralytics val API
    start_time = time.time()
    val_results = model.val(
        data=str(data_yaml_path),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save_json=False,
        plots=False,
        verbose=False
    )
    total_val_time = time.time() - start_time

    # Parse metrics
    box_metrics = val_results.box
    precision = float(box_metrics.mp)
    recall = float(box_metrics.mr)
    map50 = float(box_metrics.map50)
    map50_95 = float(box_metrics.map)

    # Per-class AP
    class_names = val_results.names
    per_class_ap50 = {}
    per_class_ap = {}
    for idx, cname in class_names.items():
        if idx < len(box_metrics.ap50):
            per_class_ap50[cname] = round(float(box_metrics.ap50[idx]), 4)
            per_class_ap[cname] = round(float(box_metrics.ap[idx]), 4)

    # Detailed image-level false positive / false negative evaluation
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    yaml_dir = data_yaml_path.parent
    split_img_dir = yaml_dir / data_cfg[args.split] if not pathlib.Path(data_cfg[args.split]).is_absolute() else pathlib.Path(data_cfg[args.split])
    split_lbl_dir = yaml_dir / "labels" / args.split

    clean_total = 0
    clean_false_positives = 0
    defect_total = 0
    defect_false_negatives = 0
    total_latency_ms = 0.0
    image_count = 0

    image_files = sorted(list(split_img_dir.glob("*.jpg")))
    for img_p in image_files:
        image_count += 1
        lbl_p = split_lbl_dir / (img_p.stem + ".txt")
        has_gt_boxes = False
        if lbl_p.exists():
            with open(lbl_p, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
                has_gt_boxes = len(lines) > 0

        # Run model inference for this image
        t0 = time.time()
        preds = model(str(img_p), conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                      device=args.device, agnostic_nms=True, verbose=False)[0]
        dt = (time.time() - t0) * 1000.0
        total_latency_ms += dt

        det_count = len(preds.boxes)
        if not has_gt_boxes:
            clean_total += 1
            if det_count > 0:
                clean_false_positives += 1
        else:
            defect_total += 1
            if det_count == 0:
                defect_false_negatives += 1

    mean_latency = round(total_latency_ms / max(1, image_count), 2)
    clean_fp_rate = round(clean_false_positives / max(1, clean_total), 4)
    defect_fn_rate = round(defect_false_negatives / max(1, defect_total), 4)

    metrics_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "weights": str(weights_path),
        "split": args.split,
        "total_images": image_count,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "mAP50": round(map50, 4),
        "mAP50_95": round(map50_95, 4),
        "per_class_ap50": per_class_ap50,
        "per_class_ap50_95": per_class_ap,
        "clean_negative_images": clean_total,
        "clean_false_positives": clean_false_positives,
        "clean_false_positive_rate": clean_fp_rate,
        "defective_images": defect_total,
        "defective_false_negatives": defect_false_negatives,
        "defective_false_negative_rate": defect_fn_rate,
        "mean_latency_ms": mean_latency
    }

    metrics_json_path = out_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    report_md_path = out_dir / "evaluation_report.md"
    report_content = f"""# YOLO Detector Evaluation Report ({args.split.upper()} Split)

**Evaluation Date**: {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Weights**: `{weights_path.name}`  
**Dataset Split**: `{args.split}` ({image_count} total images)  
**Confidence Threshold**: `{args.conf}` | **NMS IoU**: `{args.iou}` | **Device**: `{args.device}`  

---

## 1. Primary Localization Performance

| Metric | Value |
|---|---|
| **Precision** | {precision:.4f} |
| **Recall** | {recall:.4f} |
| **mAP@0.50** | {map50:.4f} |
| **mAP@0.50:0.95** | {map50_95:.4f} |
| **Mean Latency per Frame** | {mean_latency} ms |

---

## 2. Per-Class Average Precision (AP)

| Class | AP@0.50 | AP@0.50:0.95 |
|---|---|---|
"""
    for cname in class_names.values():
        report_content += f"| **{cname}** | {per_class_ap50.get(cname, 0.0):.4f} | {per_class_ap.get(cname, 0.0):.4f} |\n"

    report_content += f"""
---

## 3. Negative Sample and Defect Detection Integrity

| Assessment | Total Ground Truth | Errors Observed | Error Rate |
|---|---|---|---|
| **Clean Negative Images (False Alarms)** | {clean_total} | {clean_false_positives} | {clean_fp_rate * 100:.2f}% |
| **Defective Images (Missed Defects)**   | {defect_total} | {defect_false_negatives} | {defect_fn_rate * 100:.2f}% |

---

## 4. Split Safety Notice

> [!NOTE]
> This evaluation was performed on the **`{args.split}`** split.
> The held-out test split is strictly reserved for final verification of candidate checkpoints.
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 65)
    print(" [DETECTOR EVALUATION COMPLETE]")
    print(f" Precision : {precision:.4f} | Recall: {recall:.4f}")
    print(f" mAP50     : {map50:.4f}     | mAP50-95: {map50_95:.4f}")
    print(f" Clean FP  : {clean_false_positives}/{clean_total} ({clean_fp_rate * 100:.1f}%)")
    print(f" Defect FN : {defect_false_negatives}/{defect_total} ({defect_fn_rate * 100:.1f}%)")
    print(f" Mean Lat  : {mean_latency} ms")
    print(f" Metrics   : {metrics_json_path}")
    print(f" Report    : {report_md_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
