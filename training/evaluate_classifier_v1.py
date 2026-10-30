#!/usr/bin/env python3
"""
Classifier Evaluation Utility (Queue V1)

Evaluates MobileNetV3-Small classifier checkpoints on revised splits or synthetic smoke tests.

Features:
- Supported splits:
  - 'val': regular validation during development.
  - 'test': held-out test split (requires explicit --confirm-test-split).
  - 'synthetic': evaluation against synthetic colored/patterned samples (validation/ directory).
- Metrics reported:
  - Overall Multi-class Accuracy, Macro F1, Weighted F1.
  - Per-class Precision, Recall, F1-Score, and Support.
  - Binary Clean vs. Defective metrics (Binary Accuracy, Precision, Recall, F1, and False Alarm Rate).
  - Full Confusion Matrix.
  - Low-support class warnings (support < 30).
  - Mean inference latency per sample.
- Exports machine-readable metrics.json and detailed markdown report.
"""

import os
import sys
import json
import time
import argparse
import datetime
import pathlib
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)

# PyTorch 2.6+ weights_only compatibility patch
_orig_torch_load = torch.load
torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MobileNetV3 Fabric Classifier Checkpoint")
    parser.add_argument("--weights", type=str,
                        default=str(REPO_ROOT / "backend" / "models" / "fabric_classifier_best.pth"),
                        help="Path to classifier checkpoint (.pth)")
    parser.add_argument("--data-dir", type=str,
                        default=str(REPO_ROOT / "data_prepared_v1_revised" / "classifier"),
                        help="Path to revised classifier dataset root")
    parser.add_argument("--split", type=str, choices=["val", "test", "synthetic"], default="val",
                        help="Dataset partition to evaluate ('val', 'test', or 'synthetic')")
    parser.add_argument("--synthetic-dir", type=str,
                        default=str(REPO_ROOT / "validation"),
                        help="Path to synthetic fabric validation directory")
    parser.add_argument("--confirm-test-split", action="store_true",
                        help="Safety confirmation required to evaluate against the held-out test split")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for inference ('cpu' or 'cuda')")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for metrics and report")
    return parser.parse_args()

def load_classifier_model(weights_path: pathlib.Path, num_classes=6, device="cpu"):
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    checkpoint = torch.load(weights_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

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
    if not weights_path.exists():
        print(f"Error: Weights file does not exist: {weights_path}")
        sys.exit(1)

    if args.output_dir:
        out_dir = pathlib.Path(args.output_dir).resolve()
    else:
        out_dir = REPO_ROOT / "results" / "evaluations" / f"classifier_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print(f" MobileNetV3 Fabric Classifier Evaluation ({args.split.upper()} Split)")
    print("=" * 65)
    print(f"Weights    : {weights_path}")
    print(f"Split      : {args.split}")
    print(f"Device     : {args.device}")
    print(f"Output Dir : {out_dir}")
    print("=" * 65)

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    classes = ["Vertical", "defect free", "hole", "horizontal", "lines", "stain"]
    class_to_idx = {c: i for i, c in enumerate(classes)}
    model = load_classifier_model(weights_path, num_classes=len(classes), device=args.device)

    # Gather evaluation samples
    samples = []
    is_synthetic = False

    if args.split in ["val", "test"]:
        target_folder = pathlib.Path(args.data_dir) / args.split
        if not target_folder.exists():
            print(f"Error: Folder does not exist: {target_folder}")
            sys.exit(1)
        dataset = datasets.ImageFolder(str(target_folder), transform=None)
        classes = dataset.classes
        class_to_idx = dataset.class_to_idx
        for p, label in dataset.samples:
            samples.append((pathlib.Path(p), label, classes[label]))
    elif args.split == "synthetic":
        is_synthetic = True
        syn_dir = pathlib.Path(args.synthetic_dir)
        # Look in subdirectories of validation/
        for p in syn_dir.rglob("*.jpg"):
            # Expected binary label: clean vs defective from folder name
            parent = p.parent.name.lower()
            if "clean" in parent:
                expected_c = "defect free"
            elif "defective" in parent or "defect" in parent:
                expected_c = "hole"  # default defective representation
            else:
                expected_c = "unknown"
            label_idx = class_to_idx.get(expected_c, -1)
            samples.append((p, label_idx, expected_c))

    print(f"[Data] Evaluating {len(samples)} samples...")

    y_true = []
    y_pred = []
    latencies = []

    with torch.no_grad():
        for img_path, true_idx, true_name in samples:
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB")
                tensor = eval_transform(im_rgb).unsqueeze(0).to(args.device)

            t0 = time.time()
            out = model(tensor)
            dt = (time.time() - t0) * 1000.0
            latencies.append(dt)

            pred_idx = int(torch.argmax(out, dim=1).item())
            y_pred.append(pred_idx)
            y_true.append(true_idx)

    mean_latency = round(float(np.mean(latencies)), 2) if latencies else 0.0

    # Binary Clean vs Defective Analysis
    # Index 1 is 'defect free'
    defect_free_idx = class_to_idx["defect free"]
    binary_true = [0 if t == defect_free_idx else 1 for t in y_true]
    binary_pred = [0 if p == defect_free_idx else 1 for p in y_pred]

    bin_acc = accuracy_score(binary_true, binary_pred)
    bin_prec, bin_rec, bin_f1, _ = precision_recall_fscore_support(
        binary_true, binary_pred, average="binary", zero_division=0
    )

    clean_samples_total = sum(1 for b in binary_true if b == 0)
    clean_false_positives = sum(1 for bt, bp in zip(binary_true, binary_pred) if bt == 0 and bp == 1)
    clean_fp_rate = round(clean_false_positives / max(1, clean_samples_total), 4)

    # Multi-class metrics (for val and test splits)
    multi_acc = float(accuracy_score(y_true, y_pred)) if not is_synthetic else 0.0
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    prec_per, rec_per, f1_per, supp_per = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=list(range(len(classes))), zero_division=0
    )

    conf_mat = confusion_matrix(y_true, y_pred, labels=list(range(len(classes)))).tolist()

    per_class_metrics = {}
    for idx, cname in enumerate(classes):
        per_class_metrics[cname] = {
            "precision": round(float(prec_per[idx]), 4),
            "recall": round(float(rec_per[idx]), 4),
            "f1_score": round(float(f1_per[idx]), 4),
            "support": int(supp_per[idx]),
            "low_support_flag": int(supp_per[idx]) < 30
        }

    metrics_payload = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "weights": str(weights_path),
        "split": args.split,
        "is_synthetic": is_synthetic,
        "total_samples": len(samples),
        "mean_latency_ms": mean_latency,
        "multiclass": {
            "accuracy": round(multi_acc, 4),
            "macro_precision": round(float(prec_macro), 4),
            "macro_recall": round(float(rec_macro), 4),
            "macro_f1": round(float(f1_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "per_class": per_class_metrics
        },
        "binary_clean_vs_defective": {
            "accuracy": round(float(bin_acc), 4),
            "precision": round(float(bin_prec), 4),
            "recall": round(float(bin_rec), 4),
            "f1_score": round(float(bin_f1), 4),
            "clean_samples_total": clean_samples_total,
            "clean_false_positives": clean_false_positives,
            "clean_false_positive_rate": clean_fp_rate
        },
        "confusion_matrix": conf_mat,
        "classes": classes
    }

    metrics_json_path = out_dir / "metrics.json"
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    report_md_path = out_dir / "evaluation_report.md"
    report_content = f"""# MobileNetV3 Fabric Classifier Evaluation Report ({args.split.upper()} Split)

**Evaluation Date**: {datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Weights**: `{weights_path.name}`  
**Dataset Split**: `{args.split}` ({len(samples)} total samples)  
**Device**: `{args.device}`  
**Mean Latency per Sample**: `{mean_latency} ms`  

---

## 1. Executive Performance Summary

| Metric | Overall Multi-Class | Binary (Clean vs. Defective) |
|---|---|---|
| **Accuracy** | {multi_acc:.4f} | {bin_acc:.4f} |
| **Macro / Binary Precision** | {prec_macro:.4f} | {bin_prec:.4f} |
| **Macro / Binary Recall** | {rec_macro:.4f} | {bin_rec:.4f} |
| **Macro / Binary F1** | {f1_macro:.4f} | {bin_f1:.4f} |
| **Weighted F1** | {f1_weighted:.4f} | — |

---

## 2. Per-Class Precision, Recall, F1, and Support

| Class | Precision | Recall | F1-Score | Support | Status |
|---|---|---|---|---|---|
"""
    for cname, pcm in per_class_metrics.items():
        flag = "⚠️ Low Support (<30)" if pcm["low_support_flag"] else "OK"
        report_content += f"| **{cname}** | {pcm['precision']:.4f} | {pcm['recall']:.4f} | {pcm['f1_score']:.4f} | {pcm['support']} | {flag} |\n"

    report_content += f"""
---

## 3. False Alarm and Quality Inspection Integrity

| Quality Metric | Ground Truth Samples | False Positives | False Alarm Rate |
|---|---|---|---|
| **Clean Negative Samples ('defect free')** | {clean_samples_total} | {clean_false_positives} | {clean_fp_rate * 100:.2f}% |

---

## 4. Confusion Matrix

```
Predicted ->   {'  '.join([f'{c[:5]:>5}' for c in classes])}
"""
    for idx, row in enumerate(conf_mat):
        row_str = "  ".join([f"{val:>5}" for val in row])
        report_content += f"{classes[idx][:12]:<12}: {row_str}\n"

    report_content += f"""```

---

## 5. Dataset Split Notice

> [!NOTE]
> This evaluation was conducted on the **`{args.split}`** split.
> If synthetic images were evaluated, results reflect synthetic smoke testing, not real-world benchmark proof.
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print("\n" + "=" * 65)
    print(" [CLASSIFIER EVALUATION COMPLETE]")
    print(f" Accuracy    : {multi_acc:.4f} | Macro F1: {f1_macro:.4f}")
    print(f" Binary F1   : {bin_f1:.4f}    | Clean FP Rate: {clean_fp_rate * 100:.1f}%")
    print(f" Mean Lat    : {mean_latency} ms")
    print(f" Metrics     : {metrics_json_path}")
    print(f" Report      : {report_md_path}")
    print("=" * 65)

if __name__ == "__main__":
    main()
