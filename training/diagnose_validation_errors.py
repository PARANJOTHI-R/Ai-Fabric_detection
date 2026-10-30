import json
from pathlib import Path
from collections import defaultdict

import cv2
import torch

# For this trusted local YOLO checkpoint only
_original_torch_load = torch.load
torch.load = lambda *args, **kwargs: _original_torch_load(
    *args, **{**kwargs, "weights_only": False}
)

from ultralytics import YOLO
ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
IMAGE_DIR = DATASET / "images" / "val"
LABEL_DIR = DATASET / "labels" / "val"
MODEL_PATH = ROOT / "results" / "yolov8s_fabric_full" / "weights" / "best.pt"
OUT_PATH = ROOT / "results" / "evaluation" / "detection_diagnostic.json"

CLASS_NAMES = {
    0: "hole",
    1: "objects",
    2: "oil spot",
    3: "thread error",
}
IOU_MATCH = 0.50
PRED_CONF = 0.001

def read_ground_truth(label_path):
    boxes = []
    if not label_path.exists():
        raise FileNotFoundError(f"Missing label: {label_path}")

    for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid label {label_path}, line {line_number}")
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
        if cls not in CLASS_NAMES:
            raise ValueError(f"Unexpected class {cls} in {label_path}")
        boxes.append({
            "cls": cls,
            "box": [
                (cx - w / 2), (cy - h / 2),
                (cx + w / 2), (cy + h / 2)
            ],
        })
    return boxes

def iou_xyxy(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0

def match_boxes(gt, pred, width, height):
    candidates = []
    for gi, g in enumerate(gt):
        gbox = [g["box"][0] * width, g["box"][1] * height,
                g["box"][2] * width, g["box"][3] * height]
        for pi, p in enumerate(pred):
            overlap = iou_xyxy(gbox, p["box"])
            if overlap >= IOU_MATCH:
                candidates.append((overlap, gi, pi))

    candidates.sort(reverse=True)
    used_gt, used_pred, matches = set(), set(), []
    for overlap, gi, pi in candidates:
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        matches.append((gi, pi, overlap))
    return used_gt, used_pred, matches

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Active model not found: {MODEL_PATH}")
    if not IMAGE_DIR.exists() or not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Expected validation images and labels at:\n{IMAGE_DIR}\n{LABEL_DIR}"
        )

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not images:
        raise RuntimeError(f"No validation images found in {IMAGE_DIR}")

    print(f"Model: {MODEL_PATH}")
    print(f"Validation images: {len(images)}")
    print(f"Prediction confidence floor: {PRED_CONF}")
    print(f"Class-agnostic matching IoU: {IOU_MATCH}")
    print("Running inference; this may take a few minutes...")

    model = YOLO(str(MODEL_PATH))
    counts = defaultdict(lambda: {
        "ground_truth": 0,
        "correct_class_match_iou50": 0,
        "wrong_class_match_iou50": 0,
        "missed_no_match_iou50": 0,
        "matched_prediction_conf_below_0_05": 0,
    })
    confusions = defaultdict(int)
    false_positives = defaultdict(int)
    details = []

    for index, image_path in enumerate(images, 1):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        gt = read_ground_truth(label_path)

        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]

        result = model.predict(
            source=image, conf=PRED_CONF, iou=0.7, imgsz=640, verbose=False
        )[0]

        pred = []
        for box in result.boxes:
            pred.append({
                "cls": int(box.cls[0].item()),
                "conf": float(box.conf[0].item()),
                "box": [float(v) for v in box.xyxy[0].tolist()],
            })

        used_gt, used_pred, matches = match_boxes(gt, pred, width, height)
        image_detail = {
            "image": image_path.name,
            "ground_truth_count": len(gt),
            "prediction_count": len(pred),
            "missed": [],
            "class_confusions": [],
            "false_positives": [],
        }

        for g in gt:
            counts[g["cls"]]["ground_truth"] += 1

        for gi, pi, overlap in matches:
            g, p = gt[gi], pred[pi]
            cls_name = CLASS_NAMES[g["cls"]]
            if g["cls"] == p["cls"]:
                counts[g["cls"]]["correct_class_match_iou50"] += 1
            else:
                counts[g["cls"]]["wrong_class_match_iou50"] += 1
                confusions[(g["cls"], p["cls"])] += 1
                image_detail["class_confusions"].append({
                    "true_class": CLASS_NAMES[g["cls"]],
                    "predicted_class": CLASS_NAMES[p["cls"]],
                    "iou": round(overlap, 3),
                    "confidence": round(p["conf"], 4),
                })
            if p["conf"] < 0.05:
                counts[g["cls"]]["matched_prediction_conf_below_0_05"] += 1

        for gi, g in enumerate(gt):
            if gi not in used_gt:
                counts[g["cls"]]["missed_no_match_iou50"] += 1
                image_detail["missed"].append({
                    "true_class": CLASS_NAMES[g["cls"]],
                    "reason": "No prediction reached IoU 0.50",
                })

        for pi, p in enumerate(pred):
            if pi not in used_pred:
                false_positives[p["cls"]] += 1
                image_detail["false_positives"].append({
                    "predicted_class": CLASS_NAMES[p["cls"]],
                    "confidence": round(p["conf"], 4),
                })

        if image_detail["missed"] or image_detail["class_confusions"] or image_detail["false_positives"]:
            details.append(image_detail)

        if index % 10 == 0 or index == len(images):
            print(f"Processed {index}/{len(images)} images")

    print("\nCLASS-WISE DIAGNOSTIC")
    print("Note: class-agnostic IoU matching; not official Ultralytics mAP.")
    summary = {}
    for cls_id, name in CLASS_NAMES.items():
        c = counts[cls_id]
        total = c["ground_truth"]
        matched = c["correct_class_match_iou50"]
        wrong = c["wrong_class_match_iou50"]
        missed = c["missed_no_match_iou50"]
        recall = matched / total if total else 0.0
        summary[name] = dict(c)
        summary[name]["recall_correct_class_iou50"] = round(recall, 4)
        print(
            f"{name:15s} GT={total:3d} correct={matched:3d} "
            f"class_mismatch={wrong:3d} no_match={missed:3d} "
            f"correct-class recall={recall:.1%} "
            f"matched_conf<0.05={c['matched_prediction_conf_below_0_05']}"
        )

    print("\nCLASS CONFUSIONS (true -> predicted)")
    if confusions:
        for (true_id, pred_id), count in sorted(
            confusions.items(), key=lambda item: item[1], reverse=True
        ):
            print(f"{CLASS_NAMES[true_id]} -> {CLASS_NAMES[pred_id]}: {count}")
    else:
        print("None at IoU 0.50")

    print("\nUNMATCHED PREDICTIONS (potential false positives)")
    for cls_id, name in CLASS_NAMES.items():
        print(f"{name:15s}: {false_positives[cls_id]}")

    output = {
        "model": str(MODEL_PATH),
        "validation_images": len(images),
        "prediction_confidence_floor": PRED_CONF,
        "matching_iou": IOU_MATCH,
        "method_note": (
            "Class-agnostic greedy IoU matching at 0.50 with predictions "
            "generated at confidence 0.001. This is a diagnostic, not official mAP."
        ),
        "summary_by_class": summary,
        "class_confusions": [
            {"true_class": CLASS_NAMES[a], "predicted_class": CLASS_NAMES[b], "count": n}
            for (a, b), n in sorted(confusions.items())
        ],
        "unmatched_predictions_by_class": {
            CLASS_NAMES[k]: v for k, v in false_positives.items()
        },
        "problem_images": details,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nDetailed report saved to: {OUT_PATH}")

if __name__ == "__main__":
    main()

