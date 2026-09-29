# Project Overview: AI-Based Fabric Defect Detection — Hybrid Pipeline

## Problem Statement

Manual inspection in the textile manufacturing pipeline is notoriously slow, subject to human fatigue, and often inconsistent. Minor defects are frequently missed, leading to compromised final product quality and significant financial losses.

---

## Architecture Overview

The system now uses a **Hybrid Pipeline** combining two independent AI models:

```
                    INPUT IMAGE
                         │
              ┌──────────┴──────────┐
              │                     │
         YOLO Detector         MobileNetV3-Small
         (TILDA-400)           Classifier
         Detects + Localizes   Classifies fabric type
         Provides bounding     No bounding box
         boxes                 (classification only)
              │                     │
              └──────────┬──────────┘
                         │
                 Decision Logic
                         │
               ┌─────────┴─────────┐
               │                   │
        YOLO detects         YOLO finds nothing
               │                   │
           DEFECTIVE        Classifier ≥ 0.55 and
                            defective class?
                                   │
                          ┌────────┴────────┐
                          │                 │
                     YES (≥0.55)       NO / defect free
                          │                 │
                      DEFECTIVE           PASS
```

This resolves the primary problem of **false negatives** — cases where the YOLO model returned PASS on defective images that lie outside the TILDA-400 texture distribution.

---

## Datasets

### Dataset 1 — TILDA-400 (YOLO Detection)

| Property        | Value                                      |
|-----------------|--------------------------------------------|
| Name            | TILDA-400                                  |
| Size            | ~400 images                                |
| Annotation type | YOLO bounding box labels (.txt)            |
| Classes         | hole, objects, oil spot, thread error      |
| Purpose         | Defect localization with bounding boxes    |
| Location        | `photos/` (images), `labels/` (annotations) |

### Dataset 2 — Fabric Defects Dataset (Classifier)

| Property        | Value                                                        |
|-----------------|--------------------------------------------------------------|
| Name            | Fabric Defects Dataset                                       |
| Size            | ~2000 images                                                 |
| Annotation type | Folder-based image classification (NO bounding boxes)        |
| Classes         | Vertical, defect free, hole, horizontal, lines, stain        |
| Purpose         | Catching defects missed by YOLO; reducing false negatives    |
| Location        | `Fabric Defects Dataset/`                                    |

**Why two separate datasets?**
- TILDA-400 is an object-detection dataset with bounding box annotations. It powers the YOLO model for localization.
- The Fabric Defects Dataset is a classification dataset (folder-per-class, no bounding box annotations). Bounding boxes were NOT fabricated — the classifier provides class + confidence only.
- These are genuinely different tasks and require different models.

---

## Model Architecture

### Model 1 — YOLOv8 (TILDA-400)
- **Architecture**: YOLOv8n / YOLOv8s
- **Task**: Object detection (bounding box regression + classification)
- **Output**: `[class_name, confidence, bbox: {x1,y1,x2,y2}]`
- **Calibrated thresholds**: thread_error=0.06, hole=0.08, objects=0.10, oil_spot=0.15

### Model 2 — MobileNetV3-Small (Fabric Defects Dataset)
- **Architecture**: MobileNetV3-Small with custom classification head
- **Pretrained weights**: ImageNet (transfer learning)
- **Fine-tuned**: 2-stage training (head warmup + full network fine-tuning)
- **Task**: Image classification (6 classes)
- **Output**: `[class_name, confidence, all_probs]` — **NO bounding box**
- **Decision threshold**: 0.55 (only flags DEFECTIVE if confidence ≥ 55%)
- **Trained with**: Class-balanced weights, data augmentation, CosineAnnealingLR, early stopping

---

## Training Pipeline

### Classifier Training (`training/train_classifier.py`)
1. Dataset scanned; exact duplicate images removed (MD5 hash check)
2. Stratified train (70%) / val (15%) / test (15%) split — no data leakage
3. Class weights computed to handle imbalance
4. **Stage 1** (3 epochs): Frozen features — train head only
5. **Stage 2** (up to 12 epochs): Full network fine-tuning with CosineAnnealingLR
6. Best checkpoint saved by Val Macro-F1
7. Training history plot saved to `results/new_classifier/`
8. Split info saved to `results/new_classifier/dataset_split.json`

### Classifier Evaluation (`training/evaluate_classifier.py`)
- Runs on held-out test set (never seen during training)
- Reports: Accuracy, Macro Precision/Recall/F1, Weighted F1, per-class metrics
- Binary quality metrics (DEFECTIVE vs PASS)
- Saves confusion matrix PNG
- Sample predictions from every class

---

## Inference Workflow (Final Application)

```
POST /api/predict (multipart image)
          │
     preprocess_image (OpenCV BGR decode)
          │
   ┌──────┴──────┐
   │             │
 YOLO          Classifier
 predict       predict
   │             │
   └──────┬──────┘
          │
   Decision Logic:
   1. If YOLO detections exist → DEFECTIVE (bounding boxes shown)
   2. Elif Classifier is_defective AND confidence ≥ 0.55 → DEFECTIVE (no bbox shown)
   3. Else → PASS
          │
   PredictionResponse (JSON):
   {
     "quality_status": "DEFECTIVE" | "PASS",
     "primary_defect": "stain" | "hole" | ...,
     "confidence": 0.92,
     "detection_source": "yolo" | "classifier" | "both" | "none",
     "detections": [...],          ← YOLO only (have bboxes)
     "classification": {...},      ← Classifier only (no bbox)
     "defects": [...],             ← Same as detections (backward compat)
     "defect_count": N
   }
```

---

## System Architecture

| Layer     | Technology                                  |
|-----------|---------------------------------------------|
| Frontend  | React 18 + Vite + Tailwind CSS              |
| Backend   | FastAPI + uvicorn                           |
| Detection | Ultralytics YOLOv8 (TILDA-400)             |
| Classifier| PyTorch + MobileNetV3-Small                 |
| Serving   | Two model services loaded at startup        |

---

## False Negative Reduction Strategy

| Scenario                              | YOLO Result | Classifier Result | Final Decision |
|---------------------------------------|-------------|-------------------|----------------|
| YOLO detects hole/objects/oil/thread  | DEFECTIVE   | Any               | **DEFECTIVE** (YOLO wins) |
| YOLO misses; classifier sees stain    | PASS (miss) | stain 87%         | **DEFECTIVE** (classifier rescues) |
| YOLO misses; classifier sees defect free | PASS    | defect free 91%   | **PASS**       |
| YOLO misses; classifier uncertain     | PASS        | stain 38%         | **PASS** (below threshold) |

The 0.55 threshold was selected conservatively to minimize false positives while catching high-confidence missed defects.

---

## Evaluation Results

Actual metrics are stored in:
- `results/new_classifier/classifier_metrics.json`
- `results/new_classifier/classifier_confusion_matrix.png`
- `results/new_classifier/classifier_training_history.png`

---

## Important Notes on Bounding Boxes

> **The classifier does NOT produce bounding boxes.**

The Fabric Defects Dataset is a folder-based classification dataset. No bounding box annotations exist. The frontend is explicitly designed to:
- Show bounding boxes **only** for YOLO detections
- Show class + confidence (no box) for classifier-only results
- Display a clear "Classification result — no bounding box available" note

Bounding boxes were never fabricated for the classifier.

---

## Limitations

- YOLO limited to TILDA-400 texture distribution; rare fabric types may miss defects
- Classifier cannot localize where the defect is within the image
- Classifier has 6 classes, YOLO has 4 — some overlap (hole class present in both datasets)
- High-resolution microscopic defects may not be detected at 640×640 input resolution

---

## Future Scope

- **Real-Time Camera Integration**: Direct video feed inspection on weaving machines
- **Weak Supervision**: Generate approximate bounding boxes from classifier attention maps (Grad-CAM)
- **Unified Model**: Train a single multi-task model if annotated bounding boxes become available for all classes
- **Defect Severity**: Classify severity based on defect surface area
- **Edge Deployment**: Running on Jetson Nano or similar hardware for factory floors
