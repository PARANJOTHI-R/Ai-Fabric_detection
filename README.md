# AI-Based Fabric Defect Detection — Hybrid Pipeline

> **YOLOv8 detection** (TILDA-400) + **MobileNetV3-Small classifier** (Fabric Defects Dataset)  
> Reduces false negatives without fabricating bounding boxes.

---

## System Architecture

```
                    INPUT IMAGE
                         │
              ┌──────────┴──────────┐
              │                     │
         YOLOv8 Detector       MobileNetV3-Small
         (TILDA-400)           Classifier
         4 classes             6 classes
         Bounding boxes ✓      No bounding box ✗
              │                     │
              └──────────┬──────────┘
                         │
                 Decision Logic
                         │
               ┌─────────┴─────────┐
               │                   │
        YOLO detects         YOLO finds nothing
               │                   │
           DEFECTIVE        Classifier ≥ 0.55
                            AND defective class?
                                   │
                          ┌────────┴────────┐
                          │                 │
                     YES → DEFECTIVE    NO → PASS
```

---

## Datasets

### Dataset 1 — TILDA-400 (YOLO Object Detection)
| Property  | Value                                       |
|-----------|---------------------------------------------|
| Images    | ~400                                        |
| Format    | YOLO `.txt` bounding box annotations        |
| Classes   | hole · objects · oil spot · thread error    |
| Location  | `photos/` + `labels/`                       |

### Dataset 2 — Fabric Defects Dataset (Classification)
| Property  | Value                                              |
|-----------|----------------------------------------------------|
| Images    | 2,736 raw images (2,715 unique after deduplication)|
| Format    | Folder-per-class (NO bounding boxes)               |
| Classes   | Vertical · defect free · hole · horizontal · lines · stain |
| Location  | `Fabric Defects Dataset/`                          |

> ⚠️ The classifier dataset has **no bounding box annotations**.  
> Bounding boxes were **not fabricated**. The classifier provides class + confidence only.

---

## Defect Classes

### YOLO (with localization)
| ID | Class        |
|----|--------------|
| 0  | hole         |
| 1  | objects      |
| 2  | oil spot     |
| 3  | thread error |

### Classifier (no localization)
| Class       | Quality Status |
|-------------|---------------|
| Vertical    | DEFECTIVE     |
| defect free | **PASS**      |
| hole        | DEFECTIVE     |
| horizontal  | DEFECTIVE     |
| lines       | DEFECTIVE     |
| stain       | DEFECTIVE     |

---

## Actual Classifier Metrics (Held-Out Test Set — 408 images)

> These are **real metrics** from `training/evaluate_classifier.py` on the held-out test split.

| Metric              | Value     |
|---------------------|-----------|
| Overall Accuracy    | **94.61%** |
| Macro Precision     | 88.16%    |
| Macro Recall        | 91.32%    |
| Macro F1-Score      | 89.37%    |
| Weighted F1-Score   | 94.61%    |

### Per-Class Performance

| Class       | Precision | Recall  | F1     | Support |
|-------------|-----------|---------|--------|---------|
| Vertical    | 82.35%    | 100.00% | 90.32% | 14      |
| defect free | 99.19%    | 98.00%  | 98.59% | 250     |
| hole        | 86.84%    | 78.57%  | 82.50% | 42      |
| horizontal  | 89.47%    | 80.95%  | 85.00% | 21      |
| lines       | 77.78%    | 95.45%  | 85.71% | 22      |
| stain       | 93.33%    | 94.92%  | 94.12% | 59      |

### Binary Quality Metrics (DEFECTIVE vs PASS)

| Metric                     | Value    |
|----------------------------|----------|
| Defect Detection Accuracy  | **98.28%** |
| Defect Recall (Sensitivity)| **98.73%** |
| Defect Precision           | 96.89%   |
| Defect F1-Score            | **97.81%** |
| False Negatives            | 2 / 158 defective images |

The model checkpoint was saved from **epoch 6** (Val Macro F1 = 0.9551).

Full results saved in `results/new_classifier/`:
- `classifier_metrics.json`
- `classifier_confusion_matrix.png`
- `classifier_training_history.png`
- `dataset_split.json`
- `sample_test_predictions.json`

---

## Technology Stack

| Layer         | Technology                                           |
|---------------|------------------------------------------------------|
| Frontend      | React 19 · Vite 8 · Tailwind CSS 3.4                 |
| Backend API   | FastAPI · uvicorn · Pydantic                         |
| Detection     | Ultralytics YOLOv8s (28 epochs) / YOLOv8n · OpenCV   |
| Classifier    | PyTorch · torchvision · MobileNetV3-Small (epoch 6)  |
| Training libs | scikit-learn · matplotlib · Pillow · NumPy           |

---

## How to Run

### Backend
```bash
pip install -r requirements.txt
python -m backend.main
```

Backend starts at `http://localhost:8000`

### Frontend
```bash
cd frontend
npm install
npm run dev
```

Frontend starts at `http://localhost:5173`

---

## API Endpoints

### `GET /api/health`
```json
{
  "status": "ok",
  "yolo_loaded": true,
  "classifier_loaded": true,
  "model_loaded": true
}
```

### `POST /api/predict` (multipart/form-data, field: `file`)

**When YOLO detects a defect (with bounding box):**
```json
{
  "quality_status": "DEFECTIVE",
  "primary_defect": "hole",
  "confidence": 0.93,
  "detection_source": "yolo",
  "detections": [
    {
      "source": "yolo",
      "class_name": "hole",
      "confidence": 0.93,
      "bbox": { "x1": 120, "y1": 80, "x2": 300, "y2": 250 }
    }
  ],
  "classification": { "source": "classifier", "class_name": "hole", "confidence": 0.88, "is_defective": true, "all_probs": {...} },
  "defects": [...],
  "defect_count": 1
}
```

**When Classifier catches a defect YOLO missed (no bounding box):**
```json
{
  "quality_status": "DEFECTIVE",
  "primary_defect": "stain",
  "confidence": 0.91,
  "detection_source": "classifier",
  "detections": [],
  "classification": { "source": "classifier", "class_name": "stain", "confidence": 0.91, "is_defective": true, "all_probs": {...} },
  "defects": [],
  "defect_count": 0
}
```

**PASS:**
```json
{
  "quality_status": "PASS",
  "primary_defect": "defect free",
  "confidence": 0.998,
  "detection_source": "none",
  "detections": [],
  "classification": { "source": "classifier", "class_name": "defect free", "confidence": 0.998, "is_defective": false },
  "defects": [],
  "defect_count": 0
}
```

---

## Training the Classifier

```bash
# Train (default 10 epochs, batch size 32)
python -m training.train_classifier

# Or specify epochs
python -m training.train_classifier 15

# Evaluate on held-out test set
python -m training.evaluate_classifier
```

---

## Project Structure

```
AI-Fabric-Defect-Detection/
├── AGENTS.md                      ← Primary AI coding assistant memory & constraints
├── docs/
│   ├── PROJECT_MEMORY.md          ← Comprehensive ground-truth technical specification
│   ├── PHONE_CAMERA_INTEGRATION_PLAN.md ← Phone camera streaming & inspection plan
│   └── project_overview.md        ← High-level architecture notes
├── backend/
│   ├── main.py                    ← FastAPI app (hybrid pipeline)
│   ├── model_service.py           ← YOLOv8 detection service (auto-loads best run)
│   ├── classifier_service.py      ← MobileNetV3 classifier service
│   ├── preprocessing.py           ← OpenCV image decode
│   ├── schemas.py                 ← Pydantic response models
│   └── models/
│       ├── best.pt                ← Fallback YOLOv8n weights
│       ├── fabric_classifier_best.pth ← Active MobileNetV3 classifier weights
│       └── classifier_classes.json    ← Class mapping
├── training/
│   ├── train_classifier.py        ← MobileNetV3 training pipeline
│   ├── evaluate_classifier.py     ← Classifier evaluation
│   ├── train.py                   ← YOLO training
│   ├── prepare_dataset.py         ← TILDA-400 dataset preparation
│   └── ...
├── results/
│   ├── new_classifier/            ← Classifier metrics, plots, splits
│   └── yolov8s_fabric_full/       ← Active retrained YOLOv8s run (weights/best.pt)
├── frontend/
│   └── src/
│       ├── App.jsx                ← Main React dashboard
│       └── index.css              ← Tailwind + custom styles
├── Fabric Defects Dataset/        ← 2,736-image classification dataset
├── photos/                        ← TILDA-400 images (400 images)
├── labels/                        ← TILDA-400 labels (400 txt files)
├── dataset/                       ← YOLO split dataset (320 train / 80 val)
└── requirements.txt
```

---

## Limitations

- The Fabric Defects Dataset classifier cannot provide bounding box localization
- YOLO model limited to TILDA-400 texture distribution
- Highly varied or novel fabrics may need additional training data
- Classifier has a fixed 0.55 confidence threshold — adjust in `classifier_service.py` if needed

---

## False Negative Strategy

| Scenario                                   | Old System | New System |
|--------------------------------------------|-----------|------------|
| YOLO detects defect                        | DEFECTIVE | DEFECTIVE  |
| YOLO misses; classifier confident (≥0.55)  | PASS ❌   | DEFECTIVE ✅ |
| YOLO misses; defect free confirmed         | PASS ✅   | PASS ✅   |
| Both uncertain                             | PASS ✅   | PASS ✅   |
