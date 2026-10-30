# AGENTS.md — Persistent Developer and Agent Memory

> **IMPORTANT REQUIREMENT FOR ALL AI AGENTS & SESSIONS:**  
> Before proposing or implementing any modifications to the AI models, backend inference pipeline, decision thresholds, or frontend architecture, you **MUST** read [docs/PROJECT_MEMORY.md](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/docs/PROJECT_MEMORY.md) and [docs/PHONE_CAMERA_INTEGRATION_PLAN.md](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/docs/PHONE_CAMERA_INTEGRATION_PLAN.md).

---

## 1. Project Purpose & Architecture

This project is an **AI-Based Fabric Defect Detection and Quality Inspection System** using computer vision. It is engineered with a **Hybrid AI Pipeline** to eliminate false negatives while preventing fabricated bounding boxes:

1. **Localizer (YOLOv8)**: Trained on the TILDA-400 dataset (~400 images). Detects defect bounding boxes for 4 classes: `hole`, `objects`, `oil spot`, `thread error`.
2. **Classifier (MobileNetV3-Small)**: Trained on the Fabric Defects Dataset (~2,700 images). Classifies whole-image fabric condition across 6 classes: `Vertical`, `defect free`, `hole`, `horizontal`, `lines`, `stain`. Does **not** produce bounding boxes.
3. **Decision Logic**:
   - **YOLO detections exist** $\rightarrow$ **`DEFECTIVE`** (YOLO localization wins; bounding boxes rendered).
   - **YOLO finds nothing**, but **Classifier is defective AND confidence $\ge 0.55$** $\rightarrow$ **`DEFECTIVE`** (Rescues false negatives; no bounding box).
   - **Otherwise** $\rightarrow$ **`PASS`** (Quality standards met).
4. **Presentation**: Light-themed React dashboard displaying quality status, primary defect, confidence score, bounding-box overlays, and inspection history.

---

## 2. Verified Technology Stack

| Layer | Verified Technology | Actual Version in Codebase |
|---|---|---|
| **OS Environment** | Windows (PowerShell) | Windows 11, DirectShow/MediaFoundation camera backends |
| **Backend Framework** | FastAPI + Uvicorn | `fastapi`, `uvicorn`, `python-multipart`, `pydantic` |
| **Python Environment** | Python 3.11.0 (Windows) | PyTorch `2.14.1+cpu` (CUDA unavailable / CPU inference) |
| **Object Detection** | Ultralytics YOLO | `ultralytics==8.2.0`, YOLOv8s (retrained, 28 epochs) |
| **Image Classifier** | PyTorch + Torchvision | `torchvision` MobileNetV3-Small (trained 6 epochs) |
| **Image Processing** | OpenCV + Pillow + NumPy | `opencv-python-headless`, `pillow`, `numpy` |
| **Frontend Framework** | React + Vite | React `19.2.8`, Vite `8.3.0` (ES modules) |
| **Styling & UI** | Tailwind CSS + Lucide Icons | `tailwindcss 3.4.13`, `lucide-react 1.48.0` |
| **HTTP Client** | Axios | `axios 1.20.0` |

---

## 3. Important Source Files & Responsibilities

- [`backend/main.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/backend/main.py): FastAPI server, CORS middleware, startup model warmup loaders, `/api/health`, `/api/predict`, and `/api/camera-frame` endpoints.
- [`backend/config.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/backend/config.py): Centralized configuration for all detector, classifier, cumulative probability, and image quality thresholds.
- [`backend/decision_engine.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/backend/decision_engine.py): Transparent hybrid decision arbitration engine. Uses cumulative defect probability rescue ($P(\text{defect total}) \ge 0.50$ or top defect $\ge 0.45$) to eliminate threshold traps while preventing fabricated bounding boxes.
- [`backend/image_quality.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/backend/image_quality.py): Frame illumination, variance, and motion blur (Laplacian) assessment. Prevents dark-frame false positives and reports quality warnings.
- [`backend/model_service.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/backend/model_service.py): YOLO wrapper with explicit imgsz=640 and startup `warmup()` pass. Loads retrained YOLOv8s weights (`results/yolov8s_fabric_full/weights/best.pt`) with fallback to `backend/models/best.pt`.
- [`backend/classifier_service.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/backend/classifier_service.py): PyTorch MobileNetV3-Small classifier wrapper with `warmup()` and total defect probability calculation.
- [`backend/schemas.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/backend/schemas.py): Pydantic models: `BoundingBox`, `Detection`, `ClassificationResult`, `PredictionResponse`, `CameraFrameResponse`, `HealthResponse`.
- [`training/test_inspection_suite.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/training/test_inspection_suite.py): 10-scenario automated inspection test suite.
- [`training/evaluate.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/training/evaluate.py): YOLO validation evaluation with per-class AP/mAP metrics.
- [`training/evaluate_classifier.py`](file:///d:/AI-Fabric-Defect-Detection/AI-Fabric-Defect-Detection/training/evaluate_classifier.py): MobileNetV3-Small evaluation on 408 held-out test samples.

---

## 4. How Frontend & Backend Communicate

1. **Protocol**: HTTP REST.
2. **Health Check**: `GET http://localhost:8000/api/health` returns `{ "status": "ok", "yolo_loaded": bool, "classifier_loaded": bool, "model_loaded": bool }`.
3. **Static Image Inference**: `POST http://localhost:8000/api/predict` with `multipart/form-data` payload containing field `file`. Returns `PredictionResponse` with `quality_status`, `primary_defect`, `confidence`, `detections`, `classification`, `processing_time_ms`, `total_defect_probability`, `warning`, `status_code`.
4. **Live Camera Frame Inference**: `POST http://localhost:8000/api/camera-frame` with `multipart/form-data` payload containing field `file`. Includes pre-inference dark-frame sanitization (returns `STANDBY` if mean brightness $<15$), latency telemetry, and explicit `CameraFrameResponse` (`quality_status: PASS | DEFECTIVE | STANDBY | ERROR`, `processing_time_ms`, `mean_brightness`, `status_code`, `warning`).
5. **CORS & Vite Proxy**: Backend enables `allow_origins=["*"]`. Vite frontend dev server proxies `/api` to `http://localhost:8000` and binds with `host: true` for mobile phone browser access.

---

## 5. How to Start and Test the Project

### Start Backend
```powershell
# From repository root
python -m backend.main
# Server starts at http://localhost:8000
```

### Start Frontend
```powershell
cd frontend
npm run dev
# Vite dev server starts at http://localhost:5173
```

### Run Automated Sanity & Evaluation Tests
```powershell
# Run the 10-scenario comprehensive test suite
python training/test_inspection_suite.py

# Run FastAPI endpoint integration test
python training/test_api_endpoints.py

# Run YOLO detector evaluation on validation set
python training/evaluate.py

# Run Classifier evaluation on held-out test set
python training/evaluate_classifier.py
```

---

## 6. Model & Dataset Locations

- **YOLO Dataset (TILDA-400)**: `photos/` (400 images), `labels/` (400 txt files), `dataset/` (YOLO train/val split: 320 train / 80 val).
- **Classifier Dataset**: `Fabric Defects Dataset/` (6 subfolders: `Vertical` (98), `defect free` (1666), `hole` (281), `horizontal` (136), `lines` (157), `stain` (398)).
- **Primary Active YOLO Weights**: `results/yolov8s_fabric_full/weights/best.pt` (89.5 MB, 28 epochs).
- **Fallback YOLO Weights**: `backend/models/best.pt` (6.2 MB, YOLOv8n).
- **Active Classifier Weights**: `backend/models/fabric_classifier_best.pth` (6.2 MB, MobileNetV3-Small, epoch 6).
- **Classifier Evaluation Outputs**: `results/new_classifier/` (`classifier_metrics.json`, `classifier_confusion_matrix.png`, `classifier_training_history.png`, `dataset_split.json`).

---

## 7. Strict Rules for Making Changes Safely

1. **Preserve the Hybrid Architecture**: Never remove either model without explicit user direction. Both serve complementary purposes (localization vs. broad-defect classification).
2. **Never Fabricate Bounding Boxes**: The classifier has no localization capability. Do not invent or approximate synthetic bounding boxes for classifier-only results.
3. **Do Not Alter Decision Thresholds**: Keep `CONF_THRESHOLD = 0.08`, YOLO per-class thresholds (`0.06`, `0.08`, `0.10`, `0.15`), and classifier threshold `0.55` intact unless instructed.
4. **Preserve Light-Themed Clean UI**: The existing dashboard uses slate/neutral light aesthetics (`#F8FAFC`, `#FFFFFF`, `#2563EB`, `#16A34A`, `#DC2626`). Do not break Tailwind token structure or replace it with unstyled components.
5. **Fail-Safe Video & Camera States**: If a camera frame cannot be decoded, is dark/standby, or inference fails, **NEVER default to PASS**. Report `ERROR` or `STANDBY`.
6. **No Heavy Cloud or External Dependencies**: Keep inference 100% local on the PC. Do not introduce remote databases, cloud APIs, Docker, or complex brokers for local student hardware prototypes.
