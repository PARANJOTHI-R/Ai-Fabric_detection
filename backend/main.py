"""
FastAPI main — Hybrid YOLO + Classifier Fabric Defect Inspection API
=====================================================================

Inference Architecture
----------------------

                    INPUT IMAGE
                         │
               ┌─────────┴─────────┐
               │                   │
         [BGR Array]          [BGR Array]
               │                   │
         YOLO Detector      MobileNetV3-Small
         (TILDA-400)        Classifier
               │                   │
      Defect bounding      Class probabilities
           boxes          + defect cumulative prob
               │                   │
               └─────────┬─────────┘
                         │
                 Decision Engine
            (backend/decision_engine.py)
                         │
               ┌─────────┴─────────┐
               │                   │
          DEFECTIVE              PASS
      (YOLO bboxes or         (Standard
      classifier rescue)      approved)
"""

import time
import datetime
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .config import InspectionConfig
from .model_service import ModelService
from .classifier_service import ClassifierService
from .preprocessing import preprocess_image
from .image_quality import assess_image_quality
from .decision_engine import evaluate_hybrid_decision
from .schemas import (
    PredictionResponse, HealthResponse, CameraFrameResponse,
    Detection, BoundingBox, ClassificationResult
)
import uvicorn

app = FastAPI(title="AI Fabric Inspection API — Hybrid Pipeline")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Model singletons (loaded once at startup) ─────────────────────────────────
yolo_service: ModelService | None = None
classifier_service: ClassifierService | None = None


@app.on_event("startup")
def startup_event():
    global yolo_service, classifier_service
    print("[Startup] Initializing inspection models...")
    yolo_service = ModelService()
    classifier_service = ClassifierService()

    # Pre-warm models to eliminate the first-inference CPU latency spike (~3.0s -> ~0.2s)
    if yolo_service:
        yolo_service.warmup()
    if classifier_service:
        classifier_service.warmup()
    print("[Startup] All models loaded and warmed up successfully.")


# ── Health ─────────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=HealthResponse)
def health_check():
    yolo_ok = yolo_service is not None and yolo_service.model is not None
    clf_ok = classifier_service is not None and classifier_service.is_loaded
    return {
        "status": "ok",
        "yolo_loaded": yolo_ok,
        "classifier_loaded": clf_ok,
        "model_loaded": yolo_ok or clf_ok   # backward-compat field
    }


# ── Predict (Static Fabric Image Upload) ───────────────────────────────────────
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    start_time = time.perf_counter()

    # ── Validate Input ────────────────────────────────────────────────────────
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image.")

    if (yolo_service is None or yolo_service.model is None) and \
       (classifier_service is None or not classifier_service.is_loaded):
        raise HTTPException(status_code=503, detail="No models loaded. Start the server properly.")

    try:
        contents = await file.read()
        image = preprocess_image(contents)
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid or corrupted image file.")

        # ── Frame Quality Assessment ──────────────────────────────────────────
        iq = assess_image_quality(image)
        if iq.is_standby:
            proc_ms = round((time.perf_counter() - start_time) * 1000, 1)
            return PredictionResponse(
                quality_status="STANDBY",
                primary_defect=iq.warning or "Feed dark / Blank image (No fabric visible)",
                confidence=0.0,
                detection_source="system",
                detections=[],
                classification=None,
                defects=[],
                defect_count=0,
                processing_time_ms=proc_ms,
                total_defect_probability=0.0,
                warning=iq.warning,
                status_code="STANDBY_FEED"
            )

        # ── Step 1: YOLO detection (with bounding boxes) ──────────────────────
        yolo_detections_raw = []
        if yolo_service and yolo_service.model:
            yolo_detections_raw = yolo_service.predict(image)

        yolo_detections = [
            Detection(
                source="yolo",
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=BoundingBox(**d["bbox"])
            )
            for d in yolo_detections_raw
        ]

        # ── Step 2: Classifier (no bounding boxes) ───────────────────────────
        clf_result = None
        if classifier_service and classifier_service.is_loaded:
            raw = classifier_service.predict(image)
            if raw:
                clf_result = ClassificationResult(
                    source="classifier",
                    class_name=raw["class_name"],
                    confidence=raw["confidence"],
                    is_defective=raw["is_defective"],
                    all_probs=raw["all_probs"]
                )

        # ── Step 3: Transparent Hybrid Decision Logic ─────────────────────────
        decision = evaluate_hybrid_decision(
            yolo_detections=yolo_detections,
            clf_result=clf_result,
            is_standby=False,
            warning=iq.warning,
            config=InspectionConfig
        )

        proc_ms = round((time.perf_counter() - start_time) * 1000, 1)

        return PredictionResponse(
            quality_status=decision.quality_status,
            primary_defect=decision.primary_defect,
            confidence=decision.confidence,
            detection_source=decision.detection_source,
            detections=yolo_detections,
            classification=clf_result,
            defects=yolo_detections,
            defect_count=len(yolo_detections),
            processing_time_ms=proc_ms,
            total_defect_probability=decision.total_defect_probability,
            warning=decision.warning,
            status_code=decision.status_code
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


# ── Camera Frame Predict (Live Phone / Webcam feed) ───────────────────────────
@app.post("/api/camera-frame", response_model=CameraFrameResponse)
async def camera_frame(file: UploadFile = File(...)):
    start_time = time.perf_counter()
    iso_time = datetime.datetime.now().isoformat()

    if file.content_type and not file.content_type.startswith("image/"):
        return CameraFrameResponse(
            quality_status="ERROR",
            primary_defect="Uploaded payload is not an image frame",
            confidence=0.0,
            detection_source="system",
            processing_time_ms=0.0,
            status_code="INVALID_PAYLOAD",
            timestamp=iso_time
        )

    if (yolo_service is None or yolo_service.model is None) and \
       (classifier_service is None or not classifier_service.is_loaded):
        return CameraFrameResponse(
            quality_status="ERROR",
            primary_defect="AI models not loaded on server",
            confidence=0.0,
            detection_source="system",
            processing_time_ms=0.0,
            status_code="MODELS_NOT_LOADED",
            timestamp=iso_time
        )

    try:
        contents = await file.read()
        image = preprocess_image(contents)
        if image is None:
            proc_ms = round((time.perf_counter() - start_time) * 1000, 1)
            return CameraFrameResponse(
                quality_status="ERROR",
                primary_defect="Corrupted or unreadable image frame",
                confidence=0.0,
                detection_source="system",
                processing_time_ms=proc_ms,
                status_code="INVALID_FRAME",
                timestamp=iso_time
            )

        # ── Frame Quality / Darkness Pre-check ────────────────────────────────
        iq = assess_image_quality(image)
        if iq.is_standby:
            proc_ms = round((time.perf_counter() - start_time) * 1000, 1)
            return CameraFrameResponse(
                quality_status="STANDBY",
                primary_defect=iq.warning or "Feed dark / Camera standby (No fabric visible)",
                confidence=0.0,
                detection_source="system",
                detections=[],
                classification=None,
                defects=[],
                defect_count=0,
                processing_time_ms=proc_ms,
                mean_brightness=round(iq.mean_brightness, 1),
                total_defect_probability=0.0,
                warning=iq.warning,
                status_code="STANDBY_FEED",
                timestamp=iso_time
            )

        # ── Step 1: YOLO detection (with bounding boxes) ──────────────────────
        yolo_detections_raw = []
        if yolo_service and yolo_service.model:
            yolo_detections_raw = yolo_service.predict(image)

        yolo_detections = [
            Detection(
                source="yolo",
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=BoundingBox(**d["bbox"])
            )
            for d in yolo_detections_raw
        ]

        # ── Step 2: Classifier (no bounding boxes) ───────────────────────────
        clf_result = None
        if classifier_service and classifier_service.is_loaded:
            raw = classifier_service.predict(image)
            if raw:
                clf_result = ClassificationResult(
                    source="classifier",
                    class_name=raw["class_name"],
                    confidence=raw["confidence"],
                    is_defective=raw["is_defective"],
                    all_probs=raw["all_probs"]
                )

        # ── Step 3: Transparent Decision Logic ────────────────────────────────
        decision = evaluate_hybrid_decision(
            yolo_detections=yolo_detections,
            clf_result=clf_result,
            is_standby=False,
            warning=iq.warning,
            config=InspectionConfig
        )

        proc_ms = round((time.perf_counter() - start_time) * 1000, 1)

        return CameraFrameResponse(
            quality_status=decision.quality_status,
            primary_defect=decision.primary_defect,
            confidence=decision.confidence,
            detection_source=decision.detection_source,
            detections=yolo_detections,
            classification=clf_result,
            defects=yolo_detections,
            defect_count=len(yolo_detections),
            processing_time_ms=proc_ms,
            mean_brightness=round(iq.mean_brightness, 1),
            total_defect_probability=decision.total_defect_probability,
            warning=decision.warning,
            status_code=decision.status_code,
            timestamp=iso_time
        )

    except Exception as e:
        proc_ms = round((time.perf_counter() - start_time) * 1000, 1)
        return CameraFrameResponse(
            quality_status="ERROR",
            primary_defect=f"Inference error: {str(e)}",
            confidence=0.0,
            detection_source="system",
            processing_time_ms=proc_ms,
            status_code="INFERENCE_ERROR",
            timestamp=iso_time
        )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
