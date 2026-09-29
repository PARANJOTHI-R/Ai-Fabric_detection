"""
FastAPI main — Hybrid YOLO + Classifier Fabric Defect Inspection API
=====================================================================

Inference Architecture
----------------------

                    INPUT IMAGE
                         │
              ┌──────────┴──────────┐
              │                     │
         YOLO Detector        MobileNetV3-Small
         (TILDA-400)          Classifier
              │                     │
   Defect bounding boxes     Class + confidence
              │                     │
              └──────────┬──────────┘
                         │
                 Decision Logic
                         │
               ┌─────────┴─────────┐
               │                   │
        YOLO detects         YOLO finds nothing
               │                   │
           DEFECTIVE        Classifier confident
                            about defect class?
                                   │
                          ┌────────┴────────┐
                          │                 │
                     YES (≥0.55)       NO / defect free
                          │                 │
                      DEFECTIVE           PASS

Endpoints
---------
GET  /api/health  — model status
POST /api/predict — unified prediction (JSON)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .model_service import ModelService
from .classifier_service import ClassifierService
from .preprocessing import preprocess_image
from .schemas import (
    PredictionResponse, HealthResponse,
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
    yolo_service = ModelService()
    classifier_service = ClassifierService()


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


# ── Predict ────────────────────────────────────────────────────────────────────
@app.post("/api/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # ── Validate ──────────────────────────────────────────────────────────────
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

        # ── Step 3: Decision logic ────────────────────────────────────────────
        #
        # Priority:
        #   A) YOLO detected something with a bbox → DEFECTIVE (YOLO wins)
        #   B) YOLO found nothing AND classifier is ≥ threshold AND defective
        #      → DEFECTIVE (classifier catches what YOLO missed)
        #   C) Otherwise → PASS
        #
        CLASSIFIER_THRESHOLD = ClassifierService.CLASSIFIER_DEFECT_THRESHOLD

        if yolo_detections:
            # A – YOLO found real defects
            quality_status = "DEFECTIVE"
            primary_defect = yolo_detections[0].class_name
            confidence = yolo_detections[0].confidence
            detection_source = "yolo" if not clf_result else "both"

        elif (clf_result and clf_result.is_defective
              and clf_result.confidence >= CLASSIFIER_THRESHOLD):
            # B – classifier catches a missed defect
            quality_status = "DEFECTIVE"
            primary_defect = clf_result.class_name
            confidence = clf_result.confidence
            detection_source = "classifier"

        else:
            # C – PASS
            quality_status = "PASS"
            primary_defect = clf_result.class_name if clf_result else None
            confidence = clf_result.confidence if clf_result else None
            detection_source = "none" if not yolo_detections else "yolo"

        return PredictionResponse(
            quality_status=quality_status,
            primary_defect=primary_defect,
            confidence=confidence,
            detection_source=detection_source,
            detections=yolo_detections,
            classification=clf_result,
            # backward-compat aliases
            defects=yolo_detections,
            defect_count=len(yolo_detections)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
