"""
Pydantic schemas for the Fabric Defect Inspection API.

The /api/predict endpoint returns a unified PredictionResponse that combines:
  - YOLO object-detection results (detections list – includes bounding boxes)
  - MobileNetV3-Small classifier result (classification dict – NO bounding boxes)

The frontend must NOT draw bounding boxes for classifier-only results.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    """
    A YOLO detection result.  Always has a bounding box (real coordinates).
    source = "yolo"
    """
    source: str = "yolo"
    class_name: str
    confidence: float
    bbox: BoundingBox


class ClassificationResult(BaseModel):
    """
    A classifier-only result.  NO bounding box.
    source = "classifier"
    """
    source: str = "classifier"
    class_name: str
    confidence: float
    is_defective: bool
    all_probs: Optional[Dict[str, float]] = None


class PredictionResponse(BaseModel):
    quality_status: str            # "PASS" | "DEFECTIVE"
    primary_defect: Optional[str]  # best defect class name or None
    confidence: Optional[float]    # confidence of the primary defect / None
    detection_source: str          # "yolo" | "classifier" | "both" | "none"

    # YOLO detections (have bounding boxes)
    detections: List[Detection] = []

    # Classifier result (no bounding boxes)
    classification: Optional[ClassificationResult] = None

    # Legacy alias kept for backward-compat (mirrors detections)
    defects: List[Detection] = []
    defect_count: int = 0


class HealthResponse(BaseModel):
    status: str
    yolo_loaded: bool
    classifier_loaded: bool
    model_loaded: bool   # True if at least one model is loaded (backward compat)
