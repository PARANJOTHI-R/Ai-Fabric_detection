"""
ModelService — YOLO detection wrapper (original TILDA-400 pipeline).
Unchanged detection logic; class thresholds remain as calibrated.
"""

import os
import torch
from ultralytics import YOLO
import numpy as np

# Fix for PyTorch 2.6+ where weights_only=True is the default
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})


from .config import InspectionConfig


class ModelService:
    CONF_THRESHOLD = InspectionConfig.YOLO_DEFAULT_THRESHOLD
    CLASS_THRESHOLDS = InspectionConfig.YOLO_CLASS_THRESHOLDS

    def __init__(self):
        self.model_path = self._find_best_model()
        self.model = None
        self.classes = {
            0: 'hole',
            1: 'objects',
            2: 'oil spot',
            3: 'thread error'
        }
        self.load_model()

    def _find_best_model(self):
        """
        Select highest-priority detector checkpoint in fallback order:
        1. New YOLOv8s detector v1 (backend/models/yolov8s_detector_v1_best.pt)
        2. Retrained YOLOv8s model (results/yolov8s_fabric_full/weights/best.pt) if >= 20 epochs
        3. Baseline fallback model (backend/models/best.pt)
        """
        base = os.path.dirname(__file__)
        v1_detector = os.path.normpath(os.path.join(base, 'models', 'yolov8s_detector_v1_best.pt'))
        retrained = os.path.normpath(os.path.join(base, '..', 'results', 'yolov8s_fabric_full', 'weights', 'best.pt'))
        retrained_csv = os.path.normpath(os.path.join(base, '..', 'results', 'yolov8s_fabric_full', 'results.csv'))
        original = os.path.normpath(os.path.join(base, 'models', 'best.pt'))

        # Priority 1: New YOLOv8s detector v1
        if os.path.exists(v1_detector):
            print(f"[ModelService] Selected Priority 1 (New YOLOv8s Detector v1): {v1_detector}")
            return v1_detector

        # Priority 2: Retrained YOLOv8s model if training completed (>= 20 epochs logged)
        if os.path.exists(retrained) and os.path.exists(retrained_csv):
            try:
                with open(retrained_csv) as f:
                    lines = [l for l in f if l.strip()]
                if len(lines) > 20:  # header + >=20 epoch rows
                    print(f"[ModelService] Selected Priority 2 (Retrained model, {len(lines)-1} epochs): {retrained}")
                    return retrained
                else:
                    print(f"[ModelService] Retrained model exists but only {len(lines)-1} epochs, evaluating fallback")
            except Exception:
                pass

        # Priority 3: Original baseline fallback model
        print(f"[ModelService] Selected Priority 3 (Original baseline model): {original}")
        return original

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = YOLO(self.model_path)
                print(f"[ModelService] Model successfully loaded from {self.model_path}")
                print(f"[ModelService] Calibrated class thresholds: {self.CLASS_THRESHOLDS}")
                print(f"[ModelService] Model classes: {self.model.names}")
            except Exception as e:
                print(f"[ModelService] ERROR: Failed to load YOLO checkpoint '{self.model_path}': {e}")
                raise
        else:
            error_msg = f"Model checkpoint not found at: {self.model_path}"
            print(f"[ModelService] ERROR: {error_msg}")
            raise FileNotFoundError(error_msg)

    def warmup(self):
        """Pre-compiles PyTorch convolution kernels to eliminate first-request latency spikes."""
        if self.model:
            try:
                dummy = np.zeros((640, 640, 3), dtype=np.uint8)
                self.model(dummy, conf=InspectionConfig.YOLO_BASE_CONF, iou=InspectionConfig.YOLO_IOU_THRESHOLD,
                           imgsz=InspectionConfig.YOLO_IMGSZ, verbose=False)
                print("[ModelService] Warmup pass complete.")
            except Exception as e:
                print(f"[ModelService] Warmup warning: {e}")

    def predict(self, image: np.ndarray) -> list:
        """
        Returns a list of detection dicts, each with:
          class_name, confidence, bbox: {x1, y1, x2, y2}
        Returns empty list if nothing detected above threshold.
        """
        if not self.model:
            return []

        # Run inference with base conf=0.05, NMS iou=0.40, imgsz=640, and class-agnostic NMS
        # agnostic_nms=True prevents redundant overlapping candidate boxes across different classes
        results = self.model(
            image,
            conf=InspectionConfig.YOLO_BASE_CONF,
            iou=InspectionConfig.YOLO_IOU_THRESHOLD,
            imgsz=InspectionConfig.YOLO_IMGSZ,
            agnostic_nms=True,
            verbose=False
        )[0]

        # Determine image dimensions for boundary clamping
        if hasattr(image, 'shape'):
            img_h, img_w = image.shape[:2]
        elif hasattr(image, 'size'):
            img_w, img_h = image.size
        else:
            img_w, img_h = 640.0, 640.0

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = self.classes.get(class_id, "unknown")
            threshold = self.CLASS_THRESHOLDS.get(class_name, self.CONF_THRESHOLD)

            if confidence >= threshold:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                # Clamp coordinates strictly within image boundaries
                x1 = max(0.0, min(float(img_w), x1))
                y1 = max(0.0, min(float(img_h), y1))
                x2 = max(0.0, min(float(img_w), x2))
                y2 = max(0.0, min(float(img_h), y2))
                detections.append({
                    "class_name": class_name,
                    "confidence": round(confidence, 4),
                    "bbox": {
                        "x1": round(x1, 1),
                        "y1": round(y1, 1),
                        "x2": round(x2, 1),
                        "y2": round(y2, 1)
                    }
                })

        # Sort by confidence descending
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        # Containment-based duplicate suppression (IoM):
        # If a lower-confidence detection of the same class is heavily contained within
        # (or heavily contains) a higher-confidence detection (IoM >= 0.80), suppress the duplicate.
        iom_thresh = getattr(InspectionConfig, "YOLO_IOM_THRESHOLD", 0.80)
        filtered_detections = []
        for det in detections:
            suppress = False
            b1 = det["bbox"]
            area1 = (b1["x2"] - b1["x1"]) * (b1["y2"] - b1["y1"])
            for kept in filtered_detections:
                if det["class_name"] == kept["class_name"]:
                    b2 = kept["bbox"]
                    area2 = (b2["x2"] - b2["x1"]) * (b2["y2"] - b2["y1"])
                    ix1 = max(b1["x1"], b2["x1"])
                    iy1 = max(b1["y1"], b2["y1"])
                    ix2 = min(b1["x2"], b2["x2"])
                    iy2 = min(b1["y2"], b2["y2"])
                    iw = max(0.0, ix2 - ix1)
                    ih = max(0.0, iy2 - iy1)
                    inter = iw * ih
                    min_area = min(area1, area2)
                    if min_area > 0 and (inter / min_area) >= iom_thresh:
                        suppress = True
                        break
            if not suppress:
                filtered_detections.append(det)

        return filtered_detections
