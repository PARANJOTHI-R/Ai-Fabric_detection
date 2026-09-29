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


class ModelService:
    # Calibrated confidence thresholds empirically validated on TILDA-400:
    # - Thread error: 0.06 (subtle yarn deviations score ~0.06 - 0.20)
    # - Hole: 0.08 (puncture & weave gaps score ~0.08 - 0.70)
    # - Objects: 0.10 (foreign matter scores ~0.10 - 0.60)
    # - Oil spot: 0.15 (stains score ~0.20 - 0.98)
    CONF_THRESHOLD = 0.08
    CLASS_THRESHOLDS = {
        'thread error': 0.06,
        'hole': 0.08,
        'objects': 0.10,
        'oil spot': 0.15
    }

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
        """Prefer the retrained yolov8s model if training is complete, else use original."""
        base = os.path.dirname(__file__)
        retrained = os.path.join(base, '..', 'results', 'yolov8s_fabric_full', 'weights', 'best.pt')
        retrained_csv = os.path.join(base, '..', 'results', 'yolov8s_fabric_full', 'results.csv')
        original = os.path.join(base, 'models', 'best.pt')
        retrained = os.path.normpath(retrained)
        retrained_csv = os.path.normpath(retrained_csv)
        # Only use retrained model if training completed (>= 20 epochs logged)
        if os.path.exists(retrained) and os.path.exists(retrained_csv):
            try:
                with open(retrained_csv) as f:
                    lines = [l for l in f if l.strip()]
                if len(lines) > 20:  # header + >=20 epoch rows
                    print(f"[ModelService] Using retrained model ({len(lines)-1} epochs): {retrained}")
                    return retrained
                else:
                    print(f"[ModelService] Retrained model exists but only {len(lines)-1} epochs, using original")
            except Exception:
                pass
        print(f"[ModelService] Using original model: {original}")
        return original

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            print(f"[ModelService] Model loaded from {self.model_path}")
            print(f"[ModelService] Calibrated class thresholds: {self.CLASS_THRESHOLDS}")
            print(f"[ModelService] Model classes: {self.model.names}")
        else:
            print(f"[ModelService] Warning: Model not found at {self.model_path}")

    def predict(self, image: np.ndarray) -> list:
        """
        Returns a list of detection dicts, each with:
          class_name, confidence, bbox: {x1, y1, x2, y2}
        Returns empty list if nothing detected above threshold.
        """
        if not self.model:
            return []

        # Run inference with base conf=0.05 and NMS iou=0.40
        results = self.model(image, conf=0.05, iou=0.40, verbose=False)[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = self.classes.get(class_id, "unknown")
            threshold = self.CLASS_THRESHOLDS.get(class_name, self.CONF_THRESHOLD)

            if confidence >= threshold:
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
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
        return detections
