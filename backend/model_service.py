import os
import torch
from ultralytics import YOLO
import cv2
import numpy as np

# Fix for PyTorch 2.6+ where weights_only=True is the default
_original_load = torch.load
torch.load = lambda *args, **kwargs: _original_load(*args, **{**kwargs, 'weights_only': False})

class ModelService:
    def __init__(self):
        self.model_path = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
        self.model = None
        self.classes = {
            0: 'hole',
            1: 'objects',
            2: 'oil spot',
            3: 'thread error'
        }
        self.load_model()

    def load_model(self):
        if os.path.exists(self.model_path):
            self.model = YOLO(self.model_path)
            print(f"Model loaded from {self.model_path}")
        else:
            print(f"Warning: Model not found at {self.model_path}")

    def predict(self, image: np.ndarray):
        if not self.model:
            raise Exception("Model is not loaded.")
        
        # Run inference
        results = self.model(image)[0]
        
        defects = []
        for box in results.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
            
            defects.append({
                "class_name": self.classes.get(class_id, "unknown"),
                "confidence": round(confidence, 4),
                "bbox": {
                    "x1": round(x1, 1),
                    "y1": round(y1, 1),
                    "x2": round(x2, 1),
                    "y2": round(y2, 1)
                }
            })
            
        quality_status = "DEFECTIVE" if len(defects) > 0 else "PASS"
        
        return {
            "quality_status": quality_status,
            "defects": defects,
            "defect_count": len(defects)
        }
