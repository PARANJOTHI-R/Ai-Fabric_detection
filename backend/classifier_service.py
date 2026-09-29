"""
Fabric Defect Classifier Service
---------------------------------
Loads the MobileNetV3-Small model trained on the 'Fabric Defects Dataset'
(Vertical, defect free, hole, horizontal, lines, stain).

This service is a CLASSIFICATION-only model — it does NOT produce bounding boxes.
It is used alongside the YOLO detection model in a hybrid inference pipeline.
"""

import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image


class ClassifierService:
    """
    Hybrid-pipeline classifier. Accepts an OpenCV BGR numpy array and returns:
        {
            "class_name": str,
            "confidence": float,         # 0.0 – 1.0
            "is_defective": bool,
            "all_probs": {class_name: float, ...}
        }
    Returns None if the model is not loaded.
    """

    # Minimum softmax probability for the predicted class to be treated as
    # definitive (below this → treat as uncertain / don't override YOLO PASS).
    CLASSIFIER_DEFECT_THRESHOLD = 0.55

    def __init__(self):
        self.model = None
        self.classes = []          # ordered list  e.g. ["Vertical","defect free","hole",...]
        self.defective_classes = set()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self._load()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _find_files(self):
        base = os.path.dirname(__file__)
        weights = os.path.join(base, "models", "fabric_classifier_best.pth")
        classes_json = os.path.join(base, "models", "classifier_classes.json")
        return weights, classes_json

    def _load(self):
        weights_path, classes_path = self._find_files()

        if not os.path.exists(weights_path):
            print(f"[ClassifierService] WARNING: weights not found at {weights_path}")
            return

        if not os.path.exists(classes_path):
            print(f"[ClassifierService] WARNING: class map not found at {classes_path}")
            return

        # Load class map
        with open(classes_path, "r") as f:
            cmap = json.load(f)
        self.classes = cmap["classes"]
        self.defective_classes = set(cmap.get("defective_classes", []))

        num_classes = len(self.classes)

        # Rebuild the exact architecture used during training
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

        checkpoint = torch.load(weights_path, map_location=self.device,
                                weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        self.model = model
        epoch = checkpoint.get("epoch", "?")
        val_f1 = checkpoint.get("val_f1", 0.0)
        val_acc = checkpoint.get("val_acc", 0.0)
        print(f"[ClassifierService] Loaded MobileNetV3-Small "
              f"(epoch {epoch}, Val F1={val_f1:.4f}, Val Acc={val_acc:.4f})")
        print(f"[ClassifierService] Classes: {self.classes}")
        print(f"[ClassifierService] Defective classes: {self.defective_classes}")
        print(f"[ClassifierService] Defect threshold: {self.CLASSIFIER_DEFECT_THRESHOLD}")

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def is_loaded(self):
        return self.model is not None

    def predict(self, bgr_array: np.ndarray) -> dict | None:
        """
        Parameters
        ----------
        bgr_array : np.ndarray
            OpenCV image (BGR, uint8).

        Returns
        -------
        dict with keys: class_name, confidence, is_defective, all_probs
        """
        if not self.is_loaded:
            return None

        # Convert BGR → RGB PIL Image
        rgb = bgr_array[:, :, ::-1]  # flip channels
        pil_img = Image.fromarray(rgb.astype(np.uint8))

        tensor = self.transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()

        pred_idx = int(np.argmax(probs))
        pred_class = self.classes[pred_idx]
        confidence = float(probs[pred_idx])
        is_defective = pred_class in self.defective_classes

        all_probs = {cls: round(float(p), 4)
                     for cls, p in zip(self.classes, probs)}

        return {
            "class_name": pred_class,
            "confidence": round(confidence, 4),
            "is_defective": is_defective,
            "all_probs": all_probs
        }
