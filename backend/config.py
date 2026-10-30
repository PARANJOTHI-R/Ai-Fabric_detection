"""
Configuration settings for Fabric Defect Detection & Quality Inspection.
Consolidates all detector, classifier, and image-quality thresholds.
"""

from typing import Dict


class InspectionConfig:
    # ── YOLO Object Detector Settings ─────────────────────────────────────────
    # Base confidence and IoU for NMS
    YOLO_BASE_CONF: float = 0.05
    YOLO_IOU_THRESHOLD: float = 0.35
    YOLO_IOM_THRESHOLD: float = 0.80  # Containment suppression: Intersection / min(Area1, Area2) > 0.80
    YOLO_IMGSZ: int = 640

    # Calibrated per-class decision thresholds (empirically validated on TILDA-400)
    # - Thread error: 0.06 (subtle weave irregularities)
    # - Hole: 0.08 (puncture & weave gaps)
    # - Objects: 0.10 (foreign particles)
    # - Oil spot: 0.15 (dark localized stains)
    YOLO_CLASS_THRESHOLDS: Dict[str, float] = {
        'thread error': 0.06,
        'hole': 0.08,
        'objects': 0.10,
        'oil spot': 0.15
    }
    YOLO_DEFAULT_THRESHOLD: float = 0.08

    # ── MobileNetV3-Small Classifier Settings ──────────────────────────────────
    # Minimum probability for a single predicted defect class to trigger DEFECTIVE
    CLASSIFIER_DEFECT_THRESHOLD: float = 0.45

    # Minimum cumulative defect probability: (1.0 - P(defect free)) >= 0.50
    # Rescues cases where defect confidence is split across multiple defect classes
    CLASSIFIER_CUMULATIVE_DEFECT_THRESHOLD: float = 0.50

    # Clean fabric confidence threshold
    CLASSIFIER_CLEAN_THRESHOLD: float = 0.50

    # ── Provisional Decision-Level Arbitration Heuristics ─────────────────────
    # Note: These cutoffs are provisional heuristics for decision-level metadata arbitration,
    # not empirically calibrated model thresholds. They are applied conservatively and only
    # when attribution is unambiguous (single YOLO detection).
    # Minimum classifier probability for concordance when classes match semantically
    ARBITRATION_CONCORDANCE_MIN_PROB: float = 0.35

    # Oil spot ambiguity resolution (YOLO 'objects' -> 'oil spot'):
    # When YOLO detects a single 'objects' flaw with conf < threshold, and Classifier has strong stain probability
    ARBITRATION_OBJECTS_MAX_YOLO_CONF: float = 0.50
    ARBITRATION_STAIN_MIN_CLF_PROB: float = 0.50

    # Hole ambiguity resolution (YOLO 'thread error' -> 'hole'):
    # When YOLO detects a single 'thread error' flaw with conf <= threshold, and Classifier has strong hole probability
    ARBITRATION_THREAD_ERROR_MAX_YOLO_CONF: float = 0.15
    ARBITRATION_HOLE_MIN_CLF_PROB: float = 0.55
    ARBITRATION_HOLE_MIN_RATIO: float = 2.0  # P(hole) must be >= 2x YOLO confidence

    # Hole ambiguity resolution (YOLO 'objects' -> 'hole'):
    # When YOLO detects a single 'objects' flaw with moderate conf, and Classifier has strong hole probability
    # Resolves domain shift on striped/patterned fabrics where frayed holes resemble foreign surface matter
    ARBITRATION_OBJECTS_TO_HOLE_ENABLED: bool = True
    ARBITRATION_OBJECTS_TO_HOLE_MAX_YOLO_CONF: float = 0.45
    ARBITRATION_OBJECTS_TO_HOLE_MIN_CLF_PROB: float = 0.55

    # Semantic cross-model defect class concordance sets
    # Maps each YOLO class to corresponding classifier classes that represent the same physical anomaly
    SEMANTIC_DEFECT_MAP: Dict[str, list] = {
        'hole': ['hole'],
        'oil spot': ['stain'],
        'thread error': ['lines', 'horizontal', 'Vertical'],
        # 'objects': has no reliable one-to-one classifier equivalent; treated as unique to YOLO
    }

    # ── Image Quality & Camera Reliability Settings ───────────────────────────
    # Standby / covered camera thresholds
    DARK_MEAN_THRESHOLD: float = 15.0
    DARK_VAR_THRESHOLD: float = 5.0

    # Image quality advisory warnings
    DIM_MEAN_THRESHOLD: float = 40.0         # Frame is dimly lit
    OVEREXPOSED_MEAN_THRESHOLD: float = 235.0 # Frame is washed out / glare
    BLUR_LAPLACIAN_THRESHOLD: float = 22.0    # Motion blur or defocus

