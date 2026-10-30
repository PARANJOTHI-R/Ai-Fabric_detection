"""
Hybrid Decision Arbitration Engine for Fabric Defect Inspection.
Combines YOLO localization with MobileNetV3-Small texture classification.
"""

from typing import List, Optional, Dict, Any
from .config import InspectionConfig
from .schemas import Detection, ClassificationResult


class DecisionResult:
    def __init__(
        self,
        quality_status: str,              # "PASS" | "DEFECTIVE" | "STANDBY" | "ERROR"
        primary_defect: Optional[str],
        confidence: Optional[float],
        detection_source: str,            # "yolo" | "classifier" | "both" | "none" | "system"
        total_defect_probability: Optional[float] = None,
        status_code: str = "OK",
        warning: Optional[str] = None
    ):
        self.quality_status = quality_status
        self.primary_defect = primary_defect
        self.confidence = confidence
        self.detection_source = detection_source
        self.total_defect_probability = total_defect_probability
        self.status_code = status_code
        self.warning = warning


def evaluate_hybrid_decision(
    yolo_detections: List[Detection],
    clf_result: Optional[ClassificationResult],
    is_standby: bool = False,
    standby_reason: Optional[str] = None,
    warning: Optional[str] = None,
    config: InspectionConfig = InspectionConfig
) -> DecisionResult:
    """
    Arbitrates inspection decision between YOLO detections and Classifier probabilities.

    Rules:
    1. Standby / Dark Frame:
       Returns STANDBY with 0.0 confidence, never DEFECTIVE or PASS.
    2. Model Availability Fail-Safe:
       If neither model produced valid outputs, returns ERROR. Never silently PASS.
    3. Defect Detection Priority:
       A) Valid YOLO detections exist -> DEFECTIVE (YOLO localization wins; coordinates rendered).
          If classifier also agrees fabric is defective -> source="both".
       B) YOLO found 0 detections, but Classifier indicates defective:
          - Either top predicted class is defective AND confidence >= CLASSIFIER_DEFECT_THRESHOLD (0.45)
          - OR cumulative defect probability (1.0 - P(defect free)) >= CLASSIFIER_CUMULATIVE_DEFECT_THRESHOLD (0.50)
            and top predicted class is defective.
          -> DEFECTIVE (classifier safety-net rescue; NO bounding boxes fabricated).
       C) Defect-Free Fabric:
          If neither model found defects and classifier indicates clean fabric:
          -> PASS (primary_defect is None or 'defect free', never a misleading defect name).
    """

    # 1. Standby Check
    if is_standby:
        return DecisionResult(
            quality_status="STANDBY",
            primary_defect=standby_reason or "Feed dark / Camera standby (No fabric visible)",
            confidence=0.0,
            detection_source="system",
            total_defect_probability=0.0,
            status_code="STANDBY_FEED",
            warning=warning
        )

    # 2. Fail-Safe: Both models missing or failed
    if (not yolo_detections) and (clf_result is None):
        return DecisionResult(
            quality_status="ERROR",
            primary_defect="Model outputs unavailable or inference failed",
            confidence=0.0,
            detection_source="system",
            total_defect_probability=None,
            status_code="MODELS_UNAVAILABLE",
            warning=warning or "Inference error occurred"
        )

    # Extract classifier defect probabilities if available
    p_defect_free = 0.0
    p_defect_total = 0.0
    if clf_result and clf_result.all_probs:
        p_defect_free = float(clf_result.all_probs.get("defect free", 0.0))
        p_defect_total = round(float(1.0 - p_defect_free), 4)

    # 3. Rule A: YOLO detected real defects with bounding boxes
    if yolo_detections:
        top_yolo = yolo_detections[0]
        num_detections = len(yolo_detections)
        all_probs = clf_result.all_probs if (clf_result and clf_result.all_probs) else {}

        # Default arbitration values: adhere to YOLO spatial detection authority
        primary_defect = top_yolo.class_name
        confidence = top_yolo.confidence
        source = "yolo"

        # Check semantic class concordance between YOLO and Classifier
        # Corroborating classifier classes for the top YOLO class
        sem_classes = getattr(config, "SEMANTIC_DEFECT_MAP", {}).get(top_yolo.class_name, [])
        p_sem = sum(all_probs.get(c, 0.0) for c in sem_classes)

        # Cross-model class concordance condition:
        # Both models agree on this specific defect class (either via probability mass or top predicted class)
        has_class_concordance = (clf_result is not None) and (
            p_sem >= getattr(config, "ARBITRATION_CONCORDANCE_MIN_PROB", 0.35) or
            (clf_result.class_name in sem_classes and clf_result.is_defective)
        )

        if has_class_concordance:
            # Both models corroborate the same defect class
            source = "both"
            # Keep primary_defect as top_yolo.class_name
        elif num_detections == 1:
            # Conservative ambiguity resolution: ONLY applied when attribution is unambiguous
            # (exactly one YOLO detection exists in the entire image).

            # Condition 1: YOLO 'objects' confusion -> resolve to 'oil spot'
            # (Empirical evidence: 35% of true oil spots were confused as 'objects' by YOLO)
            p_stain = all_probs.get("stain", 0.0)
            p_thread = all_probs.get("lines", 0.0) + all_probs.get("horizontal", 0.0) + all_probs.get("Vertical", 0.0)
            if (
                top_yolo.class_name == "objects"
                and top_yolo.confidence < getattr(config, "ARBITRATION_OBJECTS_MAX_YOLO_CONF", 0.50)
                and p_stain >= getattr(config, "ARBITRATION_STAIN_MIN_CLF_PROB", 0.50)
                and p_stain > top_yolo.confidence
                and p_thread < 0.25  # Guard: do NOT override to oil spot if weave/thread flaws are evident
            ):
                primary_defect = "oil spot"
                source = "both"
                confidence = round(float(p_stain), 4)
                # Preserve raw YOLO values for debugging and synchronize bounding box label
                top_yolo.raw_class_name = top_yolo.class_name
                top_yolo.raw_confidence = top_yolo.confidence
                top_yolo.class_name = "oil spot"
                top_yolo.confidence = confidence

            # Condition 2: YOLO 'thread error' confusion -> resolve to 'hole'
            # (Empirical evidence: 21.7% of true holes were confused as 'thread error' by YOLO)
            p_hole = all_probs.get("hole", 0.0)
            if (
                top_yolo.class_name == "thread error"
                and top_yolo.confidence <= getattr(config, "ARBITRATION_THREAD_ERROR_MAX_YOLO_CONF", 0.15)
                and p_hole >= getattr(config, "ARBITRATION_HOLE_MIN_CLF_PROB", 0.55)
                and p_hole >= getattr(config, "ARBITRATION_HOLE_MIN_RATIO", 2.0) * top_yolo.confidence
            ):
                primary_defect = "hole"
                source = "both"
                confidence = round(float(p_hole), 4)
                # Preserve raw YOLO values for debugging and synchronize bounding box label
                top_yolo.raw_class_name = top_yolo.class_name
                top_yolo.raw_confidence = top_yolo.confidence
                top_yolo.class_name = "hole"
                top_yolo.confidence = confidence

            # Condition 3: YOLO 'objects' confusion -> resolve to 'hole'
            # (Domain shift on patterned/striped textiles: frayed void disrupts stripes and resembles foreign surface matter)
            if (
                getattr(config, "ARBITRATION_OBJECTS_TO_HOLE_ENABLED", True)
                and top_yolo.class_name == "objects"
                and top_yolo.confidence <= getattr(config, "ARBITRATION_OBJECTS_TO_HOLE_MAX_YOLO_CONF", 0.45)
                and p_hole >= getattr(config, "ARBITRATION_OBJECTS_TO_HOLE_MIN_CLF_PROB", 0.55)
                and p_hole > top_yolo.confidence
            ):
                primary_defect = "hole"
                source = "both"
                confidence = round(float(p_hole), 4)
                # Preserve raw YOLO values for debugging and synchronize bounding box label
                top_yolo.raw_class_name = top_yolo.class_name
                top_yolo.raw_confidence = top_yolo.confidence
                top_yolo.class_name = "hole"
                top_yolo.confidence = confidence

        # Cross-model tension advisory: localized dark region flagged by YOLO but texture indicates weave anomaly
        p_thread_total = all_probs.get("lines", 0.0) + all_probs.get("horizontal", 0.0) + all_probs.get("Vertical", 0.0)
        if top_yolo.class_name == "oil spot" and top_yolo.confidence < 0.35 and p_thread_total >= 0.40:
            warning = warning or "Localized dark spot flagged by YOLO, but fabric texture indicates weave irregularity; verify visually."

        # Note: If there was no semantic concordance and no valid override, source remains "yolo".
        # We do NOT claim cross-model agreement ('both') when models only agree that *some* defect exists.
        # Original YOLO bounding boxes and coordinates in yolo_detections are 100% preserved.

        return DecisionResult(
            quality_status="DEFECTIVE",
            primary_defect=primary_defect,
            confidence=confidence,
            detection_source=source,
            total_defect_probability=p_defect_total if clf_result else top_yolo.confidence,
            status_code="OK",
            warning=warning
        )

    # 4. Rule B: YOLO found nothing, Classifier safety-net check
    if clf_result is not None:
        is_clf_defect = clf_result.is_defective
        top_conf = clf_result.confidence

        # Defect trigger condition:
        # Either the top defect class is sufficiently confident (>= CLASSIFIER_DEFECT_THRESHOLD, 0.45)
        # OR cumulative defect probability is decisive (>= 0.50) AND clean fabric probability is low (< 0.25)
        # This prevents the cumulative trap where diffuse entropy on normal fabric triggers false positives.
        defect_triggered = is_clf_defect and (
            top_conf >= config.CLASSIFIER_DEFECT_THRESHOLD or
            (p_defect_total >= config.CLASSIFIER_CUMULATIVE_DEFECT_THRESHOLD and p_defect_free < 0.25)
        )

        if defect_triggered:
            return DecisionResult(
                quality_status="DEFECTIVE",
                primary_defect=clf_result.class_name,
                confidence=top_conf,
                detection_source="classifier",
                total_defect_probability=p_defect_total,
                status_code="OK",
                warning=warning
            )

        # Defect-free check: clean fabric meets standards
        # When neither model found definitive defects and clean fabric probability is >= 0.25 (or top class is defect free)
        if (not is_clf_defect) or (p_defect_free >= 0.25):
            advisory = warning
            if is_clf_defect and top_conf < config.CLASSIFIER_DEFECT_THRESHOLD:
                advisory = advisory or "Minor texture variation detected below defect threshold; visual check recommended."
            return DecisionResult(
                quality_status="PASS",
                primary_defect=None,       # Clean PASS: never assign a defect name
                confidence=round(p_defect_free if p_defect_free > 0 else (1.0 - top_conf), 4),
                detection_source="none",
                total_defect_probability=p_defect_total,
                status_code="OK",
                warning=advisory
            )

    # 5. Default PASS fallback only when classifier explicitly confirms clean fabric
    return DecisionResult(
        quality_status="PASS",
        primary_defect=None,
        confidence=round(p_defect_free, 4) if p_defect_free > 0 else None,
        detection_source="none",
        total_defect_probability=p_defect_total,
        status_code="OK",
        warning=warning
    )
