"""
Focused Unit Tests for Conservative Defect-Class Arbitration
============================================================
Verifies:
1. Oil-spot reconciliation (YOLO 'objects' + Classifier 'stain' >= 0.50 -> 'oil spot', source='both')
2. Hole reconciliation (YOLO 'thread error' + Classifier 'hole' >= 0.55 -> 'hole', source='both')
3. Disagreement (YOLO 'hole' + Classifier 'lines' -> 'hole', source='yolo', no false 'both')
4. Diffuse probabilities (YOLO 'objects' + diffuse classifier -> 'objects', source='yolo')
5. Multiple detections (YOLO 2 boxes + Classifier 'stain' -> no override, attribution ambiguous)
6. Exact bounding-box preservation (original detection class and coordinates never mutated)
7. Direct semantic concordance (YOLO 'oil spot' + Classifier 'stain' -> 'oil spot', source='both')
"""

import os
import sys
import unittest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.config import InspectionConfig
from backend.decision_engine import evaluate_hybrid_decision
from backend.schemas import Detection, BoundingBox, ClassificationResult


class DefectArbitrationUnitTests(unittest.TestCase):

    def test_1_oil_spot_reconciliation_from_objects(self):
        """YOLO 'objects' (conf < 0.50) + dominant Classifier 'stain' (>= 0.50) -> 'oil spot', source='both'"""
        yolo_box = BoundingBox(x1=12.5, y1=24.0, x2=55.5, y2=67.0)
        yolo_dets = [
            Detection(source="yolo", class_name="objects", confidence=0.35, bbox=yolo_box)
        ]
        clf = ClassificationResult(
            source="classifier",
            class_name="stain",
            confidence=0.85,
            is_defective=True,
            all_probs={"stain": 0.85, "defect free": 0.05, "hole": 0.04, "lines": 0.03, "horizontal": 0.02, "Vertical": 0.01}
        )

        dec = evaluate_hybrid_decision(yolo_dets, clf)

        self.assertEqual(dec.quality_status, "DEFECTIVE")
        self.assertEqual(dec.primary_defect, "oil spot", "Should reconcile ambiguous 'objects' to 'oil spot' when classifier shows dominant stain")
        self.assertEqual(dec.detection_source, "both", "Should indicate both models corroborated the flaw")
        self.assertEqual(dec.confidence, 0.85)
        # Verify YOLO detection object itself is not mutated
        self.assertEqual(yolo_dets[0].class_name, "objects", "Raw YOLO detection class must remain unaltered")
        self.assertEqual(yolo_dets[0].bbox.x1, 12.5)
        self.assertEqual(yolo_dets[0].bbox.y2, 67.0)

    def test_2_hole_reconciliation_from_thread_error(self):
        """YOLO 'thread error' (conf <= 0.15) + dominant Classifier 'hole' (>= 0.55) -> 'hole', source='both'"""
        yolo_box = BoundingBox(x1=100.0, y1=150.0, x2=130.0, y2=180.0)
        yolo_dets = [
            Detection(source="yolo", class_name="thread error", confidence=0.11, bbox=yolo_box)
        ]
        clf = ClassificationResult(
            source="classifier",
            class_name="hole",
            confidence=0.82,
            is_defective=True,
            all_probs={"hole": 0.82, "defect free": 0.06, "lines": 0.05, "stain": 0.03, "horizontal": 0.02, "Vertical": 0.02}
        )

        dec = evaluate_hybrid_decision(yolo_dets, clf)

        self.assertEqual(dec.quality_status, "DEFECTIVE")
        self.assertEqual(dec.primary_defect, "hole", "Should reconcile frayed perimeter 'thread error' to 'hole' when classifier confirms hole")
        self.assertEqual(dec.detection_source, "both")
        self.assertEqual(dec.confidence, 0.82)
        # Verify YOLO detection object itself is not mutated
        self.assertEqual(yolo_dets[0].class_name, "thread error")
        self.assertEqual(yolo_dets[0].bbox.x1, 100.0)

    def test_3_disagreement_yolo_spatial_precedence(self):
        """YOLO 'hole' (conf 0.70) + Classifier 'lines' (conf 0.65) -> primary 'hole', source='yolo' (no false 'both')"""
        yolo_box = BoundingBox(x1=50.0, y1=50.0, x2=90.0, y2=90.0)
        yolo_dets = [
            Detection(source="yolo", class_name="hole", confidence=0.70, bbox=yolo_box)
        ]
        clf = ClassificationResult(
            source="classifier",
            class_name="lines",
            confidence=0.65,
            is_defective=True,
            all_probs={"lines": 0.65, "defect free": 0.10, "hole": 0.05, "stain": 0.10, "horizontal": 0.05, "Vertical": 0.05}
        )

        dec = evaluate_hybrid_decision(yolo_dets, clf)

        self.assertEqual(dec.quality_status, "DEFECTIVE")
        self.assertEqual(dec.primary_defect, "hole", "YOLO spatial detection must take precedence when models disagree on class")
        self.assertEqual(dec.detection_source, "yolo", "Must NOT claim 'both' when models disagree on the defect class")
        self.assertEqual(dec.confidence, 0.70)

    def test_4_diffuse_probabilities_no_override(self):
        """YOLO 'objects' (conf 0.38) + diffuse classifier (no class >= 0.50) -> 'objects', source='yolo'"""
        yolo_box = BoundingBox(x1=20.0, y1=20.0, x2=40.0, y2=40.0)
        yolo_dets = [
            Detection(source="yolo", class_name="objects", confidence=0.38, bbox=yolo_box)
        ]
        clf = ClassificationResult(
            source="classifier",
            class_name="defect free",
            confidence=0.35,
            is_defective=False,
            all_probs={"defect free": 0.35, "stain": 0.25, "lines": 0.20, "hole": 0.10, "horizontal": 0.05, "Vertical": 0.05}
        )

        dec = evaluate_hybrid_decision(yolo_dets, clf)

        self.assertEqual(dec.quality_status, "DEFECTIVE")
        self.assertEqual(dec.primary_defect, "objects", "No override when classifier probabilities are diffuse")
        self.assertEqual(dec.detection_source, "yolo")
        self.assertEqual(dec.confidence, 0.38)

    def test_5_multiple_detections_no_override(self):
        """Multiple detections (e.g. 'objects' and 'thread error') -> no override; attribution is ambiguous"""
        yolo_dets = [
            Detection(source="yolo", class_name="objects", confidence=0.35, bbox=BoundingBox(x1=10.0, y1=10.0, x2=30.0, y2=30.0)),
            Detection(source="yolo", class_name="thread error", confidence=0.25, bbox=BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=150.0))
        ]
        clf = ClassificationResult(
            source="classifier",
            class_name="stain",
            confidence=0.85,
            is_defective=True,
            all_probs={"stain": 0.85, "defect free": 0.05, "hole": 0.04, "lines": 0.03, "horizontal": 0.02, "Vertical": 0.01}
        )

        dec = evaluate_hybrid_decision(yolo_dets, clf)

        self.assertEqual(dec.quality_status, "DEFECTIVE")
        self.assertEqual(dec.primary_defect, "objects", "Must NOT override when multiple detections exist; attribution is ambiguous")
        self.assertEqual(dec.detection_source, "yolo", "Must remain 'yolo' since whole-image stain cannot be attributed to a specific box")
        self.assertEqual(len(yolo_dets), 2)
        self.assertEqual(yolo_dets[0].class_name, "objects")
        self.assertEqual(yolo_dets[1].class_name, "thread error")

    def test_6_direct_semantic_concordance(self):
        """Direct concordance: YOLO 'oil spot' + Classifier 'stain' -> 'oil spot', source='both'"""
        yolo_dets = [
            Detection(source="yolo", class_name="oil spot", confidence=0.62, bbox=BoundingBox(x1=40.0, y1=40.0, x2=80.0, y2=80.0))
        ]
        clf = ClassificationResult(
            source="classifier",
            class_name="stain",
            confidence=0.78,
            is_defective=True,
            all_probs={"stain": 0.78, "defect free": 0.12, "hole": 0.03, "lines": 0.03, "horizontal": 0.02, "Vertical": 0.02}
        )

        dec = evaluate_hybrid_decision(yolo_dets, clf)

        self.assertEqual(dec.quality_status, "DEFECTIVE")
        self.assertEqual(dec.primary_defect, "oil spot")
        self.assertEqual(dec.detection_source, "both", "Semantic equivalence (oil spot <-> stain) must trigger source='both'")

    def test_7_exact_bounding_box_preservation(self):
        """Bounding box coordinates and detection dictionaries must never be altered by arbitration"""
        orig_x1, orig_y1, orig_x2, orig_y2 = 15.3, 27.8, 88.4, 99.1
        yolo_dets = [
            Detection(source="yolo", class_name="objects", confidence=0.33,
                      bbox=BoundingBox(x1=orig_x1, y1=orig_y1, x2=orig_x2, y2=orig_y2))
        ]
        clf = ClassificationResult(
            source="classifier",
            class_name="stain",
            confidence=0.88,
            is_defective=True,
            all_probs={"stain": 0.88, "defect free": 0.04, "hole": 0.02, "lines": 0.02, "horizontal": 0.02, "Vertical": 0.02}
        )

        dec = evaluate_hybrid_decision(yolo_dets, clf)

        # Confirm coordinates in yolo_dets remain bit-for-bit identical
        box = yolo_dets[0].bbox
        self.assertEqual(box.x1, orig_x1)
        self.assertEqual(box.y1, orig_y1)
        self.assertEqual(box.x2, orig_x2)
        self.assertEqual(box.y2, orig_y2)
        self.assertEqual(yolo_dets[0].class_name, "objects")


if __name__ == "__main__":
    unittest.main()
