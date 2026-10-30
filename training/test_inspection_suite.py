"""
Comprehensive Test Suite for AI Fabric Defect Detection Pipeline
==================================================================
Covers the 10 mandatory evaluation scenarios:
 1. Defect-free fabric
 2. Fabric with a visible hole
 3. Fabric with a stain or oil spot
 4. Line, horizontal, and vertical defects
 5. A defect that YOLO misses but the classifier recognizes (Rescue test)
 6. A defect that the classifier misses but YOLO recognizes
 7. A dark or black camera frame (Standby test)
 8. Invalid image input (Corrupted buffer)
 9. Model inference failure (Fail-safe test)
 10. Camera disconnect or interrupted request simulation
"""

import os
import sys
import glob
import cv2
import numpy as np
from PIL import Image

# Ensure repository root is in python path
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.config import InspectionConfig
from backend.model_service import ModelService
from backend.classifier_service import ClassifierService
from backend.image_quality import assess_image_quality
from backend.decision_engine import evaluate_hybrid_decision, DecisionResult
from backend.schemas import Detection, BoundingBox, ClassificationResult


class InspectionTestSuite:
    def __init__(self):
        print("=" * 70)
        print("INITIALIZING INSPECTION TEST SUITE")
        print("=" * 70)
        self.ms = ModelService()
        self.cs = ClassifierService()
        self.ms.warmup()
        self.cs.warmup()
        self.passed_tests = []
        self.failed_tests = []

    def record_result(self, name: str, success: bool, details: str):
        if success:
            self.passed_tests.append(name)
            print(f"[PASS] {name}: {details}")
        else:
            self.failed_tests.append((name, details))
            print(f"[FAIL] {name}: {details}")

    # ── Test 1: Defect-free fabric ─────────────────────────────────────────────
    def test_1_defect_free_fabric(self):
        print("\n--- Test 1: Defect-free fabric ---")
        df_folder = os.path.join(REPO_ROOT, "Fabric Defects Dataset", "defect free")
        candidates = sorted(glob.glob(os.path.join(df_folder, "*.*")))
        if not candidates:
            self.record_result("Test 1 (Defect-Free)", False, "No defect-free images found")
            return

        img = cv2.imread(candidates[0])
        yolo_raw = self.ms.predict(img)
        clf = self.cs.predict(img)

        yolo_dets = [Detection(source="yolo", class_name=d["class_name"], confidence=d["confidence"], bbox=BoundingBox(**d["bbox"])) for d in yolo_raw]
        clf_result = ClassificationResult(**clf) if clf else None

        dec = evaluate_hybrid_decision(yolo_dets, clf_result)
        success = (dec.quality_status == "PASS") and (dec.primary_defect is None)
        details = f"Status={dec.quality_status}, PrimaryDefect={dec.primary_defect}, Conf={dec.confidence}, P(df)={clf['all_probs'].get('defect free', 0.0) if clf else 0}"
        self.record_result("Test 1 (Defect-Free Fabric)", success, details)

    # ── Test 2: Fabric with a visible hole ─────────────────────────────────────
    def test_2_visible_hole(self):
        print("\n--- Test 2: Fabric with a visible hole ---")
        hole_candidates = glob.glob(os.path.join(REPO_ROOT, "photos", "*.*"))
        # Find one annotated as hole (class 0)
        hole_img_path = None
        for p in hole_candidates:
            stem = os.path.splitext(os.path.basename(p))[0]
            lbl = os.path.join(REPO_ROOT, "labels", stem + ".txt")
            if os.path.exists(lbl):
                with open(lbl) as f:
                    if any(line.startswith("0 ") for line in f):
                        hole_img_path = p
                        break

        if not hole_img_path:
            # Fallback to Fabric Defects Dataset
            fallback = glob.glob(os.path.join(REPO_ROOT, "Fabric Defects Dataset", "hole", "*.*"))
            if fallback:
                hole_img_path = fallback[0]

        img = cv2.imread(hole_img_path)
        yolo_raw = self.ms.predict(img)
        clf = self.cs.predict(img)

        yolo_dets = [Detection(source="yolo", class_name=d["class_name"], confidence=d["confidence"], bbox=BoundingBox(**d["bbox"])) for d in yolo_raw]
        clf_result = ClassificationResult(**clf) if clf else None

        dec = evaluate_hybrid_decision(yolo_dets, clf_result)
        success = (dec.quality_status == "DEFECTIVE") and ("hole" in (dec.primary_defect or "").lower() or len(yolo_dets) > 0)
        details = f"Status={dec.quality_status}, PrimaryDefect={dec.primary_defect}, Source={dec.detection_source}, YOLO_boxes={len(yolo_dets)}"
        self.record_result("Test 2 (Visible Hole)", success, details)

    # ── Test 3: Fabric with stain or oil spot ──────────────────────────────────
    def test_3_stain_or_oil_spot(self):
        print("\n--- Test 3: Fabric with stain or oil spot ---")
        oil_candidates = glob.glob(os.path.join(REPO_ROOT, "photos", "*.*"))
        oil_img_path = None
        for p in oil_candidates:
            stem = os.path.splitext(os.path.basename(p))[0]
            lbl = os.path.join(REPO_ROOT, "labels", stem + ".txt")
            if os.path.exists(lbl):
                with open(lbl) as f:
                    if any(line.startswith("2 ") for line in f):
                        oil_img_path = p
                        break

        if not oil_img_path:
            fallback = glob.glob(os.path.join(REPO_ROOT, "Fabric Defects Dataset", "stain", "*.*"))
            oil_img_path = fallback[0] if fallback else None

        img = cv2.imread(oil_img_path)
        yolo_raw = self.ms.predict(img)
        clf = self.cs.predict(img)

        yolo_dets = [Detection(source="yolo", class_name=d["class_name"], confidence=d["confidence"], bbox=BoundingBox(**d["bbox"])) for d in yolo_raw]
        clf_result = ClassificationResult(**clf) if clf else None

        dec = evaluate_hybrid_decision(yolo_dets, clf_result)
        success = (dec.quality_status == "DEFECTIVE")
        details = f"Status={dec.quality_status}, Defect={dec.primary_defect}, Source={dec.detection_source}, Conf={dec.confidence}"
        self.record_result("Test 3 (Stain / Oil Spot)", success, details)

    # ── Test 4: Line, horizontal, and vertical defects ─────────────────────────
    def test_4_line_horizontal_vertical(self):
        print("\n--- Test 4: Line, horizontal, and vertical defects ---")
        for defect_type in ["lines", "horizontal", "Vertical"]:
            folder = os.path.join(REPO_ROOT, "Fabric Defects Dataset", defect_type)
            candidates = glob.glob(os.path.join(folder, "*.*"))
            if not candidates:
                self.record_result(f"Test 4 ({defect_type})", False, "No samples found")
                continue

            img = cv2.imread(candidates[0])
            yolo_raw = self.ms.predict(img)
            clf = self.cs.predict(img)

            yolo_dets = [Detection(source="yolo", class_name=d["class_name"], confidence=d["confidence"], bbox=BoundingBox(**d["bbox"])) for d in yolo_raw]
            clf_result = ClassificationResult(**clf) if clf else None

            dec = evaluate_hybrid_decision(yolo_dets, clf_result)
            success = (dec.quality_status == "DEFECTIVE")
            details = f"Type={defect_type}, Status={dec.quality_status}, Defect={dec.primary_defect}, Conf={dec.confidence}, Source={dec.detection_source}"
            self.record_result(f"Test 4 ({defect_type} Defect)", success, details)

    # ── Test 5: Defect that YOLO misses but classifier rescues ─────────────────
    def test_5_yolo_misses_classifier_rescues(self):
        print("\n--- Test 5: YOLO misses defect, Classifier rescues ---")
        # 20180531_140818.jpg from Fabric defects Images is a real image where YOLO detects nothing but classifier finds 'lines'
        target_img = os.path.join(REPO_ROOT, "Fabric defects Images", "20180531_140818.jpg")
        if not os.path.exists(target_img):
            # Fallback to an image from lines folder
            target_img = glob.glob(os.path.join(REPO_ROOT, "Fabric Defects Dataset", "lines", "*.*"))[0]

        img = cv2.imread(target_img)
        yolo_raw = self.ms.predict(img)
        clf = self.cs.predict(img)

        yolo_dets = [Detection(source="yolo", class_name=d["class_name"], confidence=d["confidence"], bbox=BoundingBox(**d["bbox"])) for d in yolo_raw]
        clf_result = ClassificationResult(**clf) if clf else None

        dec = evaluate_hybrid_decision(yolo_dets, clf_result)
        # Condition: YOLO missed it (0 bboxes), but hybrid decision marks DEFECTIVE via classifier
        success = (len(yolo_dets) == 0) and (dec.quality_status == "DEFECTIVE") and (dec.detection_source == "classifier")
        details = f"YOLO_boxes={len(yolo_dets)}, Status={dec.quality_status}, Defect={dec.primary_defect}, Source={dec.detection_source}, Conf={dec.confidence}"
        self.record_result("Test 5 (Classifier Rescue / YOLO Missed)", success, details)

    # ── Test 6: Defect that classifier misses but YOLO recognizes ──────────────
    def test_6_classifier_misses_yolo_recognizes(self):
        print("\n--- Test 6: Classifier under-scores, YOLO defect detection overrides ---")
        # We simulate a case where classifier outputs 'defect free' but YOLO found a hole with confidence 0.75
        mock_yolo_dets = [
            Detection(
                source="yolo",
                class_name="hole",
                confidence=0.75,
                bbox=BoundingBox(x1=100.0, y1=150.0, x2=140.0, y2=190.0)
            )
        ]
        mock_clf = ClassificationResult(
            source="classifier",
            class_name="defect free",
            confidence=0.92,
            is_defective=False,
            all_probs={"defect free": 0.92, "hole": 0.03, "lines": 0.02, "stain": 0.01, "horizontal": 0.01, "Vertical": 0.01}
        )

        dec = evaluate_hybrid_decision(mock_yolo_dets, mock_clf)
        # Condition: YOLO detection MUST override classifier's 'defect free' result
        success = (dec.quality_status == "DEFECTIVE") and (dec.primary_defect == "hole") and (len(mock_yolo_dets) == 1)
        details = f"Status={dec.quality_status}, Defect={dec.primary_defect}, Source={dec.detection_source}, BBox={mock_yolo_dets[0].bbox.model_dump()}"
        self.record_result("Test 6 (YOLO Overrides Classifier Defect-Free)", success, details)

    # ── Test 7: Dark or black camera frame ─────────────────────────────────────
    def test_7_dark_camera_frame(self):
        print("\n--- Test 7: Dark or black camera frame ---")
        # All-black frame (e.g. camera lens covered or disconnected feed)
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        iq = assess_image_quality(black_frame)

        dec = evaluate_hybrid_decision(
            yolo_detections=[],
            clf_result=None,
            is_standby=iq.is_standby,
            standby_reason=iq.warning
        )

        # Condition: MUST return STANDBY with 0.0 confidence, NEVER DEFECTIVE and NEVER PASS
        success = (dec.quality_status == "STANDBY") and (dec.confidence == 0.0) and (iq.is_standby is True)
        details = f"Status={dec.quality_status}, Status_code={dec.status_code}, Brightness={iq.mean_brightness}, Warning={iq.warning}"
        self.record_result("Test 7 (Dark / Standby Frame)", success, details)

    # ── Test 8: Invalid image input ───────────────────────────────────────────
    def test_8_invalid_image_input(self):
        print("\n--- Test 8: Invalid or corrupted image input ---")
        corrupted_bytes = b"This is not a valid JPEG/PNG image file data"
        from backend.preprocessing import preprocess_image
        decoded = preprocess_image(corrupted_bytes)
        iq = assess_image_quality(decoded)

        success = (decoded is None) and (iq.is_standby is True)
        details = f"Decoded={decoded}, Is_standby={iq.is_standby}, Warning={iq.warning}"
        self.record_result("Test 8 (Corrupted / Invalid Image Input)", success, details)

    # ── Test 9: Model inference failure (Fail-safe) ───────────────────────────
    def test_9_model_inference_failure(self):
        print("\n--- Test 9: Model inference failure fail-safe ---")
        # Both models return None or crash
        dec = evaluate_hybrid_decision(
            yolo_detections=[],
            clf_result=None,
            is_standby=False,
            warning="Simulated inference crash"
        )

        # Condition: MUST return ERROR, NEVER PASS! Real defect must not silently pass!
        success = (dec.quality_status == "ERROR") and (dec.status_code == "MODELS_UNAVAILABLE")
        details = f"Status={dec.quality_status}, Status_code={dec.status_code}, PrimaryDefect={dec.primary_defect}"
        self.record_result("Test 9 (Model Failure Fail-Safe)", success, details)

    # ── Test 10: Camera disconnect or interrupted request ─────────────────────
    def test_10_camera_disconnect_or_interrupted(self):
        print("\n--- Test 10: Camera disconnect or interrupted request ---")
        # Simulate an empty frame buffer (camera stream disconnected)
        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        iq = assess_image_quality(empty_frame)

        dec = evaluate_hybrid_decision(
            yolo_detections=[],
            clf_result=None,
            is_standby=iq.is_standby,
            standby_reason=iq.warning
        )

        success = (dec.quality_status == "STANDBY") and (iq.is_standby is True)
        details = f"Status={dec.quality_status}, Warning={iq.warning}, Brightness={iq.mean_brightness}"
        self.record_result("Test 10 (Camera Disconnect / Interrupted)", success, details)

    def run_all(self):
        self.test_1_defect_free_fabric()
        self.test_2_visible_hole()
        self.test_3_stain_or_oil_spot()
        self.test_4_line_horizontal_vertical()
        self.test_5_yolo_misses_classifier_rescues()
        self.test_6_classifier_misses_yolo_recognizes()
        self.test_7_dark_camera_frame()
        self.test_8_invalid_image_input()
        self.test_9_model_inference_failure()
        self.test_10_camera_disconnect_or_interrupted()

        print("\n" + "=" * 70)
        print("TEST SUITE SUMMARY")
        print("=" * 70)
        print(f"Total Tests Run: {len(self.passed_tests) + len(self.failed_tests)}")
        print(f"Passed: {len(self.passed_tests)}")
        print(f"Failed: {len(self.failed_tests)}")
        if self.failed_tests:
            print("\nFailed Tests:")
            for name, details in self.failed_tests:
                print(f"  - {name}: {details}")
        print("=" * 70)
        return len(self.failed_tests) == 0


if __name__ == "__main__":
    suite = InspectionTestSuite()
    all_ok = suite.run_all()
    sys.exit(0 if all_ok else 1)
