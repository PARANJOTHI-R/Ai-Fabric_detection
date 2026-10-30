"""
Synthetic Fabric Samples Validation Runner (Smoke Test)
=======================================================

DISCLAIMER:
This evaluation is a synthetic-image smoke test designed to verify end-to-end
pipeline functionality, schema compliance, and sensitivity under out-of-domain
synthetic textures. It is NOT proof of real-world physical fabric inspection accuracy.

Processes all synthetic fabric images in:
  - validation/colored/clean
  - validation/colored/defect
  - validation/patterned/clean
  - validation/patterned/defect

Saves detailed per-sample results to:
  results/validation/fabric_validation.csv
"""

import os
import sys
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient
from backend.main import app, startup_event

VALIDATION_DIR = REPO_ROOT / "validation"
RESULTS_DIR = REPO_ROOT / "results" / "validation"
CSV_OUT_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else (RESULTS_DIR / "fabric_validation_verified.csv")


def discover_sample_images() -> List[Dict[str, Any]]:
    """Discovers all synthetic fabric samples under validation/ subfolders."""
    samples = []
    subsets = [
        ("colored", "clean", "PASS"),
        ("colored", "defect", "DEFECTIVE"),
        ("patterned", "clean", "PASS"),
        ("patterned", "defect", "DEFECTIVE"),
    ]

    for fabric_type, category, expected_status in subsets:
        folder = VALIDATION_DIR / fabric_type / category
        if not folder.exists():
            print(f"[Warning] Folder missing: {folder}")
            continue

        for file_path in sorted(folder.glob("*.png")):
            samples.append({
                "file_path": file_path,
                "filename": file_path.name,
                "relative_path": str(file_path.relative_to(REPO_ROOT)),
                "fabric_type": fabric_type,
                "category": category,
                "expected_status": expected_status,
            })

    return samples


def run_validation_batch():
    print("=" * 80)
    print("AI-BASED FABRIC DEFECT DETECTION — SYNTHETIC SAMPLES SMOKE TEST")
    print("=" * 80)
    print("DISCLAIMER:")
    print("  This evaluation is a synthetic-image smoke test for pipeline functionality")
    print("  and domain-shift diagnostics. It is NOT proof of real-world fabric accuracy.")
    print("=" * 80)

    # Initialize backend models and warmup
    print("\n[1/3] Initializing FastAPI application & model services...")
    startup_event()
    client = TestClient(app)

    # Discover images
    print("\n[2/3] Discovering synthetic fabric images...")
    samples = discover_sample_images()
    print(f"Found {len(samples)} synthetic images across 4 validation folders.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    print("\n[3/3] Running batch inference via POST /api/predict ...")
    print(f"{'Idx':<4} {'Type':<10} {'Cat':<7} {'Filename':<22} {'Expected':<10} {'Predicted':<10} {'Primary Defect':<15} {'Source':<10} {'Conf':<7} {'Latency':<8} {'Result':<8}")
    print("-" * 115)

    for idx, sample in enumerate(samples, 1):
        rel_path = sample["relative_path"]
        fname = sample["filename"]
        ftype = sample["fabric_type"]
        cat = sample["category"]
        expected = sample["expected_status"]

        record = {
            "index": idx,
            "filename": fname,
            "fabric_type": ftype,
            "category": cat,
            "expected_status": expected,
            "predicted_status": "ERROR",
            "is_correct": False,
            "error_type": "UNKNOWN",
            "primary_defect": None,
            "detection_source": "none",
            "confidence": 0.0,
            "total_defect_probability": 0.0,
            "defect_count": 0,
            "bounding_boxes": "[]",
            "latency_ms": 0.0,
            "warning": None,
            "status_code": "ERROR",
            "relative_path": rel_path,
            "error_detail": None,
        }

        try:
            with open(sample["file_path"], "rb") as f:
                file_bytes = f.read()

            t0 = time.perf_counter()
            response = client.post(
                "/api/predict",
                files={"file": (fname, file_bytes, "image/png")}
            )
            roundtrip_ms = round((time.perf_counter() - t0) * 1000, 1)

            if response.status_code == 200:
                data = response.json()
                predicted = data.get("quality_status", "UNKNOWN")
                primary_def = data.get("primary_defect")
                source = data.get("detection_source", "none")
                conf = data.get("confidence") or 0.0
                total_prob = data.get("total_defect_probability") or 0.0
                def_count = data.get("defect_count", 0)
                bboxes = data.get("detections", [])
                proc_ms = data.get("processing_time_ms", roundtrip_ms)
                warn = data.get("warning")
                status_code = data.get("status_code", "OK")

                is_correct = (predicted == expected)

                # Categorize error type
                if is_correct:
                    err_type = "NONE"
                elif expected == "PASS" and predicted == "DEFECTIVE":
                    err_type = "FALSE_POSITIVE"  # False Alarm on Clean
                elif expected == "DEFECTIVE" and predicted == "PASS":
                    err_type = "FALSE_NEGATIVE"  # Missed Defect
                else:
                    err_type = "UNEXPECTED_STATUS"

                record.update({
                    "predicted_status": predicted,
                    "is_correct": is_correct,
                    "error_type": err_type,
                    "primary_defect": primary_def,
                    "detection_source": source,
                    "confidence": round(float(conf), 4),
                    "total_defect_probability": round(float(total_prob), 4),
                    "defect_count": def_count,
                    "bounding_boxes": json.dumps(bboxes),
                    "latency_ms": proc_ms,
                    "warning": warn,
                    "status_code": status_code,
                })
            else:
                record["error_detail"] = f"HTTP {response.status_code}: {response.text}"
                record["error_type"] = "HTTP_ERROR"

        except Exception as e:
            record["error_detail"] = str(e)
            record["error_type"] = "EXCEPTION"

        results.append(record)

        match_str = "MATCH" if record["is_correct"] else f"FAIL ({record['error_type']})"
        pdef_str = str(record["primary_defect"])[:14]
        print(f"{idx:<4} {ftype:<10} {cat:<7} {fname:<22} {expected:<10} {record['predicted_status']:<10} {pdef_str:<15} {record['detection_source']:<10} {record['confidence']:<7.4f} {record['latency_ms']:<8.1f} {match_str}")

    # Write CSV
    print(f"\nSaving results to {CSV_OUT_PATH} ...")
    fieldnames = [
        "index",
        "filename",
        "fabric_type",
        "category",
        "expected_status",
        "predicted_status",
        "is_correct",
        "error_type",
        "primary_defect",
        "detection_source",
        "confidence",
        "total_defect_probability",
        "defect_count",
        "bounding_boxes",
        "latency_ms",
        "warning",
        "status_code",
        "relative_path",
        "error_detail",
    ]

    with open(CSV_OUT_PATH, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Detailed CSV saved successfully: {CSV_OUT_PATH}")

    # Compute and print comprehensive summary
    compute_and_print_summary(results)


def compute_and_print_summary(results: List[Dict[str, Any]]):
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY: COLORED VS. PATTERNED SYNTHETIC FABRICS")
    print("=" * 80)

    for ftype in ["colored", "patterned"]:
        sub = [r for r in results if r["fabric_type"] == ftype]
        clean_sub = [r for r in sub if r["category"] == "clean"]
        defect_sub = [r for r in sub if r["category"] == "defect"]

        clean_total = len(clean_sub)
        clean_correct = sum(1 for r in clean_sub if r["is_correct"])
        false_alarms = sum(1 for r in clean_sub if r["error_type"] == "FALSE_POSITIVE")

        defect_total = len(defect_sub)
        defect_correct = sum(1 for r in defect_sub if r["is_correct"])
        missed_defects = sum(1 for r in defect_sub if r["error_type"] == "FALSE_NEGATIVE")

        total = len(sub)
        total_correct = sum(1 for r in sub if r["is_correct"])
        acc = (total_correct / total * 100) if total > 0 else 0.0

        print(f"\n--- {ftype.upper()} FABRICS SUMMARY (n={total}) ---")
        print(f"  Clean Samples (Expected PASS):")
        print(f"    - Correctly Passed:  {clean_correct}/{clean_total} ({clean_correct/clean_total*100:.1f}%)" if clean_total else "    - None")
        print(f"    - False Alarms:      {false_alarms}/{clean_total} ({false_alarms/clean_total*100:.1f}%)" if clean_total else "    - None")
        print(f"  Defective Samples (Expected DEFECTIVE):")
        print(f"    - Correctly Detected: {defect_correct}/{defect_total} ({defect_correct/defect_total*100:.1f}%)" if defect_total else "    - None")
        print(f"    - Missed Defects:    {missed_defects}/{defect_total} ({missed_defects/defect_total*100:.1f}%)" if defect_total else "    - None")
        print(f"  Category Accuracy: {total_correct}/{total} ({acc:.1f}%)")

    # Overall Summary
    total_samples = len(results)
    clean_all = [r for r in results if r["category"] == "clean"]
    defect_all = [r for r in results if r["category"] == "defect"]

    total_clean = len(clean_all)
    total_false_alarms = sum(1 for r in clean_all if r["error_type"] == "FALSE_POSITIVE")
    total_defective = len(defect_all)
    total_missed = sum(1 for r in defect_all if r["error_type"] == "FALSE_NEGATIVE")
    overall_correct = sum(1 for r in results if r["is_correct"])
    overall_acc = (overall_correct / total_samples * 100) if total_samples > 0 else 0.0

    avg_latency = (sum(r["latency_ms"] for r in results) / total_samples) if total_samples > 0 else 0.0

    print("\n" + "=" * 80)
    print("OVERALL BATCH PERFORMANCE")
    print("=" * 80)
    print(f"Total Synthetic Samples:       {total_samples}")
    print(f"Overall Accuracy:              {overall_correct}/{total_samples} ({overall_acc:.1f}%)")
    print(f"Clean Samples Accuracy:        {total_clean - total_false_alarms}/{total_clean} ({(total_clean - total_false_alarms)/total_clean*100:.1f}%)")
    print(f"  - False Alarm Count (FP):    {total_false_alarms}/{total_clean} ({total_false_alarms/total_clean*100:.1f}%)")
    print(f"Defect Samples Accuracy:       {total_defective - total_missed}/{total_defective} ({(total_defective - total_missed)/total_defective*100:.1f}%)")
    print(f"  - Missed Defect Count (FN):  {total_missed}/{total_defective} ({total_missed/total_defective*100:.1f}%)")
    print(f"Average Pipeline Latency:      {avg_latency:.1f} ms")
    print("=" * 80)
    print("SMOKE TEST NOTE:")
    print("  Synthetic fabrics introduce generative artifacts, non-standard weaves, and")
    print("  intense color saturation unseen in the TILDA-400 or Fabric Defects training")
    print("  distributions. These results identify domain-shift sensitivities, not true")
    print("  production inspection efficacy on real textiles.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_validation_batch()
