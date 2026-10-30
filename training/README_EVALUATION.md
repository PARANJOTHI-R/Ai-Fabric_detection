# Evaluation Guidelines and Split Protocol (Queue V1)

This document establishes the strict evaluation protocols, split isolation rules, and reporting standards for the AI Fabric Defect Detection system.

---

## 1. Strict Dataset Split Isolation Rules

To prevent data leakage, benchmark contamination, and overly optimistic performance estimates, the dataset splits must be strictly utilized according to their explicit roles:

```
data_prepared_v1_revised/
├── train/       <-- Model parameter optimization only
├── val/         <-- Hyperparameter tuning, checkpoint selection, validation metrics
└── test/        <-- STRICTLY HELD-OUT: Final benchmark reporting ONLY
```

### Protocol Requirements:
1. **Never Train on Validation or Test Data**: All model weights must be updated strictly using batches from `train/`.
2. **Never Select Checkpoints on Test Data**: Early stopping, learning rate scheduling, and threshold calibration must use `val/`.
3. **Safety Gate on Test Evaluation**: Both `evaluate_detector_v1.py` and `evaluate_classifier_v1.py` enforce a `--confirm-test-split` command-line flag. Without this explicit confirmation, evaluating against `test` is blocked.
4. **Final Model Benchmarking Only**: The test split should only be evaluated once a candidate model has completed training and validation.

---

## 2. Model Evaluation Tooling

### 2.1 Detector Evaluation (`training/evaluate_detector_v1.py`)

Evaluates bounding-box localization, class accuracy, and false alarm rate on clean negatives.

```bash
# Evaluate on the validation split:
python training/evaluate_detector_v1.py \
  --weights results/yolov8s_fabric_full/weights/best.pt \
  --data data_prepared_v1_revised/yolo_detector/data.yaml \
  --split val

# Evaluate on the held-out test split (Candidate verification only):
python training/evaluate_detector_v1.py \
  --weights results/yolov8s_fabric_full/weights/best.pt \
  --data data_prepared_v1_revised/yolo_detector/data.yaml \
  --split test \
  --confirm-test-split
```

**Output Artifacts**:
- `results/evaluations/detector_val/metrics.json`
- `results/evaluations/detector_val/evaluation_report.md`

### 2.2 Classifier Evaluation (`training/evaluate_classifier_v1.py`)

Evaluates multi-class condition classification, per-class F1, and clean-versus-defective discrimination.

```bash
# Evaluate on the validation split:
python training/evaluate_classifier_v1.py \
  --weights backend/models/fabric_classifier_best.pth \
  --data-dir data_prepared_v1_revised/classifier \
  --split val

# Evaluate on synthetic colored/patterned fabric samples:
python training/evaluate_classifier_v1.py \
  --weights backend/models/fabric_classifier_best.pth \
  --split synthetic
```

**Output Artifacts**:
- `results/evaluations/classifier_val/metrics.json`
- `results/evaluations/classifier_val/evaluation_report.md`

---

## 3. Generalization & Real-World vs. Synthetic Data

### 3.1 Synthetic Smoke Test (`validation/`)
- Contains 20 synthetic patterned/colored textile images (10 clean, 10 defective).
- **Purpose**: Fast functional smoke testing of model inference and camera rendering.
- **Strict Rule**: Results on these 20 images **MUST NOT** be reported as real-world accuracy or generalized proof of production readiness. They serve as a domain-shift diagnostic.

### 3.2 External Real-World Evaluation Convention (`validation_external/`)
To evaluate models against genuine industrial dyed, woven, knitted, and patterned textiles:
1. Create directory `validation_external/` with subfolders:
   ```
   validation_external/
   ├── clean/       # Real-world defect-free fabrics
   └── defective/   # Real-world defective fabrics
   ```
2. Maintain an explicit metadata manifest: `validation_external/metadata.json` recording:
   - Textile substrate (cotton, polyester, denim, jacquard, etc.).
   - Weave / pattern type (solid, twill, yarn-dyed stripe, printed motif).
   - Defect type, capture lighting, and source institution.
3. Ground truth must be forensically verified; **never fabricate bounding boxes or labels**.
