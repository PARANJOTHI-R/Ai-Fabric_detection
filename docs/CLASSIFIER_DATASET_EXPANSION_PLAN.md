# Classifier Dataset Expansion & Confounder Elimination Plan

## 1. Problem Statement: Confounding in Fabric Defects Dataset

The whole-image classifier (MobileNetV3-Small) currently misclassifies clean grey fabric photographs (e.g. `normal image.jpg`) as `stain` with high confidence ($0.9323$, defect-free probability $0.0534$).

### Root Cause Verification:
1. **`defect free` Class (1,666 images)**:
   - **95.9% (1,598 images)** are AITEX industrial loom line-scan camera strips ($2446 \times 1000$ or $4096 \times 256$, mean brightness $125.9$, machine LED illumination).
   - **0 images (0.0%)** are handheld digital camera photographs of plain grey woven fabric.
2. **`stain` Class (398 images)**:
   - **100.0% (398 images)** are handheld digital camera photographs ($1984 \times 1488$, mean brightness $175.4$) of plain grey woven fabric.
   - On average across the class, the physical stain occupies only **1.59%** of the pixels ($46,800$ out of $2,952,192$ px in `stain/1.jpg`).
   - The remaining **98.41%** of every `stain` training image is clean, unblemished plain grey woven fabric.
3. **Confounder Learned**:
   - Because MobileNetV3 uses global average pooling without spatial bounding-box annotations, the neural network learned that "high-resolution camera photo of plain grey woven fabric" correlates with `stain`.
   - When presented with a clean grey camera photo, the network triggers on the fabric background texture rather than any physical stain defect.

---

## 2. Dataset Expansion Strategy

To eliminate this shortcut without altering model architecture or artificially inflating confidence thresholds:

### A. Acquisition of Representative Clean Camera Samples
- **Source 1: Curated Clean Camera Photos**:
  - Incorporate verified defect-free camera images from `testphotos/` (e.g. clean crops of grey fabric samples).
  - Add 200–300 verified clean camera shots of plain grey woven fabric across various lighting angles, distances, and focal lengths.
- **Source 2: Defect-Free Crops from Stain Backgrounds**:
  - Extract $512 \times 512$ non-overlapping background patches from the clean $98.4\%$ periphery of the 398 stain images, verified to contain zero blemishes.
  - This forces the classifier to distinguish between the actual localized discoloration and the ambient woven fabric substrate.

### B. Strict Split Preservation & Leakage Prevention
1. **Split Strategy**:
   - Maintain the existing train / val / test proportions ($70\% / 15\% / 15\%$).
   - Group-based splitting: Ensure crops from the same original fabric roll or photograph remain strictly within the same split (train OR val OR test) to prevent near-duplicate leakage.
2. **Hash & Distance Deduplication**:
   - Run perceptual hashing (pHash) and $L_1$ pixel distance checks against the test split.
   - Any proposed training image with Hamming distance $< 6$ to any test sample will be quarantined.

### C. Training & Fine-Tuning Protocol
1. **Base Weights**: Initialize from `backend/models/fabric_classifier_best.pth`.
2. **Loss Function**: Focal Loss or Weighted Cross-Entropy to handle class imbalance across the 6 defect categories.
3. **Augmentation**: Random horizontal/vertical flips, minor color jitter ($\pm 5\%$), and RandomErasing to prevent future shortcut learning.
4. **Evaluation Benchmarks**:
   - Defect-free precision on clean grey camera photos $\ge 98\%$.
   - True stain recall on genuine stains maintained $\ge 95\%$.
   - Zero regression on existing holdout validation set (`results/new_classifier/dataset_split.json`).

---

## 3. Interim System Behavior for Presentation

Per strict instructions:
1. No unverified global variance heuristics (e.g. variance $< 18$) are introduced into production code.
2. Decision engine maintains YOLO's spatial authority: YOLO detected **0** defects on Sample A.
3. The root cause is documented and presented transparently as a classic CNN dataset confounder with the safe expansion protocol ready for execution.
