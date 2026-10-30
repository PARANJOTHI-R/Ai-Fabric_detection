# External Dataset Acquisition Plan (Queue V1)

**Date**: 2026-10-29  
**Target Milestone**: Expanding Dataset toward 8,000–10,000 usable images with real-world patterned, colored, and industrial textiles.  
**Strict Safety Policy**:
- No large archives are downloaded in this queue.
- External datasets are catalogued with official URLs, terms of use, licensing, and exact overlap warnings.
- Overlapping images (such as between TILDA and TILDA-OBB) must never be counted twice.

---

## 1. Catalog of Target External Datasets

### 1.1 Full TILDA Fabric Dataset

- **Official Publisher**: Working Group on Computer Vision, Institute for Computer Science, University of Freiburg, Germany (DFG Project).
- **Official URL**: `https://lmb.informatik.uni-freiburg.de/resources/datasets/tilda.en.html`
- **Access / Registration**: Public academic access. No fee or NDA required for educational/research purposes.
- **Licence**: Academic Non-Commercial Research License. Redistribution requires citation of the original DFG/Freiburg TILDA publications.
- **Published Image Count**: **3,200 images** ($768 \times 512$ pixels).
- **Fabric Types**: 8 diverse textile substrates:
  - 4 uncolored plain / neutral weaves ($W_1$ to $W_4$).
  - 4 colored and patterned woven/printed textiles ($W_5$ to $W_8$).
- **Defects & Annotations**:
  - 4 primary defect categories: holes, foreign objects, oil spots, thread errors.
  - 50 defect images per class per textile, plus reference defect-free samples under controlled lighting variations.
- **Expected Download Size**: ~380 MB (compressed archives).
- **Staging Directory**: `data_external/tilda_raw/`
- **Validation**:
  - Verify 3,200 source images across directories `w1/` through `w8/`.
  - Validate SHA-256 signatures against Freiburg manifest.

---

### 1.2 TILDA-OBB (Oriented Bounding Boxes)

- **Publisher / Maintainers**: Academic computer vision research group / Roboflow Universe mirror (`https://universe.roboflow.com/`).
- **Access / Registration**: Free account / Roboflow API key.
- **Licence**: CC BY 4.0 / Academic derivative of original TILDA.
- **Published Count**: 1,562 annotated defect images.
- **Annotation Format**: Oriented Bounding Boxes (OBB: $[x_c, y_c, w, h, \theta]$) and standard axis-aligned horizontal boxes.
- **CRITICAL OVERLAP WARNING**:
  > [!WARNING]
  > **Zero New Physical Fabrics**: TILDA-OBB is a geometric re-annotation and cropping of samples from the 3,200-image University of Freiburg TILDA dataset.
  > TILDA-OBB **MUST NOT** be counted as independent original images. Any ingestion must run cryptographic deduplication against `photos/` and `tilda_raw/`.
- **Expected Download Size**: ~180 MB.
- **Staging Directory**: `data_external/tilda_obb_raw/`

---

### 1.3 AITEX Fabric Image Database (AFID)

- **Official Publisher**: AITEX (Textile Industry Research Association, Alcoy, Spain) & Universitat Politècnica de València (UPV).
  - *Reference*: Silvestre-Blanes et al., "A Public Fabric Image Database for Defect Detection", Machine Vision and Applications / Mendeley Data.
- **Official DOI & URL**: `https://data.mendeley.com/datasets/72pm6wtd88/1` (DOI: 10.17632/72pm6wtd88.1).
- **Access / Registration**: Open Access download via Mendeley Data / UPV repository.
- **Licence**: **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Permitted for academic and commercial research with attribution.
- **Published Image Count**:
  - **245 defective fabric sweep images** ($4096 \times 256$ pixels high-resolution continuous line-scan).
  - **140 defect-free reference images** ($4096 \times 256$ pixels).
  - Total: 385 panoramic continuous frames across 7 distinct fabric structures.
- **Defects & Annotations**:
  - 12 industrial defect classes (broken threads, holes, oil stains, weft curls, warp contamination).
  - Ground truth provided as pixel-level binary segmentation masks (`.png`).
- **Expected Download Size**: ~420 MB.
- **Staging Directory**: `data_external/aitex_raw/`
- **Preprocessing Protocol**: Convert line-scan images into $640 \times 640$ overlapping tiles; convert binary masks into normalized YOLO bounding boxes.

---

### 1.4 Alibaba Tianchi Industrial Textile Defect Detection

- **Official Publisher**: Alibaba Cloud Tianchi Platform, Guangdong Provincial Department of Industry and Information Technology, Chinatex.
- **Official URL**: `https://tianchi.aliyun.com/competition/entrance/231748/introduction`
- **Access / Registration**: Requires registered Alibaba Cloud / Tianchi account, SMS/Identity verification, and agreement to competition dataset terms.
- **Licence**: Non-Commercial Research & Competition License. Strictly prohibits public redistribution.
- **Published Image Count**:
  - Round 1: ~4,000 industrial line-scan textile frames ($2446 \times 1000$ to $4096 \times 1000$).
  - Round 2: ~6,000 industrial textile frames.
  - Total: ~10,000 images covering plain, twill, jacquard, and textured fabric rolls.
- **Defects & Annotations**:
  - Multi-class bounding boxes in JSON format (over 20 industrial defect categories: coarse weft, broken end, oil stain, hole, foreign yarn).
- **Expected Download Size**: ~18 GB to 25 GB.
- **Staging Directory**: `data_external/tianchi_raw/`

---

## 2. Dataset Synthesis & Expansion Projection

By staging and deduplicating these four sources alongside local datasets:

| Dataset Source | Raw Images | Estimated Unique $640\times640$ Tiles | Defect Annotations | Substrate Variety |
|---|---|---|---|---|
| **Local Active (TILDA-400 + AITEX clean)** | 718 | 718 | 486 boxes | Solid plain weaves |
| **Full TILDA (Freiburg)** | 3,200 | 3,200 | ROI / Boxes | 8 fabrics (4 patterned) |
| **AITEX (High-Res AFID)** | 385 | ~2,300 (tiled) | Pixel masks $\to$ boxes | 7 industrial weaves |
| **Alibaba Tianchi (Subsample)** | ~4,000 | ~4,500 | Bounding boxes | Industrial textured |
| **Target Combined Ingestion** | **~8,300** | **~10,700** | **~9,500 boxes** | **High Pattern Diversity** |

---

## 3. Resumable Staging Workflow

To ensure reproducibility, safe decompression, and integrity validation:

1. Download archive into `data_external/staging/`.
2. Run `python scripts/stage_external_dataset.py --archive <path_to_zip> --target-dir data_external/<target_folder> --verify-only`.
3. Extract and compute SHA-256 manifests using `scripts/generate_dataset_manifest.py`.
4. Ingest and merge using group-aware split scripts.
