# Dataset Manifest and Reproducibility Guide

This tool provides cryptographic integrity verification, split tracking, and session group mapping for the AI Fabric Defect Detection datasets.

---

## 1. Purpose & Capabilities

- **Cryptographic Audit**: Computes SHA-256 digests for every image and annotation file.
- **Structural Validation**: Reads image dimensions, color channels, formats, bounding box counts, and splits.
- **Capture-Group Tracking**: Extracts fabric-roll or camera-sweep session keys to detect leakage.
- **Cloud Handoff Verification**: Verifies uncorrupted transfer when copying data to Google Colab or a cloud training instance.
- **Safety Guarantee**: 100% read-only. Does not move, rename, or delete any dataset files.

---

## 2. Generating a Dataset Manifest

To generate a full manifest for `data_prepared_v1_revised/`:

```bash
python scripts/generate_dataset_manifest.py \
  --dataset-dir data_prepared_v1_revised \
  --output results/queue_v1/dataset_manifest.json \
  --version-tag v1_revised
```

Output:
- Detailed JSON manifest containing metadata for all files, duplicate clusters, split summaries, and size statistics.

---

## 3. Verifying Dataset Integrity on Another Machine

When copying `data_prepared_v1_revised/` to a cloud GPU VM (e.g. Google Colab, AWS, Lambda Labs), verify that zero files were truncated, corrupted, or omitted during compression/decompression:

```bash
python scripts/generate_dataset_manifest.py \
  --dataset-dir data_prepared_v1_revised \
  --verify results/queue_v1/dataset_manifest.json
```

Expected Output:
```text
[Verify] Matched: 4320/4320
[Verify] Missing: 0, Hash Mismatch: 0
[Verify] Verdict: PASS
```

If any file fails its checksum or is missing, the tool exits with code `1` and outputs the exact file path.

---

## 4. Manifest Schema Reference

Each entry in `entries` contains:
```json
{
  "relative_path": "yolo_detector/images/train/000_jpg.rf.0e1761d9a263bb49141f1737cbef760c.jpg",
  "filename": "000_jpg.rf.0e1761d9a263bb49141f1737cbef760c.jpg",
  "extension": ".jpg",
  "size_bytes": 45192,
  "sha256": "4391a0c87...",
  "task": "yolo_detector",
  "split": "train",
  "class_name": "unknown",
  "has_annotation": false,
  "box_count": 0,
  "group_key": "000",
  "image_properties": {
    "width": 640,
    "height": 640,
    "mode": "RGB",
    "format": "JPEG",
    "valid": true
  }
}
```
