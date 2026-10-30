#!/usr/bin/env python3
"""
Dataset Manifest & Reproducibility Utility (Queue V1)

Scans a dataset directory (read-only) and generates a cryptographic and structural manifest:
- File relative path, SHA-256 checksum, byte size, image dimensions (width, height, channels).
- Split (train/val/test), task (yolo_detector / classifier), class / annotation status.
- Grouping key / capture session ID.
- Exact duplicates and hash collision detection.
- Audit mode and verification mode (to verify dataset integrity after transfer to GPU / cloud).

Strict Safety:
- Never moves, alters, or deletes any files.
"""

import os
import sys
import json
import hashlib
import argparse
import datetime
import pathlib
from collections import defaultdict
from PIL import Image

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def inspect_image(filepath):
    try:
        with Image.open(filepath) as img:
            w, h = img.size
            mode = img.mode
            fmt = img.format or filepath.suffix.upper().lstrip(".")
            return {"width": w, "height": h, "mode": mode, "format": fmt, "valid": True}
    except Exception as e:
        return {"valid": False, "error": str(e)}

def parse_group_key(filename, class_name=None):
    stem = pathlib.Path(filename).stem
    if class_name == "lines" and stem.startswith("line_2018-10-10"):
        parts = stem.split("_")
        return "_".join(parts[:3]) if len(parts) >= 3 else parts[0]
    elif stem.startswith("clean_"):
        return stem.split("_clean_")[0]
    elif "_jpg" in stem:
        return stem.split("_jpg")[0]
    else:
        return stem.split("_")[0]

def generate_manifest(dataset_dir: pathlib.Path, version: str = "v1_revised") -> dict:
    dataset_dir = dataset_dir.resolve()
    print(f"[Manifest] Scanning directory: {dataset_dir}")
    
    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
    entries = []
    sha_map = defaultdict(list)
    group_map = defaultdict(set)
    split_counts = defaultdict(lambda: defaultdict(int))
    total_bytes = 0

    all_files = sorted(list(dataset_dir.rglob("*")))
    file_records = [f for f in all_files if f.is_file() and not f.name.startswith(".")]

    print(f"[Manifest] Total files found: {len(file_records)}")

    for idx, filepath in enumerate(file_records):
        rel_path = filepath.relative_to(dataset_dir).as_posix()
        ext = filepath.suffix.lower()
        size_bytes = filepath.stat().st_size
        total_bytes += size_bytes
        sha = compute_sha256(filepath)
        sha_map[sha].append(rel_path)

        parts = rel_path.split("/")
        task = parts[0] if len(parts) > 0 else "unknown"
        
        split = "unknown"
        class_name = "unknown"
        has_annotation = False
        box_count = 0

        if "yolo_detector" in rel_path:
            # yolo structure: yolo_detector/{images,labels}/{train,val,test}/file
            if len(parts) >= 3:
                split = parts[2]
            if ext == ".txt":
                has_annotation = True
                with open(filepath, "r", encoding="utf-8") as f:
                    box_count = sum(1 for line in f if line.strip())
        elif "classifier" in rel_path:
            # classifier structure: classifier/{train,val,test}/class_name/file
            if len(parts) >= 3:
                split = parts[1]
            if len(parts) >= 4:
                class_name = parts[2]

        # Extract dimensions if image
        img_info = {}
        if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            img_info = inspect_image(filepath)

        group_key = parse_group_key(filepath.name, class_name)
        if split != "unknown":
            group_map[split].add(group_key)
            split_counts[task][split] += 1

        entry = {
            "relative_path": rel_path,
            "filename": filepath.name,
            "extension": ext,
            "size_bytes": size_bytes,
            "sha256": sha,
            "task": task,
            "split": split,
            "class_name": class_name,
            "has_annotation": has_annotation,
            "box_count": box_count,
            "group_key": group_key,
            "image_properties": img_info
        }
        entries.append(entry)

        if (idx + 1) % 500 == 0 or idx == len(file_records) - 1:
            print(f"  Processed {idx + 1}/{len(file_records)} files...")

    # Identify duplicate clusters
    duplicate_clusters = {
        sha: paths for sha, paths in sha_map.items() if len(paths) > 1
    }

    manifest = {
        "manifest_version": "1.0",
        "dataset_version": version,
        "dataset_root": str(dataset_dir),
        "generated_at_utc": timestamp,
        "total_files": len(entries),
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / (1024 * 1024), 2),
        "unique_sha256_count": len(sha_map),
        "duplicate_clusters_count": len(duplicate_clusters),
        "duplicate_clusters": duplicate_clusters,
        "split_summary": {t: dict(sc) for t, sc in split_counts.items()},
        "group_summary": {s: len(grps) for s, grps in group_map.items()},
        "entries": entries
    }
    return manifest

def verify_manifest(manifest_path: pathlib.Path, target_dir: pathlib.Path) -> dict:
    target_dir = target_dir.resolve()
    print(f"[Verify] Verifying dataset at: {target_dir} against manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    entries = manifest.get("entries", [])
    results = {
        "total_expected": len(entries),
        "matched": 0,
        "missing": [],
        "hash_mismatch": [],
        "status": "PASS"
    }

    for idx, e in enumerate(entries):
        rel_path = e["relative_path"]
        expected_sha = e["sha256"]
        actual_path = target_dir / rel_path

        if not actual_path.exists():
            results["missing"].append(rel_path)
            continue

        actual_sha = compute_sha256(actual_path)
        if actual_sha != expected_sha:
            results["hash_mismatch"].append({
                "path": rel_path,
                "expected": expected_sha,
                "actual": actual_sha
            })
        else:
            results["matched"] += 1

        if (idx + 1) % 1000 == 0:
            print(f"  Verified {idx + 1}/{len(entries)} files...")

    if results["missing"] or results["hash_mismatch"]:
        results["status"] = "FAIL"

    print(f"[Verify] Matched: {results['matched']}/{results['total_expected']}")
    print(f"[Verify] Missing: {len(results['missing'])}, Hash Mismatch: {len(results['hash_mismatch'])}")
    print(f"[Verify] Verdict: {results['status']}")
    return results

def main():
    parser = argparse.ArgumentParser(description="Fabric Dataset Manifest & Integrity Tool")
    parser.add_argument("--dataset-dir", type=str, default="data_prepared_v1_revised",
                        help="Path to dataset directory to inspect")
    parser.add_argument("--output", type=str, default="results/queue_v1/dataset_manifest.json",
                        help="Output path for JSON manifest")
    parser.add_argument("--version-tag", type=str, default="v1_revised",
                        help="Version tag for dataset")
    parser.add_argument("--verify", type=str, default=None,
                        help="Path to existing manifest file to verify against --dataset-dir")

    args = parser.parse_args()

    dataset_path = pathlib.Path(args.dataset_dir)
    if not dataset_path.exists():
        print(f"Error: Dataset directory {dataset_path} does not exist.")
        sys.exit(1)

    if args.verify:
        v_results = verify_manifest(pathlib.Path(args.verify), dataset_path)
        sys.exit(0 if v_results["status"] == "PASS" else 1)
    else:
        manifest = generate_manifest(dataset_path, version=args.version_tag)
        out_path = pathlib.Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"[Manifest] Successfully generated manifest: {out_path} ({len(manifest['entries'])} items)")

if __name__ == "__main__":
    main()
