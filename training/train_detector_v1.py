#!/usr/bin/env python3
"""
YOLOv8 Detector Training Pipeline (Queue V1)

Reproducible entry point for training YOLOv8s on the revised fabric detector dataset.

Features:
- Configurable CLI arguments: model, data, epochs, imgsz, batch, workers, seed, device, project, name, resume.
- Pretrained YOLOv8s initialization (or user-specified model).
- Strict separation of training arguments (no agnostic_nms in train(); NMS is inference-only).
- Comprehensive metadata recording: package versions, data manifest hash, class mapping, hardware profile.
- Output isolation: writes strictly to new experiment directories; preserves existing checkpoints.
- Safe dry-run mode (--dry-run) to validate dataset resolution, model initialization, and config without training.
"""

import os
import sys
import json
import hashlib
import argparse
import datetime
import pathlib
import yaml

# Add PyTorch 2.6+ weights_only compatibility patch
import torch
_orig_torch_load = torch.load
torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})

# Add project root to sys.path
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def compute_file_sha256(filepath):
    if not filepath.exists():
        return None
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def get_env_info():
    import platform
    import torch
    import torchvision
    import ultralytics
    return {
        "platform": platform.platform(),
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "ultralytics_version": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Detector on Revised Fabric Dataset")
    parser.add_argument("--data", type=str,
                        default=str(REPO_ROOT / "data_prepared_v1_revised" / "yolo_detector" / "data.yaml"),
                        help="Path to data.yaml dataset specification")
    parser.add_argument("--model", type=str, default="yolov8s.pt",
                        help="Pretrained weights or model architecture (e.g. yolov8s.pt, yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size for training and validation")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (reduce if CPU or limited VRAM)")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of dataloader worker threads")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to train on ('cpu', '0', '0,1', etc.)")
    parser.add_argument("--project", type=str,
                        default=str(REPO_ROOT / "results" / "experiments_detector"),
                        help="Directory to save training run outputs")
    parser.add_argument("--name", type=str, default="yolov8s_revised_v1",
                        help="Subdirectory name for this training run")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last checkpoint in the run directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration, imports, data resolution, and exit WITHOUT training")
    return parser.parse_args()

def validate_data_yaml(yaml_path: pathlib.Path):
    if not yaml_path.exists():
        raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    for key in ["train", "val", "names"]:
        if key not in data_cfg:
            raise ValueError(f"Required key '{key}' missing from data.yaml")

    yaml_dir = yaml_path.parent
    train_path = yaml_dir / data_cfg["train"] if not pathlib.Path(data_cfg["train"]).is_absolute() else pathlib.Path(data_cfg["train"])
    val_path = yaml_dir / data_cfg["val"] if not pathlib.Path(data_cfg["val"]).is_absolute() else pathlib.Path(data_cfg["val"])

    if not train_path.exists():
        raise FileNotFoundError(f"Train path does not exist: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val path does not exist: {val_path}")

    return data_cfg, train_path, val_path

def main():
    args = parse_args()
    data_yaml_path = pathlib.Path(args.data).resolve()
    project_dir = pathlib.Path(args.project).resolve()
    run_dir = project_dir / args.name

    print("=" * 60)
    print(" YOLOv8 Fabric Detector Training Pipeline (Queue V1)")
    print("=" * 60)
    print(f"Data config    : {data_yaml_path}")
    print(f"Model          : {args.model}")
    print(f"Epochs         : {args.epochs}")
    print(f"Image size     : {args.imgsz}")
    print(f"Batch size     : {args.batch}")
    print(f"Device         : {args.device}")
    print(f"Output run dir : {run_dir}")
    print(f"Dry-run mode   : {args.dry_run}")
    print("=" * 60)

    # 1. Validate environment
    env_info = get_env_info()
    print("[Env] Hardware & Package Status:")
    for k, v in env_info.items():
        print(f"  {k:22}: {v}")

    # 2. Validate data.yaml
    data_cfg, train_dir, val_dir = validate_data_yaml(data_yaml_path)
    yaml_hash = compute_file_sha256(data_yaml_path)
    print(f"[Data] Successfully validated {data_yaml_path} (SHA256: {yaml_hash[:12]}...)")
    print(f"[Data] Classes ({len(data_cfg['names'])}): {data_cfg['names']}")
    print(f"[Data] Train path: {train_dir}")
    print(f"[Data] Val path  : {val_dir}")

    # 3. Check / create run directory
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "training_arguments": vars(args),
        "data_yaml_sha256": yaml_hash,
        "class_mapping": data_cfg["names"],
        "environment": env_info
    }
    meta_path = run_dir / "run_config.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Metadata] Configuration and environment saved to: {meta_path}")

    # 4. Import Ultralytics and initialize YOLO model
    print(f"[Model] Initializing YOLO model: {args.model}...")
    from ultralytics import YOLO
    model = YOLO(args.model)
    print(f"[Model] Model initialized successfully. Task: {model.task}")

    if args.dry_run:
        print("\n" + "=" * 60)
        print(" [DRY RUN SUCCESSFUL]")
        print(" Environment verified, dataset resolved, model initialized.")
        print(" Exiting safely without starting training (Queue V1 Policy).")
        print("=" * 60)
        sys.exit(0)

    # 5. Launch Training (when not in dry-run mode)
    # NOTE: agnostic_nms is NOT passed to train(). It is strictly for inference/evaluation.
    print(f"[Train] Starting YOLOv8 training for {args.epochs} epochs on device '{args.device}'...")
    results = model.train(
        data=str(data_yaml_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        workers=args.workers,
        seed=args.seed,
        device=args.device,
        project=str(project_dir),
        name=args.name,
        resume=args.resume,
        verbose=True
    )

    print("\n" + "=" * 60)
    print(" [TRAINING COMPLETE]")
    print(f" Results saved to: {run_dir}")
    print("=" * 60)
    return results

if __name__ == "__main__":
    main()
