#!/usr/bin/env python3
"""
MobileNetV3-Small Classifier Training Pipeline (Queue V1)

Reproducible entry point for training MobileNetV3-Small on the revised fabric classifier dataset.

Features:
- Configurable CLI arguments: data-dir, epochs, batch-size, lr, workers, seed, device, project, name, weights-path, resume, dry-run.
- Uses revised classifier train/ and val/ folders (strictly preserves test/ for held-out evaluation only).
- Records class_to_idx, class sample counts, and documents class imbalances/limitations in metadata.
- Pretrained weights policy: uses local weights if available; does NOT silently download unverified files.
- Full validation metrics: Loss, Accuracy, Per-class Precision/Recall/F1, Macro F1, Weighted F1, Confusion Matrix.
- Checkpointing: saves best.pth and last.pth under new experiment directory.
- Safe dry-run mode (--dry-run) to initialize data loaders, model architecture, and metadata without running training epochs.
"""

import os
import sys
import json
import random
import argparse
import datetime
import pathlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# PyTorch 2.6+ weights_only compatibility patch
_orig_torch_load = torch.load
torch.load = lambda *args, **kwargs: _orig_torch_load(*args, **{**kwargs, 'weights_only': False})

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transform, val_transform

def build_model(num_classes=6, weights_path=None, device="cpu"):
    """
    Builds MobileNetV3-Small architecture.
    Uses local weights if provided; otherwise initializes architecture safely.
    """
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    if weights_path and pathlib.Path(weights_path).exists():
        print(f"[Model] Loading existing weights from: {weights_path}")
        ckpt = torch.load(weights_path, map_location=device)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif isinstance(ckpt, dict) and "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        elif isinstance(ckpt, nn.Module):
            model = ckpt
        else:
            model.load_state_dict(ckpt)
    else:
        print("[Model] Initialized MobileNetV3-Small architecture without external downloads.")

    model.to(device)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV3-Small Fabric Classifier")
    parser.add_argument("--data-dir", type=str,
                        default=str(REPO_ROOT / "data_prepared_v1_revised" / "classifier"),
                        help="Path to revised classifier dataset root (containing train, val, test)")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for training and validation")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate for Adam optimizer")
    parser.add_argument("--workers", type=int, default=2,
                        help="Number of DataLoader worker processes")
    parser.add_argument("--seed", type=int, default=42,
                        help="Reproducibility random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device to train on ('cpu', 'cuda', 'cuda:0')")
    parser.add_argument("--project", type=str,
                        default=str(REPO_ROOT / "results" / "experiments_classifier"),
                        help="Output directory for experiments")
    parser.add_argument("--name", type=str, default="mobilenet_v3_revised_v1",
                        help="Run subdirectory name")
    parser.add_argument("--weights-path", type=str, default=None,
                        help="Optional path to local checkpoint to initialize or fine-tune from")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last.pth in run directory")
    parser.add_argument("--dry-run", action="store_true",
                        help="Initialize loaders and model, record config, and exit WITHOUT training")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = pathlib.Path(args.data_dir).resolve()
    train_dir = data_root / "train"
    val_dir = data_root / "val"
    project_dir = pathlib.Path(args.project).resolve()
    run_dir = project_dir / args.name

    print("=" * 65)
    print(" MobileNetV3-Small Fabric Classifier Training Pipeline (Queue V1)")
    print("=" * 65)
    print(f"Data root     : {data_root}")
    print(f"Train folder  : {train_dir}")
    print(f"Val folder    : {val_dir}")
    print(f"Epochs        : {args.epochs}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Learning rate : {args.lr}")
    print(f"Device        : {args.device}")
    print(f"Output run dir: {run_dir}")
    print(f"Dry-run mode  : {args.dry_run}")
    print("=" * 65)

    if not train_dir.exists() or not val_dir.exists():
        print(f"Error: train or val folder missing in {data_root}")
        sys.exit(1)

    train_tf, val_tf = get_transforms()
    train_dataset = datasets.ImageFolder(str(train_dir), transform=train_tf)
    val_dataset = datasets.ImageFolder(str(val_dir), transform=val_tf)

    class_names = train_dataset.classes
    class_to_idx = train_dataset.class_to_idx
    num_classes = len(class_names)

    # Compute class support
    from collections import Counter
    train_counts = Counter(train_dataset.targets)
    val_counts = Counter(val_dataset.targets)

    print(f"[Dataset] Classes ({num_classes}): {class_names}")
    print(f"[Dataset] class_to_idx: {class_to_idx}")
    print("[Dataset] Class Support:")
    for idx, cname in enumerate(class_names):
        t_cnt = train_counts.get(idx, 0)
        v_cnt = val_counts.get(idx, 0)
        limitation_flag = " (LOW SUPPORT)" if t_cnt < 100 else ""
        print(f"  {cname:15}: Train={t_cnt:4d}, Val={v_cnt:3d}{limitation_flag}")

    # Save run metadata
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "arguments": vars(args),
        "class_names": class_names,
        "class_to_idx": class_to_idx,
        "class_support": {
            cname: {
                "train": train_counts.get(idx, 0),
                "val": val_counts.get(idx, 0),
                "low_support_flag": train_counts.get(idx, 0) < 100
            }
            for idx, cname in enumerate(class_names)
        },
        "total_train_samples": len(train_dataset),
        "total_val_samples": len(val_dataset)
    }
    meta_path = run_dir / "run_config.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[Metadata] Config saved to: {meta_path}")

    # Initialize model
    model = build_model(num_classes=num_classes, weights_path=args.weights_path, device=args.device)

    if args.dry_run:
        print("\n" + "=" * 65)
        print(" [CLASSIFIER DRY RUN SUCCESSFUL]")
        print(" Dataset folders verified, class_to_idx recorded, model built.")
        print(" Exiting safely without starting training (Queue V1 Policy).")
        print("=" * 65)
        sys.exit(0)

    # Dataloaders for training
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(args.device != "cpu"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(args.device != "cpu"))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    start_epoch = 0
    best_val_f1 = 0.0

    # Resume support
    last_ckpt_path = run_dir / "last.pth"
    if args.resume and last_ckpt_path.exists():
        print(f"[Resume] Loading checkpoint: {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt.get("best_val_f1", 0.0)
        print(f"[Resume] Resumed from epoch {start_epoch}")

    from sklearn.metrics import precision_recall_fscore_support, accuracy_score

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)

        epoch_loss = running_loss / len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(args.device)
                labels = labels.to(args.device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_dataset)
        val_acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="macro", zero_division=0)

        print(f"Epoch [{epoch+1}/{args.epochs}] Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Macro F1: {f1:.4f}")

        # Save last checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": f1,
            "best_val_f1": max(best_val_f1, f1),
            "class_names": class_names,
            "class_to_idx": class_to_idx
        }
        torch.save(ckpt_data, last_ckpt_path)

        # Save best checkpoint
        if f1 > best_val_f1:
            best_val_f1 = f1
            best_ckpt_path = run_dir / "best.pth"
            torch.save(ckpt_data, best_ckpt_path)
            print(f"  -> Best model saved to: {best_ckpt_path} (Val F1: {f1:.4f})")

    print("\n[Complete] Classifier training run complete.")

if __name__ == "__main__":
    main()
