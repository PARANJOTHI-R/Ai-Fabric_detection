import os
import sys
import json
import random
import time
import hashlib
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_ROOT = os.path.join(PROJECT_ROOT, "Fabric Defects Dataset")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "new_classifier")
BACKEND_MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BACKEND_MODELS_DIR, exist_ok=True)


class FabricDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, preload=True):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.cached_images = []
        if preload:
            print(f"[Dataset] Preloading {len(image_paths)} images into memory (resized to 256x256)...", flush=True)
            for i, p in enumerate(image_paths):
                try:
                    with Image.open(p) as img:
                        # Resize to 256x256 once to fit in RAM (~500MB) and avoid decoding huge JPEGs every epoch
                        cached = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
                        self.cached_images.append(cached)
                except Exception:
                    # Fallback blank image
                    self.cached_images.append(Image.new("RGB", (256, 256)))
            print(f"[Dataset] Preloading complete! {len(self.cached_images)} images cached in RAM.", flush=True)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if self.cached_images:
            img = self.cached_images[idx]
        else:
            path = self.image_paths[idx]
            with Image.open(path) as f:
                img = f.convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def scan_and_prepare_data():
    """Scans dataset, removes duplicate files, and creates stratified train/val/test splits."""
    if not os.path.exists(DATASET_ROOT):
        raise FileNotFoundError(f"Dataset root not found: {DATASET_ROOT}")

    class_names = sorted([d for d in os.listdir(DATASET_ROOT) if os.path.isdir(os.path.join(DATASET_ROOT, d))])
    class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
    
    print(f"[Data Prep] Found classes: {class_to_idx}")

    all_paths = []
    all_labels = []
    seen_hashes = {}
    duplicates_dropped = 0

    for cls in class_names:
        folder = os.path.join(DATASET_ROOT, cls)
        for fname in os.listdir(folder):
            fpath = os.path.join(folder, fname)
            ext = os.path.splitext(fname)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png']:
                # Hash check to drop exact bit-for-bit duplicates
                try:
                    with open(fpath, "rb") as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    if file_hash in seen_hashes:
                        duplicates_dropped += 1
                        continue
                    seen_hashes[file_hash] = fpath
                    all_paths.append(fpath)
                    all_labels.append(class_to_idx[cls])
                except Exception:
                    continue

    print(f"[Data Prep] Total unique images: {len(all_paths)} (Dropped {duplicates_dropped} duplicates)")
    counts = Counter([class_names[lbl] for lbl in all_labels])
    for cls, cnt in counts.items():
        print(f"  {cls:<15}: {cnt}")

    # Stratified Train (70%), Val (15%), Test (15%) split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_paths, all_labels, test_size=0.30, random_state=SEED, stratify=all_labels
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=0.50, random_state=SEED, stratify=temp_labels
    )

    print(f"\n[Split] Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")

    # Save split info so evaluate_classifier.py can test on exact test set
    split_info = {
        "class_to_idx": class_to_idx,
        "classes": class_names,
        "train": [{"path": p, "label": int(l)} for p, l in zip(train_paths, train_labels)],
        "val": [{"path": p, "label": int(l)} for p, l in zip(val_paths, val_labels)],
        "test": [{"path": p, "label": int(l)} for p, l in zip(test_paths, test_labels)],
    }
    split_path = os.path.join(RESULTS_DIR, "dataset_split.json")
    with open(split_path, "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"[Split] Saved split details to {split_path}")

    # Save class mapping to backend models and results
    class_map_info = {
        "classes": class_names,
        "class_to_idx": class_to_idx,
        "idx_to_class": {idx: cls for idx, cls in enumerate(class_names)},
        "defective_classes": [cls for cls in class_names if cls.lower() != "defect free"]
    }
    with open(os.path.join(RESULTS_DIR, "classifier_classes.json"), "w") as f:
        json.dump(class_map_info, f, indent=2)
    with open(os.path.join(BACKEND_MODELS_DIR, "classifier_classes.json"), "w") as f:
        json.dump(class_map_info, f, indent=2)

    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_names


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_test_transform


def compute_class_weights(labels, num_classes):
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for i in range(num_classes):
        cnt = counts.get(i, 1)
        # Standard balanced weight: total / (num_classes * count)
        w = total / (num_classes * cnt)
        weights.append(w)
    # Normalize weights so mean is 1.0
    weights = np.array(weights, dtype=np.float32)
    weights = weights / np.mean(weights)
    print(f"[Class Weights] Normalized class weights: {weights.tolist()}")
    return torch.tensor(weights, dtype=torch.float32)


def build_model(num_classes):
    print("[Model] Initializing MobileNetV3-Small with ImageNet pretrained weights...")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Replace classifier head
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1


def evaluate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1


def main(epochs=12, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels), class_names = scan_and_prepare_data()
    num_classes = len(class_names)

    train_tf, val_tf = get_transforms()

    train_dataset = FabricDataset(train_paths, train_labels, transform=train_tf)
    val_dataset = FabricDataset(val_paths, val_labels, transform=val_tf)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Class weights to avoid false negatives on minority classes
    class_weights = compute_class_weights(train_labels, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Build model
    model = build_model(num_classes).to(device)

    # Stage 1: Freeze features and train classifier head for 3 epochs
    print("\n--- Stage 1: Classifier Head Warmup (3 epochs) ---")
    for param in model.features.parameters():
        param.requires_grad = False

    head_optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)

    for ep in range(1, 4):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, head_optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        el_time = time.time() - t0
        print(f"Warmup Ep {ep}/3 [{el_time:.1f}s] - Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}", flush=True)

    # Stage 2: Fine-tune entire network with smaller learning rate
    print("\n--- Stage 2: Full Network Fine-Tuning ---", flush=True)
    for param in model.features.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW([
        {'params': model.features.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 5e-4}
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [], 'train_acc': [], 'train_f1': [],
        'val_loss': [], 'val_acc': [], 'val_f1': []
    }

    best_val_f1 = 0.0
    best_epoch = 0
    patience = 5
    patience_counter = 0

    best_weights_path = os.path.join(RESULTS_DIR, "best_classifier.pth")
    backend_weights_path = os.path.join(BACKEND_MODELS_DIR, "fabric_classifier_best.pth")

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        el_time = time.time() - t0

        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['train_f1'].append(tr_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch:2d}/{epochs:2d} [{el_time:.1f}s] - Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} F1: {tr_f1:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}", flush=True)

        # Track best macro F1 (prioritizes balanced performance across all defect classes)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            
            # Save checkpoint
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'classes': class_names,
                'class_to_idx': {cls: i for i, cls in enumerate(class_names)},
                'architecture': 'mobilenet_v3_small',
                'epoch': epoch,
                'val_f1': val_f1,
                'val_acc': val_acc
            }
            torch.save(checkpoint, best_weights_path)
            torch.save(checkpoint, backend_weights_path)
            print(f"  ==> Saved new best model (Val Macro F1: {val_f1:.4f}, Acc: {val_acc:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best epoch was {best_epoch} with Val F1: {best_val_f1:.4f}")
                break

    print(f"\nTraining completed! Best model saved from epoch {best_epoch} with Val Macro-F1: {best_val_f1:.4f}")

    # Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', color='#2563EB', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', color='#DC2626', linewidth=2)
    plt.title('Training & Validation Loss', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', color='#2563EB', linewidth=2)
    plt.plot(history['val_acc'], label='Val Acc', color='#16A34A', linewidth=2)
    plt.plot(history['val_f1'], label='Val Macro-F1', color='#D97706', linewidth=2, linestyle='--')
    plt.title('Accuracy & Macro F1 Score', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "classifier_training_history.png")
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved training history plot to {plot_path}")


if __name__ == "__main__":
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    main(epochs=epochs, batch_size=32)
