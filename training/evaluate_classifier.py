import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "new_classifier")
BACKEND_MODELS_DIR = os.path.join(PROJECT_ROOT, "backend", "models")


class SimpleDataset(Dataset):
    def __init__(self, items, transform=None):
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        path = item["path"]
        if not os.path.exists(path):
            if "Fabric Defects Dataset" in path:
                rel = path[path.index("Fabric Defects Dataset"):]
                path = os.path.join(PROJECT_ROOT, rel)
        label = item["label"]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


def load_model(checkpoint_path, num_classes, device):
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 65)
    print("NEW FABRIC CLASSIFIER EVALUATION ON HELD-OUT TEST SET")
    print("=" * 65)
    print(f"Device: {device}")

    split_file = os.path.join(RESULTS_DIR, "dataset_split.json")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Split file not found: {split_file}. Run train_classifier.py first.")

    with open(split_file, "r") as f:
        split_data = json.load(f)

    classes = split_data["classes"]
    test_items = split_data["test"]
    num_classes = len(classes)
    print(f"Classes ({num_classes}): {classes}")
    print(f"Held-out test images: {len(test_items)}")

    checkpoint_path = os.path.join(RESULTS_DIR, "best_classifier.pth")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(BACKEND_MODELS_DIR, "fabric_classifier_best.pth")

    model, checkpoint = load_model(checkpoint_path, num_classes, device)
    print(f"Loaded checkpoint saved from epoch {checkpoint.get('epoch', '?')} (Val F1: {checkpoint.get('val_f1', 0):.4f})")

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = SimpleDataset(test_items, transform=eval_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    y_true = []
    y_pred = []
    y_probs = []
    all_paths = []

    with torch.no_grad():
        for images, targets, paths in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(targets.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())
            all_paths.extend(paths)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # Calculate actual metrics
    acc = accuracy_score(y_true, y_pred)
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    wt_p, wt_r, wt_f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    per_class_p, per_class_r, per_class_f1, per_class_sup = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)

    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "-" * 65)
    print(f"Overall Accuracy:  {acc * 100:.2f}%")
    print(f"Macro Precision:   {macro_p * 100:.2f}%")
    print(f"Macro Recall:      {macro_r * 100:.2f}%")
    print(f"Macro F1-Score:    {macro_f1 * 100:.2f}%")
    print(f"Weighted F1-Score: {wt_f1 * 100:.2f}%")
    print("-" * 65)

    print("\nPer-Class Detailed Performance:")
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("-" * 65)
    per_class_dict = {}
    for idx, cls in enumerate(classes):
        p, r, f, s = per_class_p[idx], per_class_r[idx], per_class_f1[idx], per_class_sup[idx]
        per_class_dict[cls] = {
            "precision": round(float(p), 4),
            "recall": round(float(r), 4),
            "f1_score": round(float(f), 4),
            "support": int(s)
        }
        print(f"{cls:<15} | {p*100:8.2f}% | {r*100:8.2f}% | {f*100:8.2f}% | {s:8d}")

    # Binary quality classification metrics (Defective vs Pass)
    defect_free_idx = classes.index("defect free") if "defect free" in classes else -1
    if defect_free_idx != -1:
        binary_true = (y_true != defect_free_idx).astype(int)  # 1 = Defective, 0 = Pass
        binary_pred = (y_pred != defect_free_idx).astype(int)
        bin_acc = accuracy_score(binary_true, binary_pred)
        bin_p, bin_r, bin_f1, _ = precision_recall_fscore_support(binary_true, binary_pred, average='binary', zero_division=0)
        
        # False negatives (Defective fabric classified as defect free)
        fn_count = int(np.sum((binary_true == 1) & (binary_pred == 0)))
        fp_count = int(np.sum((binary_true == 0) & (binary_pred == 1)))
        tp_count = int(np.sum((binary_true == 1) & (binary_pred == 1)))
        tn_count = int(np.sum((binary_true == 0) & (binary_pred == 0)))
        fn_rate = fn_count / (tp_count + fn_count) if (tp_count + fn_count) > 0 else 0.0

        binary_metrics = {
            "defect_detection_accuracy": round(float(bin_acc), 4),
            "defect_recall_sensitivity": round(float(bin_r), 4),
            "defect_precision": round(float(bin_p), 4),
            "defect_f1": round(float(bin_f1), 4),
            "false_negative_count": fn_count,
            "false_positive_count": fp_count,
            "false_negative_rate": round(float(fn_rate), 4),
            "true_defective_detected": tp_count,
            "true_defect_free_passed": tn_count
        }
        print("\nBinary Quality Metrics (DEFECTIVE vs PASS):")
        print(f"  Defect Detection Accuracy: {bin_acc * 100:.2f}%")
        print(f"  Defect Recall (Sensitivity): {bin_r * 100:.2f}% (Only {fn_count} false negatives out of {tp_count + fn_count} defective)")
        print(f"  Defect Precision:            {bin_p * 100:.2f}%")
        print(f"  Defect F1-Score:             {bin_f1 * 100:.2f}%")
    else:
        binary_metrics = {}

    # Save metrics JSON
    metrics_summary = {
        "model_architecture": "mobilenet_v3_small",
        "dataset": "Fabric Defects Dataset",
        "test_sample_count": len(test_items),
        "overall_accuracy": round(float(acc), 4),
        "macro_precision": round(float(macro_p), 4),
        "macro_recall": round(float(macro_r), 4),
        "macro_f1": round(float(macro_f1), 4),
        "weighted_f1": round(float(wt_f1), 4),
        "per_class": per_class_dict,
        "binary_quality": binary_metrics,
        "confusion_matrix": cm.tolist(),
        "classes": classes
    }

    metrics_path = os.path.join(RESULTS_DIR, "classifier_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\n[Saved] Metrics saved to {metrics_path}")

    # Plot Confusion Matrix
    plt.figure(figsize=(8, 7))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Fabric Defect Classifier - Confusion Matrix', fontsize=13, fontweight='bold', pad=12)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=35, ha='right', fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val > thresh else "black"
            plt.text(j, i, f"{val}", horizontalalignment="center", verticalalignment="center",
                     color=color, fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.ylabel('Actual Label', fontsize=11, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=11, fontweight='bold')
    cm_path = os.path.join(RESULTS_DIR, "classifier_confusion_matrix.png")
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[Plot] Saved confusion matrix to {cm_path}")

    # Run explicit test predictions from EACH class
    print("\n" + "=" * 65)
    print("SAMPLE PREDICTIONS ACROSS EVERY CLASS")
    print("=" * 65)
    print(f"{'Actual Class':<15} | {'Predicted Class':<15} | {'Conf':<6} | {'Status':<10} | {'Match'}")
    print("-" * 65)

    sample_tests = []
    for cls_idx, cls_name in enumerate(classes):
        # Pick 2-3 sample images from this class from test set
        cls_test_indices = [i for i, item in enumerate(test_items) if item["label"] == cls_idx]
        for idx in cls_test_indices[:2]:
            pred_idx = y_pred[idx]
            pred_name = classes[pred_idx]
            conf = float(y_probs[idx][pred_idx])
            status = "PASS" if pred_name.lower() == "defect free" else "DEFECTIVE"
            match = "MATCH" if pred_idx == cls_idx else "MISMATCH"
            print(f"{cls_name:<15} | {pred_name:<15} | {conf*100:5.1f}% | {status:<10} | {match}")
            sample_tests.append({
                "actual": cls_name,
                "predicted": pred_name,
                "confidence": round(conf, 4),
                "status": status,
                "match": match,
                "path": all_paths[idx]
            })

    with open(os.path.join(RESULTS_DIR, "sample_test_predictions.json"), "w") as f:
        json.dump(sample_tests, f, indent=2)

    return metrics_summary


if __name__ == "__main__":
    evaluate_classifier()
