import os, torch, json

weights = os.path.join("backend", "models", "fabric_classifier_best.pth")
results_weights = os.path.join("results", "new_classifier", "best_classifier.pth")
metrics_path = os.path.join("results", "new_classifier", "classifier_metrics.json")
split_path = os.path.join("results", "new_classifier", "dataset_split.json")

print("=== Checking Trained Model Files ===")
for label, path in [
    ("backend/models/fabric_classifier_best.pth", weights),
    ("results/new_classifier/best_classifier.pth", results_weights),
    ("results/new_classifier/classifier_metrics.json", metrics_path),
    ("results/new_classifier/dataset_split.json", split_path),
]:
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"  FOUND   {label}  ({size/1e6:.2f} MB)")
    else:
        print(f"  MISSING {label}")

if os.path.exists(weights):
    ck = torch.load(weights, map_location="cpu", weights_only=False)
    print()
    print("=== Checkpoint Details ===")
    print(f"  Architecture : {ck.get('architecture', '?')}")
    print(f"  Trained epoch: {ck.get('epoch', '?')}")
    print(f"  Val Accuracy : {ck.get('val_acc', 0)*100:.2f}%")
    print(f"  Val Macro-F1 : {ck.get('val_f1', 0)*100:.2f}%")
    print(f"  Classes      : {ck.get('classes', [])}")
else:
    print()
    print("WARNING: No trained model found. You need to run training first.")
    print("  Run: python -m training.train_classifier")

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        m = json.load(f)
    bq = m.get("binary_quality", {})
    print()
    print("=== Evaluation Metrics (held-out test set) ===")
    print(f"  Test samples  : {m['test_sample_count']}")
    print(f"  Accuracy      : {m['overall_accuracy']*100:.2f}%")
    print(f"  Macro F1      : {m['macro_f1']*100:.2f}%")
    print(f"  Defect Recall : {bq.get('defect_recall_sensitivity',0)*100:.2f}%")
    print(f"  Defect F1     : {bq.get('defect_f1',0)*100:.2f}%")
    fn = bq.get('false_negative_count', '?')
    tp = bq.get('true_defective_detected', 0)
    total_def = (tp + fn) if isinstance(fn, int) else '?'
    print(f"  False Negs    : {fn} out of {total_def} defective images")
    print()
    print("=== Per-class Performance ===")
    for cls, v in m.get("per_class", {}).items():
        print(f"  {cls:<15} P={v['precision']*100:.1f}%  R={v['recall']*100:.1f}%  F1={v['f1_score']*100:.1f}%  support={v['support']}")
else:
    print()
    print("Metrics file not found. Run: python -m training.evaluate_classifier")
