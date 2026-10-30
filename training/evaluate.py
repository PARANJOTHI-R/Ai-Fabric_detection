import os
import torch

# Fix for PyTorch 2.6+ where weights_only=True is the default
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, 'weights_only': False})

from ultralytics import YOLO
import json
import shutil

def evaluate_model():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Prefer retrained full model if present, else fallback
    full_model_path = os.path.join(base_dir, 'results', 'yolov8s_fabric_full', 'weights', 'best.pt')
    fast_model_path = os.path.join(base_dir, 'results', 'yolov8_fabric_fast', 'weights', 'best.pt')
    backend_model_path = os.path.join(base_dir, 'backend', 'models', 'best.pt')
    
    if os.path.exists(full_model_path):
        model_path = full_model_path
    elif os.path.exists(fast_model_path):
        model_path = fast_model_path
    elif os.path.exists(backend_model_path):
        model_path = backend_model_path
    else:
        print(f"Error: Model not found. Searched {full_model_path}, {fast_model_path}, {backend_model_path}")
        return

    data_yaml = os.path.join(base_dir, 'dataset', 'data.yaml')
    results_dir = os.path.join(base_dir, 'results', 'evaluation')
    os.makedirs(results_dir, exist_ok=True)

    model = YOLO(model_path)
    print(f"Evaluating model: {model_path}")
    print(f"Using dataset config: {data_yaml}")
    
    # Run validation
    metrics = model.val(data=data_yaml, split='val', project=base_dir, name='results/evaluation', exist_ok=True)
    
    print("\n" + "=" * 65)
    print("YOLO DETECTION VALIDATION EVALUATION")
    print("=" * 65)
    print(f"mAP50-95:  {metrics.box.map:.4f} ({metrics.box.map * 100:.2f}%)")
    print(f"mAP50:     {metrics.box.map50:.4f} ({metrics.box.map50 * 100:.2f}%)")
    print(f"Precision: {metrics.box.mp:.4f} ({metrics.box.mp * 100:.2f}%)")
    print(f"Recall:    {metrics.box.mr:.4f} ({metrics.box.mr * 100:.2f}%)")
    
    # Per-class metrics
    per_class = {}
    names = model.names
    print("\nPer-Class Detection Performance:")
    print(f"{'Class':<15} | {'Precision':<10} | {'Recall':<10} | {'AP@50':<10} | {'mAP@50-95':<10}")
    print("-" * 65)
    for i, name in names.items():
        p = float(metrics.box.p[i]) if i < len(metrics.box.p) else 0.0
        r = float(metrics.box.r[i]) if i < len(metrics.box.r) else 0.0
        ap50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
        ap = float(metrics.box.ap[i]) if i < len(metrics.box.ap) else 0.0
        per_class[name] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "ap50": round(ap50, 4),
            "map50_95": round(ap, 4)
        }
        print(f"{name:<15} | {p*100:8.2f}% | {r*100:8.2f}% | {ap50*100:8.2f}% | {ap*100:8.2f}%")

    eval_metrics = {
        "model_path": model_path,
        "mAP50-95": round(float(metrics.box.map), 4),
        "mAP50": round(float(metrics.box.map50), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class": per_class
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(eval_metrics, f, indent=4)
        
    print(f"\n[Saved] Metrics saved to {results_dir}/metrics.json")
    return eval_metrics

if __name__ == '__main__':
    evaluate_model()
