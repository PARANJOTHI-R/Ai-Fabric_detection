import os
from ultralytics import YOLO
import json
import shutil

def evaluate_model():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    model_path = os.path.join(base_dir, 'results', 'yolov8_fabric_fast', 'weights', 'best.pt')
    data_yaml = os.path.join(base_dir, 'dataset', 'data.yaml')
    results_dir = os.path.join(base_dir, 'results', 'evaluation')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Please run train.py first.")
        return

    model = YOLO(model_path)
    print(f"Evaluating model: {model_path}")
    
    # Run validation
    metrics = model.val(data=data_yaml, project=base_dir, name='results/evaluation', exist_ok=True)
    
    print("Evaluation complete!")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    
    # Save a JSON file with metrics
    eval_metrics = {
        "mAP50-95": float(metrics.box.map),
        "mAP50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr)
    }
    
    with open(os.path.join(results_dir, 'metrics.json'), 'w') as f:
        json.dump(eval_metrics, f, indent=4)
        
    print(f"Metrics saved to {results_dir}/metrics.json")
    
    # Setup backend models directory
    backend_model_dir = os.path.join(base_dir, 'backend', 'models')
    os.makedirs(backend_model_dir, exist_ok=True)
    backend_model_path = os.path.join(backend_model_dir, 'best.pt')
    
    # Copy best model to backend
    shutil.copy(model_path, backend_model_path)
    print(f"Copied best model to {backend_model_path}")

if __name__ == '__main__':
    evaluate_model()
