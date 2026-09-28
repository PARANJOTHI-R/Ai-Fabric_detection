import os
from ultralytics import YOLO

def train_model():
    # Detect absolute path
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_yaml = os.path.join(base_dir, 'dataset', 'data.yaml')
    results_dir = os.path.join(base_dir, 'results')
    
    # Load a model
    # Using yolov8n as a lightweight model, compatible with most ultralytics versions
    model = YOLO("yolov8n.pt")  

    # Fast mode configuration
    # Set higher epochs for FULL MODE if needed
    epochs = 15
    img_size = 640
    batch_size = 16
    
    print(f"Starting training on dataset: {data_yaml}")
    print(f"Mode: FAST MODE | Epochs: {epochs} | Img Size: {img_size}")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=results_dir,
        name='yolov8_fabric_fast',
        exist_ok=True,
        patience=5,
        device='' # Auto-detects GPU or CPU
    )
    
    print("Training complete. Results saved in:", results_dir)

if __name__ == '__main__':
    train_model()
