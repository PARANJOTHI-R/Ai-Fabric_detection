import os
import torch

# Fix for PyTorch 2.6+ where weights_only=True is the default
_orig_load = torch.load
torch.load = lambda *args, **kwargs: _orig_load(*args, **{**kwargs, 'weights_only': False})

from ultralytics import YOLO

def train_model():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_yaml = os.path.join(base_dir, 'dataset', 'data.yaml')
    results_dir = os.path.join(base_dir, 'results')

    model = YOLO("yolov8s.pt")  # yolov8s: larger than nano, better recall

    # Full mode: 50 epochs for proper convergence
    epochs = 50
    img_size = 640
    batch_size = 16

    print(f"Starting FULL RETRAIN on: {data_yaml}")
    print(f"Model: yolov8s | Epochs: {epochs} | ImgSize: {img_size}")

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        project=results_dir,
        name='yolov8s_fabric_full',
        exist_ok=True,
        patience=15,
        device='',
        # Better augmentation for small dataset
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
        # Regularization
        dropout=0.1,
        weight_decay=0.0005,
    )

    print("Training complete. Best weights at:", 
          os.path.join(results_dir, 'yolov8s_fabric_full', 'weights', 'best.pt'))

if __name__ == '__main__':
    train_model()
