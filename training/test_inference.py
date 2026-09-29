"""
Test YOLO inference on fabric images.
Usage: python training/test_inference.py --image <path> [--conf 0.15]
       python training/test_inference.py --all         (test all class samples)
"""
import argparse
import os
import sys
import json
import torch

# Fix for PyTorch 2.6+
_orig = torch.load
torch.load = lambda *a, **kw: _orig(*a, **{**kw, 'weights_only': False})

from ultralytics import YOLO

CLASS_NAMES = {0: 'hole', 1: 'objects', 2: 'oil spot', 3: 'thread error'}

def find_model():
    paths = [
        os.path.join('results', 'yolov8s_fabric_full', 'weights', 'best.pt'),
        os.path.join('backend', 'models', 'best.pt'),
    ]
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def test_image(model, img_path, conf, gt_label=None):
    results = model(img_path, conf=conf, verbose=False)[0]
    dets = []
    for box in results.boxes:
        cid = int(box.cls[0].item())
        cv = float(box.conf[0].item())
        x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
        dets.append({
            'class': CLASS_NAMES.get(cid, '?'),
            'confidence': cv,
            'bbox': [round(x1,1), round(y1,1), round(x2,1), round(y2,1)]
        })
    
    print(f"Image: {os.path.basename(img_path)}")
    if gt_label:
        print(f"  Ground Truth: {gt_label}")
    print(f"  Detections: {len(dets)}")
    for i, d in enumerate(dets, 1):
        print(f"  {i}. {d['class']}")
        print(f"     Confidence: {d['confidence']:.4f}")
        print(f"     BBox: {d['bbox']}")
    if not dets:
        print("  (no detections)")
    print()
    return dets

def main():
    parser = argparse.ArgumentParser(description="Test YOLO fabric defect model")
    parser.add_argument("--image", type=str, help="Path to a single image")
    parser.add_argument("--all", action="store_true", help="Test sample images for all 4 classes")
    parser.add_argument("--conf", type=float, default=0.15, help="Confidence threshold (default: 0.15)")
    args = parser.parse_args()

    model_path = find_model()
    if not model_path:
        print("ERROR: Could not find model weights")
        sys.exit(1)

    print(f"Model: {model_path}")
    print(f"Conf threshold: {args.conf}\n")
    model = YOLO(model_path)

    if args.image:
        test_image(model, args.image, args.conf)
    elif args.all:
        from collections import defaultdict
        labels_dir = 'labels'
        photos_dir = 'photos'
        imgs_per_class = defaultdict(list)
        for lf in sorted(os.listdir(labels_dir)):
            if not lf.endswith('.txt'): continue
            with open(os.path.join(labels_dir, lf)) as f:
                lines = [l.strip() for l in f if l.strip()]
            for line in lines:
                cls = int(line.split()[0])
                base = lf.replace('.txt', '')
                if base not in imgs_per_class[cls]:
                    imgs_per_class[cls].append(base)

        for cls_id in range(4):
            name = CLASS_NAMES[cls_id]
            bases = imgs_per_class[cls_id][:2]
            print(f"=== {name.upper()} ===")
            for base in bases:
                for ext in ['.jpg', '.jpeg', '.png']:
                    p = os.path.join(photos_dir, base + ext)
                    if os.path.exists(p):
                        test_image(model, p, args.conf, gt_label=name)
                        break
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
