import os
from PIL import Image

def analyze_dataset(workspace_path):
    # Checking for 'photos' vs 'images'
    photos_dir = os.path.join(workspace_path, 'photos')
    if not os.path.exists(photos_dir):
        images_dir = os.path.join(workspace_path, 'images')
        if os.path.exists(images_dir):
            os.rename(images_dir, photos_dir)
            print(f"Renamed {images_dir} to {photos_dir}")
        else:
            print(f"Error: Neither 'photos' nor 'images' directory found.")
            return

    labels_dir = os.path.join(workspace_path, 'labels')
    
    # Get lists of files
    photo_files = [f for f in os.listdir(photos_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
    
    print(f"Number of images: {len(photo_files)}")
    print(f"Number of label files: {len(label_files)}")
    
    # Analyze dimensions and formats
    formats = set()
    dimensions = set()
    for pf in photo_files[:10]: # Check first 10 for speed
        try:
            with Image.open(os.path.join(photos_dir, pf)) as img:
                formats.add(img.format)
                dimensions.add(img.size)
        except Exception as e:
            print(f"Error reading {pf}: {e}")
            
    print(f"Sample Image Formats: {formats}")
    print(f"Sample Image Dimensions: {dimensions}")
    
    # Analyze YOLO class IDs and distribution
    class_distribution = {}
    total_annotations = 0
    images_with_labels = 0
    malformed_annotations = []
    
    for lf in label_files:
        with open(os.path.join(labels_dir, lf), 'r') as f:
            lines = f.readlines()
            if lines:
                images_with_labels += 1
            for line_num, line in enumerate(lines, 1):
                parts = line.strip().split()
                if not parts:
                    continue
                if len(parts) != 5:
                    malformed_annotations.append(f"{lf}:{line_num} -> {line.strip()}")
                    continue
                
                class_id = parts[0]
                total_annotations += 1
                class_distribution[class_id] = class_distribution.get(class_id, 0) + 1

    print(f"Total annotations: {total_annotations}")
    print(f"Number of annotated images: {images_with_labels}")
    print(f"Number of images without annotations (good/pass): {len(photo_files) - images_with_labels}")
    print(f"Class IDs found: {sorted(list(class_distribution.keys()))}")
    print(f"Class Distribution: {class_distribution}")
    
    if malformed_annotations:
        print(f"Found {len(malformed_annotations)} malformed annotations:")
        for ma in malformed_annotations[:5]:
            print(f"  - {ma}")
        if len(malformed_annotations) > 5:
            print(f"  ... and {len(malformed_annotations) - 5} more.")
    else:
        print("No malformed annotations found.")

if __name__ == '__main__':
    repo_root = os.path.dirname(os.path.abspath(__file__))
    analyze_dataset(repo_root)
