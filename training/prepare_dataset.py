import os
import shutil
import random
import yaml
from pathlib import Path

def prepare_dataset(base_dir):
    random.seed(42)
    
    photos_dir = os.path.join(base_dir, 'photos')
    labels_dir = os.path.join(base_dir, 'labels')
    
    dataset_dir = os.path.join(base_dir, 'dataset')
    
    # Create dataset directories
    dirs = [
        os.path.join(dataset_dir, 'images', 'train'),
        os.path.join(dataset_dir, 'images', 'val'),
        os.path.join(dataset_dir, 'labels', 'train'),
        os.path.join(dataset_dir, 'labels', 'val')
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    images = [f for f in os.listdir(photos_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Shuffle for random split
    random.shuffle(images)
    
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    def copy_files(img_list, split):
        for img_name in img_list:
            # Copy image
            src_img = os.path.join(photos_dir, img_name)
            dst_img = os.path.join(dataset_dir, 'images', split, img_name)
            shutil.copy2(src_img, dst_img)
            
            # Copy label
            base_name = os.path.splitext(img_name)[0]
            label_name = base_name + '.txt'
            src_label = os.path.join(labels_dir, label_name)
            dst_label = os.path.join(dataset_dir, 'labels', split, label_name)
            
            if os.path.exists(src_label):
                shutil.copy2(src_label, dst_label)
            else:
                # Create empty label if no defects
                Path(dst_label).touch()

    copy_files(train_images, 'train')
    copy_files(val_images, 'val')
    
    print(f"Copied {len(train_images)} images/labels to train")
    print(f"Copied {len(val_images)} images/labels to val")
    
    # Create data.yaml
    classes = ['hole', 'objects', 'oil spot', 'thread error']
    
    data_yaml = {
        'path': dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {i: name for i, name in enumerate(classes)}
    }
    
    yaml_path = os.path.join(dataset_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
        
    print(f"Created {yaml_path}")

if __name__ == '__main__':
    prepare_dataset('d:/PROJECTS/AI-Fabric-Defect-Detection')
