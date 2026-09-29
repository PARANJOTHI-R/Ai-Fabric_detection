import os
import hashlib
from collections import defaultdict
from PIL import Image

DATASET_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Fabric Defects Dataset")

def inspect_dataset():
    print("=" * 60)
    print("FABRIC DEFECTS DATASET INSPECTION")
    print("=" * 60)
    print(f"Dataset root: {DATASET_ROOT}")

    if not os.path.exists(DATASET_ROOT):
        # Check alternate spelling if any
        alt = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Fabric Defect Dataset")
        if os.path.exists(alt):
            dataset_path = alt
        else:
            print("ERROR: Dataset directory not found!")
            return
    else:
        dataset_path = DATASET_ROOT

    entries = sorted(os.listdir(dataset_path))
    class_folders = [e for e in entries if os.path.isdir(os.path.join(dataset_path, e))]
    print(f"Top-level folder entries: {entries}")
    print(f"Class folders ({len(class_folders)}): {class_folders}")

    total_images = 0
    class_counts = {}
    format_counts = defaultdict(int)
    dimensions_set = set()
    dimensions_summary = defaultdict(int)
    hashes = {}
    duplicates = []
    corrupted_images = []
    annotation_files = []

    for folder in class_folders:
        folder_path = os.path.join(dataset_path, folder)
        folder_files = os.listdir(folder_path)
        img_count = 0

        for f in folder_files:
            file_path = os.path.join(folder_path, f)
            if os.path.isdir(file_path):
                print(f"  Warning: Nested directory found in {folder}: {f}")
                continue

            ext = os.path.splitext(f)[1].lower()
            if ext in ['.txt', '.xml', '.json', '.csv']:
                annotation_files.append((folder, f))
                continue

            # Verify image file
            try:
                with Image.open(file_path) as img:
                    img.verify()
                
                # Reopen to check dimensions and format because verify closes or restricts
                with Image.open(file_path) as img:
                    fmt = img.format or ext.replace('.', '').upper()
                    w, h = img.size
                    format_counts[fmt] += 1
                    dimensions_set.add((w, h))
                    dimensions_summary[(w, h)] += 1
                    
                    # Check MD5 hash for exact duplication
                    with open(file_path, "rb") as bf:
                        file_hash = hashlib.md5(bf.read()).hexdigest()
                    if file_hash in hashes:
                        duplicates.append((file_path, hashes[file_hash]))
                    else:
                        hashes[file_hash] = file_path

                img_count += 1
            except Exception as e:
                corrupted_images.append((file_path, str(e)))

        class_counts[folder] = img_count
        total_images += img_count

    print("\n" + "-" * 40)
    print("DATASET SUMMARY")
    print("-" * 40)
    print(f"Total Images: {total_images}")
    print("\nClasses and Counts:")
    for cls, count in class_counts.items():
        print(f"  {cls:<15} -> {count}")

    print("\nImage Formats:")
    for fmt, count in format_counts.items():
        print(f"  {fmt}: {count}")

    print(f"\nUnique Dimensions ({len(dimensions_set)} unique sizes):")
    for dim, count in sorted(dimensions_summary.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {dim[0]}x{dim[1]}: {count} images")

    print(f"\nAnnotation Files Found: {len(annotation_files)}")
    if annotation_files:
        for folder, f in annotation_files[:10]:
            print(f"  {folder}/{f}")
    else:
        print("  None. Pure class-folder based classification dataset.")

    print(f"\nCorrupted Images: {len(corrupted_images)}")
    if corrupted_images:
        for path, err in corrupted_images:
            print(f"  {path}: {err}")

    print(f"Exact Duplicate Images: {len(duplicates)}")
    if duplicates:
        for f1, f2 in duplicates[:5]:
            print(f"  {os.path.basename(f1)} is identical to {os.path.basename(f2)}")

    print("\nDataset Structure Type:")
    if len(annotation_files) == 0 and len(class_folders) > 0:
        print("  -> Pure Image Classification (directory-per-class, NO bounding boxes)")
    else:
        print("  -> Object Detection / Annotated dataset")
    print("=" * 60)

if __name__ == "__main__":
    inspect_dataset()
