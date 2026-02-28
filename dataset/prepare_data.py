import os
import shutil
import random
from pathlib import Path

# Config
dataset_dir = Path("dataset/garbage_classification")
output_dir = Path("dataset/yolo_format")
split_ratio = 0.8 # 80% train, 20% val

# Create directories
for split in ['train', 'val']:
    for kind in ['images', 'labels']:
        (output_dir / split / kind).mkdir(parents=True, exist_ok=True)

# Get classes
classes = sorted([d.name for d in dataset_dir.iterdir() if d.is_dir()])
print(f"Classes found: {classes}")

# Process each class
for class_id, class_name in enumerate(classes):
    print(f"Processing {class_name}...")
    
    # Get all images
    images = list((dataset_dir / class_name).glob("*"))
    random.shuffle(images)
    
    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]
    
    # Copy images and create labels
    for img_path, subset in [(train_imgs, 'train'), (val_imgs, 'val')]:
        for img in img_path:
            # Copy image
            dest_img_path = output_dir / subset / 'images' / img.name
            shutil.copy(img, dest_img_path)
            
            # Create label file (YOLO format: class_id center_x center_y width height)
            # Since this is classification dataset (no bbox), we use full image as bbox?
            # NO! YOLO classification (v8-cls) uses folder structure: train/class_name/img.jpg
            # YOLO detection (v8-det) requires bounding boxes.
            
            # IF using YOLOv8-cls (Classification):
            # Structure: dataset/train/class_name/image.jpg
            
            # IF using YOLOv8-det (Detection):
            # We need labels! But we don't have them.
            # We can fake detection by saying the whole image is the object.
            # bbox: 0.5 0.5 1.0 1.0 (center_x, center_y, width, height)
            
            label_path = output_dir / subset / 'labels' / (img.stem + ".txt")
            with open(label_path, 'w') as f:
                 f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("Data splitting complete.")
print(f"Data prepared at: {output_dir}")
