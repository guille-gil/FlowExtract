#!/usr/bin/env python3
"""
Reorganize YOLO training data into clean train/val split.

Current structure:
- yolo_annotations/images/ - 24 images with UUID prefixes
- yolo_annotations/labels/ - 24 labels with UUID prefixes  
- tocaps/ - 24 clean images without UUIDs

Target structure:
- yolo_annotations/
  ├── images/
  │   ├── train/ (19 images, 80%)
  │   └── val/ (5 images, 20%)
  ├── labels/
  │   ├── train/ (19 labels, 80%)
  │   └── val/ (5 labels, 20%)
  └── yolo_data.yaml
"""

import os
import shutil
from pathlib import Path
import random

# Set seed for reproducibility
random.seed(42)

# Paths
base_dir = Path("data/input/yolo_annotations")
images_dir = base_dir / "images"
labels_dir = base_dir / "labels"

# Get all image files
image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.png')])
print(f"Found {len(image_files)} images")

# Create new directory structure
for split in ['train', 'val']:
    (images_dir / split).mkdir(exist_ok=True)
    (labels_dir / split).mkdir(exist_ok=True)

# Shuffle and split (80/20)
random.shuffle(image_files)
split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

print(f"Train: {len(train_files)} images")
print(f"Val: {len(val_files)} images")

# Move files
for img_file in train_files:
    # Get corresponding label file
    label_file = img_file.replace('.png', '.txt')
    
    # Move to train
    shutil.move(images_dir / img_file, images_dir / 'train' / img_file)
    shutil.move(labels_dir / label_file, labels_dir / 'train' / label_file)

for img_file in val_files:
    # Get corresponding label file
    label_file = img_file.replace('.png', '.txt')
    
    # Move to val
    shutil.move(images_dir / img_file, images_dir / 'val' / img_file)
    shutil.move(labels_dir / label_file, labels_dir / 'val' / label_file)

print("\n✅ Reorganization complete!")
print(f"Train: {len(os.listdir(images_dir / 'train'))} images")
print(f"Val: {len(os.listdir(images_dir / 'val'))} images")
