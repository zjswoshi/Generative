#!/usr/bin/env python3
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

random.seed(42)

DATA_ROOT = Path("/home/cn/yolo/AnomalyAny/fengjiyepian260310_v17")

print("Step 1: Collecting all images from train and val...")
train_images = sorted([f for f in (DATA_ROOT / "train/images").iterdir() 
                       if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
val_images = sorted([f for f in (DATA_ROOT / "val/images").iterdir() 
                     if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])

print(f"  Train images: {len(train_images)}")
print(f"  Val images: {len(val_images)}")

all_images = train_images + val_images
print(f"  Total: {len(all_images)} images")

print("\nStep 2: Analyzing class distribution for each image...")
image_classes = {}
for img_path in all_images:
    label_path = img_path.with_suffix('.txt')
    label_path = DATA_ROOT / "train" / "labels" / label_path.name if "train" in str(img_path) else label_path
    
    classes_in_img = set()
    if label_path.exists():
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    classes_in_img.add(int(parts[0]))
    
    image_classes[img_path] = classes_in_img

print(f"  Analyzed {len(image_classes)} images")

print("\nStep 3: Stratified splitting (80:10:10) based on class distribution...")

train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

images_by_class = defaultdict(list)
for img_path, classes in image_classes.items():
    if classes:
        for cls in classes:
            images_by_class[cls].append(img_path)
    else:
        images_by_class[-1].append(img_path)

train_set = []
val_set = []
test_set = []
used_images = set()

for cls, images in sorted(images_by_class.items()):
    random.shuffle(images)
    
    n_train = int(len(images) * train_ratio)
    n_val = int(len(images) * test_ratio)
    
    for img in images[:n_train]:
        if img not in used_images:
            train_set.append(img)
            used_images.add(img)
    
    for img in images[n_train:n_train+n_val]:
        if img not in used_images:
            val_set.append(img)
            used_images.add(img)
    
    for img in images[n_train+n_val:]:
        if img not in used_images:
            test_set.append(img)
            used_images.add(img)

final_train = [img for img in all_images if img not in val_set and img not in test_set]
final_val = val_set
final_test = test_set

random.shuffle(final_train)
random.shuffle(final_val)
random.shuffle(final_test)

print(f"\nSplit result:")
print(f"  Train: {len(final_train)} ({len(final_train)/len(all_images)*100:.1f}%)")
print(f"  Val:   {len(final_val)} ({len(final_val)/len(all_images)*100:.1f}%)")
print(f"  Test:  {len(final_test)} ({len(final_test)/len(all_images)*100:.1f}%)")

print("\nStep 4: Creating new directory structure...")
for subset in ['train', 'val', 'test']:
    (DATA_ROOT / subset / "images").mkdir(parents=True, exist_ok=True)
    (DATA_ROOT / subset / "labels").mkdir(parents=True, exist_ok=True)

def get_label_path(img_path, src_folder):
    label_name = img_path.stem + ".txt"
    if src_folder == "train":
        return DATA_ROOT / "train" / "labels" / label_name
    else:
        return DATA_ROOT / "val" / "labels" / label_name

print("\nStep 5: Moving files to new structure...")

subsets = {'train': final_train, 'val': final_val, 'test': final_test}

for subset_name, images in subsets.items():
    print(f"  Moving {subset_name} ({len(images)} files)...")
    moved = 0
    
    for img_path in images:
        src_folder = "train" if "train" in str(img_path) else "val"
        
        dst_img = DATA_ROOT / subset_name / "images" / img_path.name
        dst_label = DATA_ROOT / subset_name / "labels" / f"{img_path.stem}.txt"
        
        shutil.move(str(img_path), str(dst_img))
        
        label_path = get_label_path(img_path, src_folder)
        if label_path.exists():
            shutil.move(str(label_path), str(dst_label))
        else:
            open(dst_label, 'w').close()
        
        moved += 1
        if moved % 1000 == 0:
            print(f"    Moved {moved}/{len(images)}")

print("\n=== Final Summary ===")
for subset in ['train', 'val', 'test']:
    n_images = len(list((DATA_ROOT / subset / "images").iterdir()))
    n_labels = len(list((DATA_ROOT / subset / "labels").iterdir()))
    print(f"{subset}: {n_images} images, {n_labels} labels")

print("\nDone! Dataset has been resplit to 80:10:10 ratio with stratified sampling.")