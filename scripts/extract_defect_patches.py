"""
=================================================================
通用缺陷Patch提取脚本
=================================================================

【脚本用途】
从任意YOLO/COCO标注数据集中提取缺陷区域（带mask）作为训练样本。
生成标准化的缺陷patch，可用于img2img训练。

【功能特点】
1. 支持YOLO格式数据集（.txt标注）
2. 支持COCO格式数据集（.json标注）
3. 自动裁剪缺陷区域并添加周围上下文
4. 生成标准化的patch和mask
5. 支持批量处理多个数据集

【输入数据格式】
支持以下两种格式：

格式1: YOLO格式
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── img002.png
│   └── labels/
│       ├── img001.txt
│       └── img002.txt
└── valid/
    └── ...

格式2: COCO格式
dataset/
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   └── img002.png
│   └── _annotations.coco.json
└── valid/
    └── ...

【输出格式】
output_dir/
├── dataset_name/
│   ├── train/
│   │   ├── images/           # 裁剪的缺陷patch
│   │   │   ├── img001_defect0_crack.png
│   │   │   └── img002_defect1_damage.png
│   │   └── masks/            # 对应的mask
│   │       ├── img001_defect0_crack.png
│   │       └── img002_defect1_damage.png
│   └── valid/
│       └── ...
└── metadata.json              # 元数据

【命令行参数】
必选参数:
  --dataset          数据集路径（支持YOLO或COCO格式）
  --label-map        缺陷类别映射，格式: "class_id:label,class_id:label"
                    例如: "0:crack,1:damage,2:erosion"
  --output-dir       输出目录

可选参数:
  --dataset-name    数据集名称，默认从路径推断
  --patch-size      patch大小，默认256
  --context-padding 缺陷周围上下文padding比例，默认0.3（30%）
  --train-split     训练集目录名，默认"train"
  --valid-split     验证集目录名，默认"valid"
  --test-split      测试集目录名，默认"test"

【Patch提取说明】
1. 裁剪策略:
   - 以缺陷bbox为中心
   - 添加周围上下文区域（默认30%）
   - 保持宽高比
   
2. Mask生成:
   - 生成与patch同样大小的mask
   - 缺陷区域填充为白色(255)
   - 背景为黑色(0)
   
3. 文件命名:
   - 格式: {原图名}_defect{序号}_{类别名}.png
   - 例如: img001_defect0_crack.png

【使用方法】
示例1: 处理YOLO格式数据集
python extract_defect_patches.py \
    --dataset /path/to/your/yolo_dataset \
    --label-map "0:crack,1:damage,2:erosion" \
    --output-dir ./defect_patches \
    --dataset-name mydataset

示例2: 处理COCO格式数据集
python extract_defect_patches.py \
    --dataset /path/to/your/coco_dataset \
    --label-map "1:crack,2:damage" \
    --output-dir ./defect_patches \
    --dataset-name mydataset

示例3: 自定义参数
python extract_defect_patches.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage" \
    --output-dir ./output \
    --patch-size 256 \
    --context-padding 0.5

【注意事项】
1. YOLO标注格式: class_id x_center y_center width height (归一化到0-1)
2. COCO标注格式: COCO JSON文件必须包含images和annotations字段
3. context-padding=0.3表示在缺陷周围添加30%的上下文区域
4. patch会resize到指定大小，默认256x256
5. 只处理有标注的图片，无标注的图片会被忽略

【与其他脚本的关系】
本脚本输出用于 train_lora_img2img.py（单阶段缺陷patch训练）
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm


def parse_label_map(label_map_str):
    """
    解析标签映射字符串
    
    参数:
        label_map_str: 格式 "0:crack,1:damage,2:erosion"
        
    返回:
        dict: {class_id: label_name}
    """
    label_map = {}
    pairs = label_map_str.split(',')
    for pair in pairs:
        if ':' in pair:
            class_id, label = pair.split(':', 1)
            label_map[int(class_id.strip())] = label.strip()
    return label_map


def extract_patch_with_context(img, bbox, context_padding, patch_size):
    """
    从图片中提取带上下文的patch及其mask
    
    参数:
        img: PIL.Image原始图片
        bbox: tuple (x1, y1, x2, y2) 像素坐标
        context_padding: float 上下文padding比例
        patch_size: int 输出patch大小
        
    返回:
        tuple: (patch, mask) 或 (None, None)如果无效
    """
    img_w, img_h = img.size
    x1, y1, x2, y2 = bbox
    
    patch_width = x2 - x1
    patch_height = y2 - y1
    
    if patch_width <= 0 or patch_height <= 0:
        return None, None
        
    # 添加上下文padding
    pad_w = int(patch_width * context_padding)
    pad_h = int(patch_height * context_padding)
    
    x1_pad = max(0, x1 - pad_w)
    y1_pad = max(0, y1 - pad_h)
    x2_pad = min(img_w, x2 + pad_w)
    y2_pad = min(img_h, y2 + pad_h)
    
    if x2_pad <= x1_pad or y2_pad <= y1_pad:
        return None, None
        
    # 裁剪patch
    patch = img.crop((x1_pad, y1_pad, x2_pad, y2_pad))
    patch_resized = patch.resize((patch_size, patch_size), Image.Resampling.LANCZOS)
    
    # 生成mask
    mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
    
    defect_ratio_x = patch_size / (x2 - x1) if x2 > x1 else 1
    defect_ratio_y = patch_size / (y2 - y1) if y2 > y1 else 1
    
    # 计算缺陷在resize后patch中的位置
    mask_x1 = int(pad_w * defect_ratio_x)
    mask_y1 = int(pad_h * defect_ratio_y)
    mask_x2 = int((pad_w + (x2 - x1)) * defect_ratio_x)
    mask_y2 = int((pad_h + (y2 - y1)) * defect_ratio_y)
    
    # 填充缺陷区域
    mask[mask_y1:mask_y2, mask_x1:mask_x2] = 255
    
    return patch_resized, Image.fromarray(mask)


def process_yolo_dataset(dataset_path, label_map, args, metadata):
    """
    从YOLO格式数据集提取缺陷patch
    
    参数:
        dataset_path: 数据集根目录
        label_map: 类别映射
        args: 命令行参数
        metadata: 元数据列表
        
    返回:
        int: 提取的patch总数
    """
    dataset_path = Path(dataset_path)
    total_extracted = 0
    
    # 处理train、valid、test三个划分
    for split in [args.train_split, args.valid_split, args.test_split]:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"
        
        if not images_dir.exists():
            continue
            
        # 查找所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
            
        print(f"  {split}: 处理 {len(image_files)} 张图片...")
        
        # 创建输出目录
        output_images = Path(args.output_dir) / args.dataset_name / split / "images"
        output_masks = Path(args.output_dir) / args.dataset_name / split / "masks"
        output_images.mkdir(parents=True, exist_ok=True)
        output_masks.mkdir(parents=True, exist_ok=True)
        
        # 处理每张图片
        for img_path in tqdm(image_files, desc=f"  {args.dataset_name}/{split}"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
                
            try:
                img = Image.open(img_path)
                img_w, img_h = img.size
            except Exception as e:
                print(f"  错误: 无法打开 {img_path}: {e}")
                continue
                
            # 解析YOLO标注
            bboxes = []
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            if class_id not in label_map:
                                continue
                                
                            cx, cy, w, h = map(float, parts[1:5])
                            # 转换为像素坐标
                            x1 = int((cx - w/2) * img_w)
                            y1 = int((cy - h/2) * img_h)
                            x2 = int((cx + w/2) * img_w)
                            y2 = int((cy + h/2) * img_h)
                            
                            bboxes.append({
                                'class_id': class_id,
                                'class_name': label_map[class_id],
                                'bbox': [x1, y1, x2, y2]
                            })
            except Exception as e:
                print(f"  错误: 无法读取标注 {label_path}: {e}")
                continue
                
            # 提取每个缺陷patch
            for i, bbox_info in enumerate(bboxes):
                x1, y1, x2, y2 = bbox_info['bbox']
                class_name = bbox_info['class_name']
                
                patch, mask = extract_patch_with_context(
                    img, (x1, y1, x2, y2),
                    args.context_padding, args.patch_size
                )
                
                if patch is None:
                    continue
                    
                # 保存patch和mask
                patch_name = f"{img_path.stem}_defect{i}_{class_name}.png"
                patch.save(output_images / patch_name)
                mask.save(output_masks / patch_name)
                
                # 记录元数据
                metadata.append({
                    'image': str(output_images / patch_name),
                    'mask': str(output_masks / patch_name),
                    'source': args.dataset_name,
                    'split': split,
                    'class_name': class_name,
                    'original_bbox': [x1, y1, x2, y2]
                })
                
                total_extracted += 1
                
    return total_extracted


def process_coco_dataset(dataset_path, label_map, args, metadata):
    """
    从COCO格式数据集提取缺陷patch
    
    参数:
        dataset_path: 数据集根目录
        label_map: 类别映射
        args: 命令行参数
        metadata: 元数据列表
        
    返回:
        int: 提取的patch总数
    """
    dataset_path = Path(dataset_path)
    total_extracted = 0
    
    # 处理train、valid、test三个划分
    for split in [args.train_split, args.valid_split, args.test_split]:
        anno_path = dataset_path / split / "_annotations.coco.json"
        
        if not anno_path.exists():
            continue
            
        # 读取COCO JSON
        with open(anno_path, 'r') as f:
            coco_data = json.load(f)
            
        # 建立映射
        img_id_to_info = {}
        for img_info in coco_data.get("images", []):
            img_id_to_info[img_info["id"]] = img_info
            
        # 按图片分组标注
        ann_by_img = defaultdict(list)
        for ann in coco_data.get("annotations", []):
            img_id = ann["image_id"]
            cat_id = ann["category_id"]
            if cat_id not in label_map:
                continue
            ann_by_img[img_id].append({
                'bbox': ann['bbox'],
                'category_id': cat_id,
                'class_name': label_map[cat_id]
            })
            
        print(f"  {split}: 处理 {len(ann_by_img)} 张有缺陷的图片...")
        
        # 创建输出目录
        output_images = Path(args.output_dir) / args.dataset_name / split / "images"
        output_masks = Path(args.output_dir) / args.dataset_name / split / "masks"
        output_images.mkdir(parents=True, exist_ok=True)
        output_masks.mkdir(parents=True, exist_ok=True)
        
        # 处理每张图片
        for img_id, anns in tqdm(ann_by_img.items(), desc=f"  {args.dataset_name}/{split}"):
            img_info = img_id_to_info.get(img_id)
            if not img_info:
                continue
                
            img_filename = img_info["file_name"]
            img_path = dataset_path / split / "images" / img_filename
            
            if not img_path.exists():
                continue
                
            try:
                img = Image.open(img_path)
            except Exception as e:
                print(f"  错误: 无法打开 {img_path}: {e}")
                continue
                
            # 提取每个缺陷patch
            for i, ann in enumerate(anns):
                x, y, w, h = ann['bbox']
                x1 = int(x)
                y1 = int(y)
                x2 = int(x + w)
                y2 = int(y + h)
                class_name = ann['class_name']
                
                patch, mask = extract_patch_with_context(
                    img, (x1, y1, x2, y2),
                    args.context_padding, args.patch_size
                )
                
                if patch is None:
                    continue
                    
                # 保存patch和mask
                patch_name = f"{Path(img_filename).stem}_defect{i}_{class_name}.png"
                patch.save(output_images / patch_name)
                mask.save(output_masks / patch_name)
                
                # 记录元数据
                metadata.append({
                    'image': str(output_images / patch_name),
                    'mask': str(output_masks / patch_name),
                    'source': args.dataset_name,
                    'split': split,
                    'class_name': class_name,
                    'original_bbox': [x1, y1, x2, y2]
                })
                
                total_extracted += 1
                
    return total_extracted


def main():
    parser = argparse.ArgumentParser(
        description="通用缺陷Patch提取脚本 - YOLO/COCO数据集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python extract_defect_patches.py \\
      --dataset /path/to/yolo_dataset \\
      --label-map "0:crack,1:damage,2:erosion" \\
      --output-dir ./defect_patches \\
      --dataset-name mydataset
      
  python extract_defect_patches.py \\
      --dataset /path/to/coco_dataset \\
      --label-map "1:crack,2:damage" \\
      --output-dir ./defect_patches \\
      --patch-size 256 \\
      --context-padding 0.5
        """
    )
    
    # 必选参数
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集路径（支持YOLO或COCO格式）")
    parser.add_argument("--label-map", type=str, required=True,
                        help="缺陷类别映射，格式: 'class_id:label,class_id:label'，例如: '0:crack,1:damage'")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")
    
    # 可选参数
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="数据集名称，默认从路径推断")
    parser.add_argument("--patch-size", type=int, default=256,
                        help="patch大小，默认256")
    parser.add_argument("--context-padding", type=float, default=0.3,
                        help="缺陷周围上下文padding比例，默认0.3（30%%）")
    parser.add_argument("--train-split", type=str, default="train",
                        help="训练集目录名，默认'train'")
    parser.add_argument("--valid-split", type=str, default="valid",
                        help="验证集目录名，默认'valid'")
    parser.add_argument("--test-split", type=str, default="test",
                        help="测试集目录名，默认'test'")
    
    args = parser.parse_args()
    
    # 解析标签映射
    label_map = parse_label_map(args.label_map)
    
    # 推断数据集名称
    if args.dataset_name is None:
        args.dataset_name = Path(args.dataset).name.replace(" ", "_")
    
    print("=" * 70)
    print("通用缺陷Patch提取")
    print("=" * 70)
    print(f"数据集路径: {args.dataset}")
    print(f"数据集名称: {args.dataset_name}")
    print(f"输出目录: {args.output_dir}")
    print(f"Patch大小: {args.patch_size}x{args.patch_size}")
    print(f"上下文padding: {args.context_padding} ({args.context_padding*100}%%)")
    print(f"类别映射: {label_map}")
    print("=" * 70)
    
    metadata = []
    
    # 检测数据集格式并处理
    dataset_path = Path(args.dataset)
    
    # 检查是否为COCO格式
    coco_anno = dataset_path / args.train_split / "_annotations.coco.json"
    if coco_anno.exists():
        print(f"\n检测到COCO格式数据集")
        total_extracted = process_coco_dataset(
            dataset_path, label_map, args, metadata
        )
    else:
        print(f"\n检测到YOLO格式数据集")
        total_extracted = process_yolo_dataset(
            dataset_path, label_map, args, metadata
        )
    
    # 保存元数据
    metadata_path = Path(args.output_dir) / args.dataset_name / 'metadata.json'
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({'samples': metadata}, f, indent=2, ensure_ascii=False)
        
    print(f"\n元数据保存至: {metadata_path}")
    print("=" * 70)
    
    # 统计各类别数量
    class_counts = {}
    for m in metadata:
        class_name = m['class_name']
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
    print("各类别统计:")
    for name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {name}: {count}")
    
    print("\n" + "=" * 70)
    print(f"提取完成! 共提取 {total_extracted} 个缺陷patch")
    print("=" * 70)
    print("\n可用于 train_lora_img2img.py 进行缺陷LoRA训练")


if __name__ == "__main__":
    main()
