"""
=================================================================
通用数据集预处理脚本
=================================================================

【脚本用途】
将任意格式的YOLO/COCO数据集转换为统一的LoRA训练格式。
支持处理normal/defect分类，自动生成caption。

【功能特点】
1. 支持YOLO格式数据集（.txt标注）
2. 支持COCO格式数据集（.json标注）
3. 自动区分normal/defect样本
4. 自定义类别标签映射
5. 生成标准化caption

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
├── metadata.jsonl          # 每行一个JSON，包含file_name和text
├── normal/                 # 正常样本图片
│   ├── dataset1_normal_00001.png
│   └── dataset2_normal_00002.png
└── defect/                 # 缺陷样本图片
    ├── dataset1_defect_00001.png
    └── dataset2_defect_00002.png

【命令行参数】
必选参数:
  --dataset          数据集路径（支持YOLO或COCO格式）
  --label-map        缺陷类别映射，格式: "class_id:label,class_id:label"
                    例如: "0:crack,1:damage,2:erosion"
  --output-dir       输出目录

可选参数:
  --dataset-name    数据集名称（用于文件名标识），默认从路径推断
  --resolution      输出图片分辨率，默认512
  --max-normal       正常样本最大数量，默认3000
  --max-defect       缺陷样本最大数量，默认1000
  --train-split      训练集目录名，默认"train"
  --valid-split      验证集目录名，默认"valid"

【Caption生成规则】
- Normal样本: "a photo of a normal {material} {object}, clean and intact, no defects"
- Defect样本: "a photo of a {object} with {defect_types}, damaged area"

可以通过 --normal-caption 和 --defect-caption-template 覆盖

【使用方法】
示例1: 处理YOLO格式数据集
python prepare_datasets.py \
    --dataset /path/to/your/yolo_dataset \
    --label-map "0:crack,1:damage,2:erosion" \
    --output-dir ./output \
    --dataset-name mydataset

示例2: 处理COCO格式数据集
python prepare_datasets.py \
    --dataset /path/to/your/coco_dataset \
    --label-map "1:crack,2:damage,3:erosion" \
    --output-dir ./output \
    --dataset-name mydataset

示例3: 自定义caption模板
python prepare_datasets.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage" \
    --output-dir ./output \
    --normal-caption "normal turbine blade surface" \
    --defect-caption-template "blade with {defects}" \
    --resolution 512

示例4: 限制样本数量
python prepare_datasets.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage" \
    --output-dir ./output \
    --max-normal 1000 \
    --max-defect 500

【注意事项】
1. YOLO标注格式: class_id x_center y_center width height (归一化到0-1)
2. COCO标注格式: COCO JSON文件必须包含images和annotations字段
3. 如果数据集没有标注文件，所有图片都会被当作normal样本
4. 如果标注文件为空（没有缺陷），图片会被当作normal样本

【与其他脚本的关系】
本脚本输出用于 train_lora_txt2img.py（两阶段训练）
- normal/ 用于 Stage 1（学习叶片外观）
- defect/ 用于 Stage 2（学习缺陷特征）
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
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


def resize_image(img_path, resolution):
    """
    调整图片大小
    
    参数:
        img_path: 图片路径
        resolution: 目标分辨率
        
    返回:
        PIL.Image: 调整后的图片，失败返回None
    """
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        return img
    except Exception as e:
        print(f"  警告: 无法处理图片 {img_path}: {e}")
        return None


def process_yolo_dataset(dataset_path, label_map, args, metadata):
    """
    处理YOLO格式数据集
    
    参数:
        dataset_path: 数据集根目录
        label_map: 类别映射 dict{class_id: label}
        args: 命令行参数
        metadata: 元数据列表（会被修改）
        
    返回:
        tuple: (处理总数, 正常数量)
    """
    dataset_path = Path(dataset_path)
    count = 0
    normal_count = 0
    
    # 处理train和valid两个划分
    for split in [args.train_split, args.valid_split]:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"
        
        if not images_dir.exists():
            continue
            
        # 查找所有图片文件
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
            
        print(f"  {split}: 找到 {len(image_files)} 张图片")
        
        # 处理每张图片
        for img_path in tqdm(image_files, desc=f"  {args.dataset_name}/{split}"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            # 分析标注
            defect_types = set()
            is_normal = True
            
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                # YOLO格式: class_id x_center y_center width height
                                class_id = int(parts[0])
                                if class_id in label_map:
                                    defect_types.add(label_map[class_id])
                                    is_normal = False
                except Exception as e:
                    print(f"  警告: 无法读取标注 {label_path}: {e}")
                    
            # 确定保存路径和caption
            if is_normal:
                save_name = f"{args.dataset_name}_normal_{count:05d}.png"
                caption = args.normal_caption
            else:
                defects_str = " and ".join(sorted(defect_types))
                caption = args.defect_caption_template.format(
                    defects=defects_str
                )
                save_name = f"{args.dataset_name}_defect_{count:05d}.png"
                
            # 调整大小并保存
            img = resize_image(img_path, args.resolution)
            if img is None:
                continue
                
            save_path = Path(args.output_dir) / ("normal" if is_normal else "defect") / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path)
            
            # 记录元数据
            rel_path = str(save_path.relative_to(args.output_dir))
            metadata.append({
                "file_name": rel_path,
                "text": caption
            })
            
            if is_normal:
                normal_count += 1
            count += 1
            
    return count, normal_count


def process_coco_dataset(dataset_path, label_map, args, metadata):
    """
    处理COCO格式数据集
    
    参数:
        dataset_path: 数据集根目录
        label_map: 类别映射 dict{class_id: label}
        args: 命令行参数
        metadata: 元数据列表（会被修改）
        
    返回:
        tuple: (处理总数, 正常数量)
    """
    dataset_path = Path(dataset_path)
    count = 0
    normal_count = 0
    
    # 处理train和valid两个划分
    for split in [args.train_split, args.valid_split]:
        anno_path = dataset_path / split / "_annotations.coco.json"
        
        if not anno_path.exists():
            continue
            
        # 读取COCO JSON
        with open(anno_path, 'r') as f:
            coco_data = json.load(f)
            
        # 建立映射
        cat_id_to_name = {}
        for cat in coco_data.get("categories", []):
            if cat["id"] in label_map:
                cat_id_to_name[cat["id"]] = label_map[cat["id"]]
                
        img_id_to_info = {}
        for img_info in coco_data.get("images", []):
            img_id_to_info[img_info["id"]] = img_info
            
        # 按图片分组标注
        img_to_defects = defaultdict(set)
        for ann in coco_data.get("annotations", []):
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            if cat_id in label_map:
                img_to_defects[img_id].add(label_map[cat_id])
                
        print(f"  {split}: 找到 {len(img_id_to_info)} 张图片, {len(img_to_defects)} 张有标注")
        
        # 处理每张图片
        for img_id, img_info in tqdm(img_id_to_info.items(), desc=f"  {args.dataset_name}/{split}"):
            img_filename = img_info["file_name"]
            img_path = dataset_path / split / "images" / img_filename
            
            if not img_path.exists():
                continue
                
            # 分析缺陷
            defect_types = img_to_defects.get(img_id, set())
            is_normal = len(defect_types) == 0
            
            # 确定保存路径和caption
            if is_normal:
                save_name = f"{args.dataset_name}_normal_{count:05d}.png"
                caption = args.normal_caption
            else:
                defects_str = " and ".join(sorted(defect_types))
                caption = args.defect_caption_template.format(
                    defects=defects_str
                )
                save_name = f"{args.dataset_name}_defect_{count:05d}.png"
                
            # 调整大小并保存
            img = resize_image(img_path, args.resolution)
            if img is None:
                continue
                
            save_path = Path(args.output_dir) / ("normal" if is_normal else "defect") / save_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(save_path)
            
            # 记录元数据
            rel_path = str(save_path.relative_to(args.output_dir))
            metadata.append({
                "file_name": rel_path,
                "text": caption
            })
            
            if is_normal:
                normal_count += 1
            count += 1
            
    return count, normal_count


def balance_and_save(metadata, args):
    """
    类别平衡并保存元数据
    
    参数:
        metadata: 所有样本的元数据列表
        args: 命令行参数
    """
    print("\n" + "=" * 70)
    print("类别平衡处理...")
    print("=" * 70)
    
    # 分离normal和defect样本
    normal_entries = [e for e in metadata if "normal" in e["file_name"]]
    defect_entries = [e for e in metadata if "defect" in e["file_name"]]
    
    print(f"原始: 正常 {len(normal_entries)} 张, 缺陷 {len(defect_entries)} 张")
    
    # 随机采样normal样本
    if len(normal_entries) > args.max_normal:
        random.seed(42)
        normal_entries = random.sample(normal_entries, args.max_normal)
        print(f"  正常样本截取到 {args.max_normal} 张")
        
    # 按类型采样defect样本
    defect_counter = Counter()
    filtered_defect = []
    
    for entry in defect_entries:
        # 从caption提取缺陷类型
        defect_type = entry["text"].split("with ")[-1].split(",")[0].strip()
        if defect_counter[defect_type] < args.max_defect:
            filtered_defect.append(entry)
            defect_counter[defect_type] += 1
            
    defect_entries = filtered_defect
    print(f"  缺陷样本按类型截取，每种最多 {args.max_defect} 张")
    print(f"  缺陷类型分布: {dict(defect_counter)}")
    
    # 合并并保存
    balanced_metadata = normal_entries + defect_entries
    
    metadata_path = Path(args.output_dir) / "metadata.jsonl"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for entry in balanced_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
            
    print(f"\n最终: {len(balanced_metadata)} 张 (正常: {len(normal_entries)}, 缺陷: {len(defect_entries)})")
    print(f"元数据保存至: {metadata_path}")
    
    return len(normal_entries), len(defect_entries)


def main():
    parser = argparse.ArgumentParser(
        description="通用数据集预处理脚本 - YOLO/COCO转LoRA训练格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python prepare_datasets.py \\
      --dataset /path/to/yolo_dataset \\
      --label-map "0:crack,1:damage,2:erosion" \\
      --output-dir ./output \\
      --dataset-name mydataset
      
  python prepare_datasets.py \\
      --dataset /path/to/coco_dataset \\
      --label-map "1:crack,2:damage" \\
      --output-dir ./output \\
      --resolution 512
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
                        help="数据集名称（用于文件名标识），默认从路径推断")
    parser.add_argument("--resolution", type=int, default=512,
                        help="输出图片分辨率，默认512")
    parser.add_argument("--max-normal", type=int, default=3000,
                        help="正常样本最大数量，默认3000")
    parser.add_argument("--max-defect", type=int, default=1000,
                        help="每种缺陷类型最大数量，默认1000")
    parser.add_argument("--train-split", type=str, default="train",
                        help="训练集目录名，默认'train'")
    parser.add_argument("--valid-split", type=str, default="valid",
                        help="验证集目录名，默认'valid'")
    
    # Caption模板
    parser.add_argument("--normal-caption", type=str, 
                        default="a photo of a normal wind turbine blade surface, clean and intact, no defects",
                        help="正常样本的caption模板")
    parser.add_argument("--defect-caption-template", type=str,
                        default="a photo of a wind turbine blade surface with {defects}, damaged area",
                        help="缺陷样本的caption模板，{defects}会被替换为缺陷类型")
    
    args = parser.parse_args()
    
    # 解析标签映射
    label_map = parse_label_map(args.label_map)
    
    # 推断数据集名称
    if args.dataset_name is None:
        args.dataset_name = Path(args.dataset).name.replace(" ", "_")
        
    # 创建输出目录
    os.makedirs(Path(args.output_dir) / "normal", exist_ok=True)
    os.makedirs(Path(args.output_dir) / "defect", exist_ok=True)
    
    print("=" * 70)
    print("通用数据集预处理")
    print("=" * 70)
    print(f"数据集路径: {args.dataset}")
    print(f"数据集名称: {args.dataset_name}")
    print(f"输出目录: {args.output_dir}")
    print(f"分辨率: {args.resolution}")
    print(f"正常样本上限: {args.max_normal}")
    print(f"缺陷样本上限: {args.max_defect}")
    print(f"类别映射: {label_map}")
    print("=" * 70)
    
    metadata = []
    
    # 检测数据集格式并处理
    dataset_path = Path(args.dataset)
    
    # 检查是否为COCO格式
    coco_anno = dataset_path / args.train_split / "_annotations.coco.json"
    if coco_anno.exists():
        print(f"\n检测到COCO格式数据集")
        total_count, normal_count = process_coco_dataset(
            dataset_path, label_map, args, metadata
        )
    else:
        print(f"\n检测到YOLO格式数据集")
        total_count, normal_count = process_yolo_dataset(
            dataset_path, label_map, args, metadata
        )
        
    print(f"\n处理完成: 共 {total_count} 张 (正常: {normal_count}, 缺陷: {total_count - normal_count})")
    
    # 类别平衡并保存
    normal_final, defect_final = balance_and_save(metadata, args)
    
    print("\n" + "=" * 70)
    print("数据集处理完成!")
    print("=" * 70)
    print(f"输出目录: {args.output_dir}")
    print(f"  normal/  - {normal_final} 张正常叶片")
    print(f"  defect/ - {defect_final} 张缺陷叶片")
    print(f"  metadata.jsonl - {normal_final + defect_final} 条记录")
    print("=" * 70)
    print("\n可用于 train_lora_txt2img.py 进行LoRA训练")


if __name__ == "__main__":
    main()
