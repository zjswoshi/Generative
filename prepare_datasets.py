"""
数据集统一处理脚本
==================
将5个风机叶片数据集统一处理为LoRA微调格式

输出结构:
---------
processed_dataset/
├── metadata.jsonl          # 每行: {"file_name": "xxx.png", "text": "caption"}
├── normal/                 # 正常叶片图片
│   ├── nordtank_00001.png
│   └── ...
└── defect/                 # 缺陷叶片图片
    ├── fengji_crack_00001.png
    └── ...

使用方法:
--------
python prepare_datasets.py
python prepare_datasets.py --output-dir ./processed_dataset --resolution 512
"""

import os
import json
import glob
import argparse
import shutil
from collections import defaultdict, Counter

from PIL import Image
from tqdm import tqdm


DEFECT_LABEL_MAP = {
    "fengjiyepian": {
        0: "paint area",
        1: "contamination",
        2: "fracture",
        3: "surface damage",
        4: "debris",
        5: "hole",
        6: "crack",
        7: "adhesive area",
        8: "peeling",
        9: "groove",
    },
    "nordtank": {
        0: "dirt",
        1: "damage",
    },
    "offshore": {
        0: "burn damage",
        1: "coating damage",
        2: "contamination",
        3: "crack",
        4: "delamination",
        5: "erosion",
        6: "hole penetration",
        7: "scratch",
        8: "surface imperfection",
    },
    "blade_v7": {
        0: "crack",
    },
    "wind_turbine_v18": {
        1: "edge erosion",
        2: "lightning receptor damage",
        3: "surface damage",
        4: "vortex generator panel damage",
    },
}

NORMAL_CAPTION = "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects"
DEFECT_CAPTION_TEMPLATE = "a photo of a white fiberglass wind turbine blade surface with {defects}, damaged area"


def resize_and_save(img_path, save_path, resolution=512):
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((resolution, resolution), Image.Resampling.LANCZOS)
        img.save(save_path)
        return True
    except Exception as e:
        print(f"  警告: 无法处理 {img_path}: {e}")
        return False


def process_fengjiyepian(data_dir, output_dir, resolution, metadata):
    print("\n[1/5] 处理 fengjiyepian_506_v5 ...")
    base_dir = os.path.join(data_dir, "fengjiyepian_506_v5", "fengjiyepian_506_v5")

    label_map = DEFECT_LABEL_MAP["fengjiyepian"]
    count = 0
    normal_count = 0

    for split in ["train", "val"]:
        image_dir = os.path.join(base_dir, split, "images")
        label_dir = os.path.join(base_dir, split, "labels")

        if not os.path.isdir(image_dir):
            continue

        images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
        print(f"  {split}: 找到 {len(images)} 张图片")

        for img_path in tqdm(images, desc=f"  fengjiyepian/{split}"):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{basename}.txt")

            defect_types = set()
            is_normal = True

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            class_id = int(parts[0])
                            if class_id in label_map:
                                defect_types.add(label_map[class_id])
                                is_normal = False

            if is_normal:
                save_name = f"fengji_normal_{count:05d}.png"
                save_path = os.path.join(output_dir, "normal", save_name)
                caption = NORMAL_CAPTION
                normal_count += 1
            else:
                defects_str = " and ".join(sorted(defect_types))
                save_name = f"fengji_defect_{count:05d}.png"
                save_path = os.path.join(output_dir, "defect", save_name)
                caption = DEFECT_CAPTION_TEMPLATE.format(defects=defects_str)

            if resize_and_save(img_path, save_path, resolution):
                rel_path = os.path.join("normal" if is_normal else "defect", save_name)
                metadata.append({"file_name": rel_path, "text": caption})
                count += 1

    print(f"  完成: {count} 张 (正常: {normal_count}, 缺陷: {count - normal_count})")
    return count


def process_nordtank(data_dir, output_dir, resolution, metadata):
    print("\n[2/5] 处理 NordTank586x371 ...")
    base_dir = os.path.join(data_dir, "NordTank586x371")

    image_dir = os.path.join(base_dir, "images")
    label_dir = os.path.join(base_dir, "labels")

    label_map = DEFECT_LABEL_MAP["nordtank"]

    images = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    print(f"  找到 {len(images)} 张图片")

    label_files = set(
        os.path.splitext(f)[0]
        for f in os.listdir(label_dir)
        if f.endswith('.txt') and f != 'labels.txt'
    )

    count = 0
    normal_count = 0

    for img_path in tqdm(images, desc="  NordTank"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(label_dir, f"{basename}.txt")

        defect_types = set()
        is_normal = True

        if basename in label_files and os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0].isdigit():
                        class_id = int(parts[0])
                        if class_id in label_map:
                            defect_types.add(label_map[class_id])
                            is_normal = False

        if is_normal:
            save_name = f"nordtank_normal_{count:05d}.png"
            save_path = os.path.join(output_dir, "normal", save_name)
            caption = NORMAL_CAPTION
            normal_count += 1
        else:
            defects_str = " and ".join(sorted(defect_types))
            save_name = f"nordtank_defect_{count:05d}.png"
            save_path = os.path.join(output_dir, "defect", save_name)
            caption = DEFECT_CAPTION_TEMPLATE.format(defects=defects_str)

        if resize_and_save(img_path, save_path, resolution):
            rel_path = os.path.join("normal" if is_normal else "defect", save_name)
            metadata.append({"file_name": rel_path, "text": caption})
            count += 1

    print(f"  完成: {count} 张 (正常: {normal_count}, 缺陷: {count - normal_count})")
    return count


def process_offshore(data_dir, output_dir, resolution, metadata):
    print("\n[3/5] 处理 offshore wind turbine blade v3 ...")
    base_dir = os.path.join(data_dir, "offshore wind turbine blade.v3i.yolov11")

    label_map = DEFECT_LABEL_MAP["offshore"]

    count = 0
    normal_count = 0

    for split in ["train", "valid", "test"]:
        image_dir = os.path.join(base_dir, split, "images")
        label_dir = os.path.join(base_dir, split, "labels")

        if not os.path.isdir(image_dir):
            continue

        images = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg")) +
            glob.glob(os.path.join(image_dir, "*.png"))
        )
        print(f"  {split}: 找到 {len(images)} 张图片")

        for img_path in tqdm(images, desc=f"  offshore/{split}"):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{basename}.txt")

            defect_types = set()
            is_normal = True

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                if content:
                    for line in content.split('\n'):
                        parts = line.strip().split()
                        if parts and parts[0].isdigit():
                            class_id = int(parts[0])
                            if class_id in label_map:
                                defect_types.add(label_map[class_id])
                                is_normal = False

            if is_normal:
                save_name = f"offshore_normal_{count:05d}.png"
                save_path = os.path.join(output_dir, "normal", save_name)
                caption = NORMAL_CAPTION
                normal_count += 1
            else:
                defects_str = " and ".join(sorted(defect_types))
                save_name = f"offshore_defect_{count:05d}.png"
                save_path = os.path.join(output_dir, "defect", save_name)
                caption = DEFECT_CAPTION_TEMPLATE.format(defects=defects_str)

            if resize_and_save(img_path, save_path, resolution):
                rel_path = os.path.join("normal" if is_normal else "defect", save_name)
                metadata.append({"file_name": rel_path, "text": caption})
                count += 1

    print(f"  完成: {count} 张 (正常: {normal_count}, 缺陷: {count - normal_count})")
    return count


def process_blade_v7(data_dir, output_dir, resolution, metadata):
    print("\n[4/5] 处理 Wind turbine blade v7 ...")
    base_dir = os.path.join(data_dir, "Wind turbine blade.v7i.yolov11")

    label_map = DEFECT_LABEL_MAP["blade_v7"]

    count = 0
    normal_count = 0

    for split in ["train", "valid", "test"]:
        image_dir = os.path.join(base_dir, split, "images")
        label_dir = os.path.join(base_dir, split, "labels")

        if not os.path.isdir(image_dir):
            continue

        images = sorted(
            glob.glob(os.path.join(image_dir, "*.jpg")) +
            glob.glob(os.path.join(image_dir, "*.png"))
        )
        print(f"  {split}: 找到 {len(images)} 张图片")

        for img_path in tqdm(images, desc=f"  blade_v7/{split}"):
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(label_dir, f"{basename}.txt")

            defect_types = set()
            is_normal = True

            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                if content:
                    for line in content.split('\n'):
                        parts = line.strip().split()
                        if parts and parts[0].isdigit():
                            class_id = int(parts[0])
                            if class_id in label_map:
                                defect_types.add(label_map[class_id])
                                is_normal = False

            if is_normal:
                save_name = f"bladev7_normal_{count:05d}.png"
                save_path = os.path.join(output_dir, "normal", save_name)
                caption = NORMAL_CAPTION
                normal_count += 1
            else:
                defects_str = " and ".join(sorted(defect_types))
                save_name = f"bladev7_defect_{count:05d}.png"
                save_path = os.path.join(output_dir, "defect", save_name)
                caption = DEFECT_CAPTION_TEMPLATE.format(defects=defects_str)

            if resize_and_save(img_path, save_path, resolution):
                rel_path = os.path.join("normal" if is_normal else "defect", save_name)
                metadata.append({"file_name": rel_path, "text": caption})
                count += 1

    print(f"  完成: {count} 张 (正常: {normal_count}, 缺陷: {count - normal_count})")
    return count


def process_wind_turbine_v18(data_dir, output_dir, resolution, metadata):
    print("\n[5/5] 处理 Wind turbine v18 (COCO) ...")
    base_dir = os.path.join(data_dir, "Wind turbine.v18i.coco")

    label_map = DEFECT_LABEL_MAP["wind_turbine_v18"]

    count = 0
    normal_count = 0

    for split in ["train", "valid", "test"]:
        anno_path = os.path.join(base_dir, split, "_annotations.coco.json")
        if not os.path.exists(anno_path):
            continue

        with open(anno_path, 'r') as f:
            coco_data = json.load(f)

        cat_id_to_name = {}
        for cat in coco_data.get("categories", []):
            cat_id_to_name[cat["id"]] = cat["name"]

        img_to_defects = defaultdict(set)
        for ann in coco_data.get("annotations", []):
            cat_id = ann["category_id"]
            img_id = ann["image_id"]
            if cat_id in label_map:
                img_to_defects[img_id].add(label_map[cat_id])

        img_id_to_info = {}
        for img_info in coco_data.get("images", []):
            img_id_to_info[img_info["id"]] = img_info

        print(f"  {split}: 找到 {len(img_id_to_info)} 张图片, {len(img_to_defects)} 张有标注")

        for img_id, img_info in tqdm(img_id_to_info.items(), desc=f"  v18/{split}"):
            img_filename = img_info["file_name"]
            img_path = os.path.join(base_dir, split, img_filename)

            if not os.path.exists(img_path):
                continue

            defect_types = img_to_defects.get(img_id, set())
            is_normal = len(defect_types) == 0

            if is_normal:
                save_name = f"v18_normal_{count:05d}.png"
                save_path = os.path.join(output_dir, "normal", save_name)
                caption = NORMAL_CAPTION
                normal_count += 1
            else:
                defects_str = " and ".join(sorted(defect_types))
                save_name = f"v18_defect_{count:05d}.png"
                save_path = os.path.join(output_dir, "defect", save_name)
                caption = DEFECT_CAPTION_TEMPLATE.format(defects=defects_str)

            if resize_and_save(img_path, save_path, resolution):
                rel_path = os.path.join("normal" if is_normal else "defect", save_name)
                metadata.append({"file_name": rel_path, "text": caption})
                count += 1

    print(f"  完成: {count} 张 (正常: {normal_count}, 缺陷: {count - normal_count})")
    return count


def main():
    parser = argparse.ArgumentParser(description="统一处理风机叶片数据集")
    parser.add_argument("--data-dir", type=str, default="/home/cn/yolo/AnomalyAny",
                        help="包含所有数据集的根目录")
    parser.add_argument("--output-dir", type=str, default="/home/cn/yolo/AnomalyAny/processed_dataset",
                        help="处理后的输出目录")
    parser.add_argument("--resolution", type=int, default=512,
                        help="输出图片分辨率")
    parser.add_argument("--max-normal", type=int, default=3000,
                        help="正常样本最大数量(防止类别严重不平衡)")
    parser.add_argument("--max-defect-per-type", type=int, default=500,
                        help="每种缺陷类型最大数量")
    args = parser.parse_args()

    os.makedirs(os.path.join(args.output_dir, "normal"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "defect"), exist_ok=True)

    print("=" * 70)
    print("风机叶片数据集统一处理")
    print("=" * 70)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"分辨率: {args.resolution}")
    print(f"正常样本上限: {args.max_normal}")
    print(f"每种缺陷上限: {args.max_defect_per_type}")
    print("=" * 70)

    metadata = []

    process_fengjiyepian(args.data_dir, args.output_dir, args.resolution, metadata)
    process_nordtank(args.data_dir, args.output_dir, args.resolution, metadata)
    process_offshore(args.data_dir, args.output_dir, args.resolution, metadata)
    process_blade_v7(args.data_dir, args.output_dir, args.resolution, metadata)
    process_wind_turbine_v18(args.data_dir, args.output_dir, args.resolution, metadata)

    print("\n" + "=" * 70)
    print("类别平衡处理...")
    print("=" * 70)

    normal_entries = [e for e in metadata if "normal" in e["file_name"]]
    defect_entries = [e for e in metadata if "defect" in e["file_name"]]

    print(f"原始: 正常 {len(normal_entries)} 张, 缺陷 {len(defect_entries)} 张")

    if len(normal_entries) > args.max_normal:
        import random
        random.seed(42)
        normal_entries = random.sample(normal_entries, args.max_normal)
        print(f"  正常样本截取到 {args.max_normal} 张")

    defect_counter = Counter()
    filtered_defect = []
    for entry in defect_entries:
        defect_type = entry["text"].replace(DEFECT_CAPTION_TEMPLATE.split("{defects}")[0], "").replace(", damaged area", "")
        if defect_counter[defect_type] < args.max_defect_per_type:
            filtered_defect.append(entry)
            defect_counter[defect_type] += 1
    defect_entries = filtered_defect
    print(f"  缺陷样本按类型截取到每种最多 {args.max_defect_per_type} 张")
    print(f"  缺陷类型分布: {dict(defect_counter)}")

    balanced_metadata = normal_entries + defect_entries

    metadata_path = os.path.join(args.output_dir, "metadata.jsonl")
    with open(metadata_path, 'w') as f:
        for entry in balanced_metadata:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"\n最终: {len(balanced_metadata)} 张 (正常: {len(normal_entries)}, 缺陷: {len(defect_entries)})")
    print(f"元数据保存至: {metadata_path}")

    print("\n" + "=" * 70)
    print("数据集处理完成!")
    print("=" * 70)
    print(f"输出目录: {args.output_dir}")
    print(f"  normal/  - {len(normal_entries)} 张正常叶片")
    print(f"  defect/  - {len(defect_entries)} 张缺陷叶片")
    print(f"  metadata.jsonl - {len(balanced_metadata)} 条记录")
    print("=" * 70)


if __name__ == "__main__":
    main()
