"""
=================================================================
数据集类别均衡脚本
=================================================================

【脚本用途】
自动诊断数据集类别分布，识别少数类，
调用缺陷生成脚本补充样本，使类别分布均衡。

【功能特点】
1. 自动分析数据集类别分布
2. 识别样本数量过少的类别
3. 调用生成脚本补充少数类样本
4. 支持多种生成管道：txt2img、img2img

【工作流程】
1. 诊断：分析数据集各类别样本数量
2. 规划：根据目标分布计算需要生成的样本数
3. 生成：调用缺陷生成脚本补充少数类
4. 验证：检查生成后的类别分布

【命令行参数】
必选参数:
  --dataset          数据集路径（YOLO或COCO格式）
  --label-map       类别映射，格式: "class_id:label,class_id:label"

可选参数:
  --output-dir      输出目录，默认"./balanced_dataset"
  --target-ratio   目标类别比例，默认1.0（各类别样本数相同）
  --min-samples    每类最小样本数，默认100
  --generate-mode   生成模式: txt2img或img2img
  --pipe            管道: sd或anomalyany
  --lora-path       LoRA路径

【使用方法】

示例1: 分析并生成（使用img2img）
python balance_dataset.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage,2:erosion" \
    --generate-mode img2img \
    --output-dir ./balanced

示例2: 只分析不生成
python balance_dataset.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage" \
    --output-dir ./balanced \
    --dry-run

示例3: 使用AnomalyAny管道
python balance_dataset.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage" \
    --generate-mode img2img \
    --pipe anomalyany \
    --lora-path ./outputs/lora_stage2_defect/final \
    --output-dir ./balanced

【注意事项】
1. 需要先生成LoRA权重
2. img2img模式需要提供mask，可以通过extract_defect_patches.py生成
3. txt2img模式直接从文字生成，需要较大样本数可能较慢
"""

import os
import json
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
import subprocess


def parse_label_map(label_map_str):
    """
    解析类别映射字符串
    
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


def analyze_dataset(dataset_path, label_map, args):
    """
    分析数据集类别分布
    
    参数:
        dataset_path: 数据集路径
        label_map: 类别映射
        args: 命令行参数
        
    返回:
        dict: {class_id: sample_count}
    """
    dataset_path = Path(dataset_path)
    class_counts = defaultdict(int)
    
    # 检测数据集格式
    if (dataset_path / args.train_split / "_annotations.coco.json").exists():
        # COCO格式
        for split in [args.train_split, args.valid_split, args.test_split]:
            anno_path = dataset_path / split / "_annotations.coco.json"
            if not anno_path.exists():
                continue
                
            with open(anno_path, 'r') as f:
                coco_data = json.load(f)
                
            # 统计每个类别的样本数
            for ann in coco_data.get("annotations", []):
                cat_id = ann["category_id"]
                if cat_id in label_map:
                    class_counts[cat_id] += 1
    else:
        # YOLO格式
        for split in [args.train_split, args.valid_split, args.test_split]:
            labels_dir = dataset_path / split / "labels"
            if not labels_dir.exists():
                continue
                
            for lbl_file in labels_dir.glob("*.txt"):
                try:
                    with open(lbl_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if parts:
                                class_id = int(parts[0])
                                if class_id in label_map:
                                    class_counts[class_id] += 1
                except Exception as e:
                    print(f"  警告: 无法读取 {lbl_file}: {e}")
                    
    return dict(class_counts)


def calculate_generation_plan(class_counts, label_map, args):
    """
    计算需要生成的样本数
    
    参数:
        class_counts: 各类别当前样本数
        label_map: 类别映射
        args: 命令行参数
        
    返回:
        dict: {class_id: need_to_generate}
    """
    # 计算目标数量
    if args.target_ratio:
        # 按比例平衡
        max_count = max(class_counts.values())
        target_counts = {
            cid: int(max_count * args.target_ratio) 
            for cid in label_map.keys()
        }
    else:
        # 按最小样本数平衡
        min_needed = args.min_samples
        target_counts = {
            cid: max(class_counts.get(cid, 0), min_needed)
            for cid in label_map.keys()
        }
    
    # 计算需要生成的样本数
    generation_plan = {}
    for class_id, target in target_counts.items():
        current = class_counts.get(class_id, 0)
        need = max(0, target - current)
        if need > 0:
            generation_plan[class_id] = {
                'class_name': label_map[class_id],
                'current': current,
                'target': target,
                'need': need
            }
            
    return generation_plan


def print_analysis(class_counts, label_map, generation_plan, args):
    """打印分析结果"""
    print("\n" + "=" * 70)
    print("数据集类别分析")
    print("=" * 70)
    
    # 打印各类别统计
    print("\n各类别样本数:")
    print("-" * 70)
    print(f"{'类别ID':<10} {'类别名':<20} {'当前数量':<12} {'目标数量':<12} {'需生成':<12}")
    print("-" * 70)
    
    for class_id in sorted(label_map.keys()):
        class_name = label_map[class_id]
        current = class_counts.get(class_id, 0)
        if class_id in generation_plan:
            target = generation_plan[class_id]['target']
            need = generation_plan[class_id]['need']
        else:
            target = current
            need = 0
        print(f"{class_id:<10} {class_name:<20} {current:<12} {target:<12} {need:<12}")
        
    print("-" * 70)
    
    # 打印汇总
    total_current = sum(class_counts.values())
    total_need = sum(p['need'] for p in generation_plan.values())
    
    print(f"\n总计:")
    print(f"  当前样本数: {total_current}")
    print(f"  需要生成: {total_need}")
    print(f"  目标样本数: {total_current + total_need}")
    
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"  类别不平衡度: {ratio:.2f}x")
        
    print("=" * 70)


def generate_samples(generation_plan, args):
    """
    调用生成脚本补充少数类样本
    
    参数:
        generation_plan: 生成计划
        args: 命令行参数
    """
    if not generation_plan:
        print("\n✅ 所有类别已达标，无需生成!")
        return
        
    if args.dry_run:
        print(f"\n🔍 Dry run模式: 跳过生成")
        return
        
    print(f"\n开始生成样本...")
    print("=" * 70)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成各类别样本
    total_generated = 0
    
    for class_id, plan in generation_plan.items():
        class_name = plan['class_name']
        need = plan['need']
        
        print(f"\n类别 {class_id} ({class_name}): 需要生成 {need} 个样本")
        
        # 构建prompt
        prompt = f"a wind turbine blade with {class_name} damage"
        
        # 生成样本
        if args.generate_mode == "txt2img":
            # txt2img模式
            for i in range(need):
                seed = args.seed + class_id * 10000 + i
                output_file = output_dir / f"class{class_id}_{class_name}_{i:05d}.png"
                
                cmd = [
                    "python", "scripts/generate_defects.py",
                    "--mode", "txt2img",
                    "--pipe", args.pipe,
                    "--prompt", prompt,
                    "--seed", str(seed),
                    "--steps", str(args.steps),
                    "--guidance", str(args.guidance),
                    "--output", str(output_file),
                ]
                
                if args.lora_path:
                    cmd.extend(["--lora-path", args.lora_path])
                    
                print(f"  生成 {i+1}/{need}: {output_file.name}")
                
                try:
                    subprocess.run(cmd, check=True, capture_output=True, text=True)
                    total_generated += 1
                except subprocess.CalledProcessError as e:
                    print(f"  ❌ 生成失败: {e.stderr}")
                    
        else:
            # img2img模式（需要mask）
            print(f"  ⚠️ img2img模式需要手动提供mask图片")
            print(f"  提示: 可以使用extract_defect_patches.py生成mask")
            print(f"  或者切换到 --generate-mode txt2img")
            break
            
    print("\n" + "=" * 70)
    print(f"生成完成! 共生成 {total_generated} 个样本")
    print(f"保存至: {args.output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="数据集类别均衡脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 分析数据集（不生成）
  python balance_dataset.py --dataset ./my_dataset \\
      --label-map "0:crack,1:damage,2:erosion" --dry-run
  
  # 生成补充样本
  python balance_dataset.py --dataset ./my_dataset \\
      --label-map "0:crack,1:damage" \\
      --generate-mode txt2img --pipe sd \\
      --output-dir ./balanced
  
  # 使用LoRA生成
  python balance_dataset.py --dataset ./my_dataset \\
      --label-map "0:crack,1:damage" \\
      --generate-mode txt2img --pipe anomalyany \\
      --lora-path ./outputs/lora_stage2/final \\
      --output-dir ./balanced
        """
    )
    
    # 必选参数
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集路径（YOLO或COCO格式）")
    parser.add_argument("--label-map", type=str, required=True,
                        help="类别映射，格式: 'class_id:label,class_id:label'")
    
    # 输出参数
    parser.add_argument("--output-dir", type=str, default="./balanced_dataset",
                        help="输出目录")
    parser.add_argument("--dry-run", action="store_true",
                        help="只分析不生成")
    
    # 生成参数
    parser.add_argument("--generate-mode", type=str, default="txt2img",
                        choices=["txt2img", "img2img"],
                        help="生成模式")
    parser.add_argument("--pipe", type=str, default="sd",
                        choices=["sd", "anomalyany"],
                        help="生成管道")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="LoRA权重路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--steps", type=int, default=30,
                        help="推理步数")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="guidance scale")
    
    # 平衡参数
    parser.add_argument("--target-ratio", type=float, default=None,
                        help="目标类别比例（0-1），如0.5表示各类别样本数为目标最大类的50%%")
    parser.add_argument("--min-samples", type=int, default=100,
                        help="每类最小样本数")
    
    # 数据集格式参数
    parser.add_argument("--train-split", type=str, default="train",
                        help="训练集目录名")
    parser.add_argument("--valid-split", type=str, default="valid",
                        help="验证集目录名")
    parser.add_argument("--test-split", type=str, default="test",
                        help="测试集目录名")
    
    args = parser.parse_args()
    
    # 解析标签映射
    label_map = parse_label_map(args.label_map)
    
    print("=" * 70)
    print("数据集类别均衡")
    print("=" * 70)
    print(f"数据集路径: {args.dataset}")
    print(f"类别映射: {label_map}")
    print(f"输出目录: {args.output_dir}")
    print(f"生成模式: {args.generate_mode}")
    print(f"管道: {args.pipe}")
    if args.lora_path:
        print(f"LoRA路径: {args.lora_path}")
    print("=" * 70)
    
    # 分析数据集
    class_counts = analyze_dataset(args.dataset, label_map, args)
    
    # 计算生成计划
    generation_plan = calculate_generation_plan(class_counts, label_map, args)
    
    # 打印分析结果
    print_analysis(class_counts, label_map, generation_plan, args)
    
    # 生成样本
    generate_samples(generation_plan, args)
    
    print("\n✅ 完成!")


if __name__ == "__main__":
    main()
