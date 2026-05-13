"""
=================================================================
统一LoRA训练脚本 - 支持单阶段和两阶段训练
=================================================================

【脚本用途】
本脚本整合了单阶段和两阶段LoRA训练，专注于使用prepare_hq_datasets.py
生成的高质量数据集进行训练。

【支持的数据来源】
1. prepare_hq_datasets.py 输出：
   ├── metadata.jsonl        # 元数据（包含详细caption）
   ├── full_images/          # 完整叶片图片
   │   ├── blade_00001.png
   │   └── ...
   ├── full_masks/           # 缺陷mask
   │   ├── blade_00001_mask.png
   │   └── ...
   └── patches/             # 缺陷patch
       ├── defect_00001.png
       └── ...

2. 标准格式：
   ├── metadata.jsonl
   ├── normal/              # 正常叶片图片
   │   └── *.png
   ├── defect/              # 缺陷叶片图片
   │   └── *.png
   └── patches/            # 缺陷patch
       └── *.png

【训练模式】
1. 单阶段训练（--mode single）：
   - 输入：patches + metadata.text
   - 目标：学习缺陷的局部纹理特征
   - 适用：img2img局部替换任务
   - 不学：叶片整体外观（由输入图片决定）

2. 两阶段训练（--mode two-stage）：
   - Stage 1：full_images + metadata.text → 学习叶片外观
   - Stage 2：patches + metadata.patch_text → 学习缺陷特征
   - 适用：txt2img直接生成任务
   - 输出：统一的LoRA（叶片+缺陷知识）

3. 自动模式（--mode auto）：
   - 检测数据源，自动选择训练模式
   - 有full_images → 两阶段
   - 只有patches → 单阶段

4. 双LoRA并行模式（--mode dual-lora）：
   - Background LoRA：仅用正常图片训练，学习载体外观
   - Defect LoRA：仅用缺陷图片训练，学习缺陷特征
   - 两个LoRA完全独立训练，推理时加权融合
   - 优势：解耦背景和缺陷，避免灾难性遗忘
   - 推理时：W = W_base + α×LoRA_defect + β×LoRA_bg

【使用方法】
# 自动检测模式（推荐）
python train_lora_unified.py \
    --data-dir ./hq_output \
    --mode auto

# 单阶段训练（仅学习缺陷patch）
python train_lora_unified.py \
    --data-dir ./hq_output \
    --mode single \
    --max-steps 3000

# 两阶段训练（学习完整叶片和缺陷）
python train_lora_unified.py \
    --data-dir ./hq_output \
    --mode two-stage \
    --stage1-steps 4000 \
    --stage2-steps 3000

# 组合多个数据集
python train_lora_unified.py \
    --data-dirs ./hq_fengji ./hq_offshore ./hq_nordtank \
    --mode two-stage

# 双LoRA并行训练（Defect-LoRA风格）
python train_lora_unified.py \
    --data-dir ./hq_output \
    --mode dual-lora \
    --bg-steps 2000 \
    --defect-steps 3000

【输出】
outputs/
├── lora_single/              # 单阶段输出
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── final/
└── lora_two_stage/           # 两阶段输出
    ├── stage1/
    │   ├── checkpoint-500/
    │   └── final/
    └── stage2/
        ├── checkpoint-500/
        ├── checkpoint-1000/
        └── final/
└── lora_dual/                # 双LoRA并行输出
    ├── background/           # 背景LoRA
    │   ├── checkpoint-500/
    │   └── final/
    └── defect/               # 缺陷LoRA
        ├── checkpoint-500/
        ├── checkpoint-1000/
        └── final/

【训练参数建议】
单阶段：
- resolution: 256（patch较小）
- learning_rate: 1e-4
- max_steps: 2000-3000

两阶段：
- Stage 1: resolution=512, lr=1e-4, steps=1500-2000
- Stage 2: resolution=256, lr=5e-5, steps=2000-3000

【LoRA超参数】
- lora_rank: 16（推荐），可调整范围4-32
- lora_alpha: 通常设置为lora_rank的值
"""

import os
import json
import argparse
import gc
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel, StableDiffusionPipeline
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import CLIPTextModel, CLIPTokenizer


MODEL_PATH = "/home/cn/yolo/AnomalyAny/sd-2-1-base"
DEFAULT_DATA_DIR = "./hq_output"
DEFAULT_OUTPUT = "./outputs/lora_unified"


class UnifiedBladeDataset(Dataset):
    """
    统一的风机叶片数据集类
    
    支持的数据格式：
    1. prepare_hq_datasets.py 输出格式：
       - full_images/ + metadata.jsonl
       - patches/ + metadata.jsonl
    
    2. 标准格式：
       - normal/ + defect/ + metadata.jsonl
       - patches/ + metadata.jsonl
    
    参数：
    - data_dir: 数据目录
    - mode: 数据模式
      * "all": 所有图片
      * "full_images": 仅完整叶片
      * "patches": 仅缺陷patch
      * "normal_only": 仅正常叶片
      * "defect_only": 仅缺陷叶片
    - size: resize目标尺寸
    - use_detailed_caption: 是否使用metadata中的详细caption
    """
    
    def __init__(self, data_dir: str, mode: str = "all", 
                 size: int = 512, use_detailed_caption: bool = True):
        self.size = size
        self.data_dir = data_dir
        self.mode = mode
        self.use_detailed_caption = use_detailed_caption
        self.samples = []
        self.metadata = {}
        
        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        self._load_metadata()
        self._find_samples()
        
        print(f"  数据模式: {mode}, 样本数: {len(self.samples)}")
        if self.metadata and use_detailed_caption:
            print(f"  使用详细caption（{len(self.metadata)}条记录）")
    
    def _load_metadata(self):
        """加载metadata.jsonl"""
        metadata_path = Path(self.data_dir) / "metadata.jsonl"
        
        if not metadata_path.exists():
            return
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line.strip())
                    
                    file_name = data.get('file_name', '')
                    patch_name = data.get('patch_name', '')
                    text = data.get('text', '')
                    
                    if file_name:
                        self.metadata[file_name] = data
                    if patch_name:
                        self.metadata[patch_name] = data
                        
            print(f"  已加载metadata: {len(self.metadata)}条记录")
        except Exception as e:
            print(f"  警告: 无法加载metadata.jsonl: {e}")
    
    def _find_samples(self):
        """查找样本图片"""
        data_path = Path(self.data_dir)
        
        if self.mode == "full_images" or self.mode == "all":
            full_images_dir = data_path / "full_images"
            if full_images_dir.exists():
                for img_file in sorted(full_images_dir.glob("*.png")):
                    rel_path = f"full_images/{img_file.name}"
                    self.samples.append({
                        'path': str(img_file),
                        'rel_path': rel_path,
                        'type': 'full_image'
                    })
        
        if self.mode == "patches" or self.mode == "all":
            patches_dir = data_path / "patches"
            if patches_dir.exists():
                for img_file in sorted(patches_dir.glob("*.png")):
                    rel_path = f"patches/{img_file.name}"
                    self.samples.append({
                        'path': str(img_file),
                        'rel_path': rel_path,
                        'type': 'patch'
                    })
        
        if self.mode == "normal_only":
            normal_dir = data_path / "normal"
            if normal_dir.exists():
                for img_file in sorted(normal_dir.glob("*.png")):
                    rel_path = f"normal/{img_file.name}"
                    self.samples.append({
                        'path': str(img_file),
                        'rel_path': rel_path,
                        'type': 'normal'
                    })
        
        if self.mode == "defect_only":
            defect_dir = data_path / "defect"
            if defect_dir.exists():
                for img_file in sorted(defect_dir.glob("*.png")):
                    rel_path = f"defect/{img_file.name}"
                    self.samples.append({
                        'path': str(img_file),
                        'rel_path': rel_path,
                        'type': 'defect'
                    })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert("RGB")
        image = self.image_transforms(image)
        
        caption = self._get_caption(sample)
        
        return {
            "pixel_values": image,
            "caption": caption,
            "type": sample['type']
        }
    
    def _get_caption(self, sample):
        """获取caption"""
        if sample['rel_path'] in self.metadata:
            data = self.metadata[sample['rel_path']]
            # Stage 2 patch模式：使用patch_text（纯视觉特征，无位置信息）
            if sample['type'] == 'patch' and 'patch_text' in data and self.use_detailed_caption:
                return data['patch_text']
            # Stage 1 full_image模式：使用text（包含位置信息）
            if self.use_detailed_caption:
                text = data.get('text', '')
                if text:
                    return text

        type_map = {
            'full_image': 'wind turbine blade with defect',
            'patch': 'damaged wind turbine blade surface with defect',
            'normal': 'normal wind turbine blade',
            'defect': 'wind turbine blade with damage'
        }
        return type_map.get(sample['type'], 'wind turbine blade')


class CombinedUnifiedDataset(Dataset):
    """
    组合统一数据集
    
    从多个数据目录组合样本，支持不同的训练模式。
    """
    
    def __init__(self, data_dirs: List[str], mode: str = "all",
                 size: int = 512, use_detailed_caption: bool = True):
        self.size = size
        self.mode = mode
        self.use_detailed_caption = use_detailed_caption
        self.samples = []
        self.all_metadata = {}
        
        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        
        for data_dir in data_dirs:
            metadata_path = Path(data_dir) / "metadata.jsonl"
            if metadata_path.exists() and use_detailed_caption:
                try:
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip():
                                continue
                            data = json.loads(line.strip())
                            file_name = data.get('file_name', '')
                            patch_name = data.get('patch_name', '')
                            if file_name:
                                self.all_metadata[f"{data_dir}/{file_name}"] = data
                            if patch_name:
                                self.all_metadata[f"{data_dir}/{patch_name}"] = data
                except Exception as e:
                    print(f"  警告: 无法加载 {metadata_path}: {e}")
            
            for sample_type in ['full_images', 'patches', 'normal', 'defect']:
                type_dir = Path(data_dir) / sample_type
                if type_dir.exists():
                    for img_file in sorted(type_dir.glob("*.png")):
                        self.samples.append({
                            'path': str(img_file),
                            'rel_path': f"{sample_type}/{img_file.name}",
                            'data_dir': data_dir,
                            'type': sample_type
                        })
        
        print(f"  总样本数: {len(self.samples)}")
        if self.all_metadata:
            print(f"  加载metadata: {len(self.all_metadata)}条记录")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['path']).convert("RGB")
        image = self.image_transforms(image)
        
        if self.use_detailed_caption:
            key = f"{sample['data_dir']}/{sample['rel_path']}"
            metadata_entry = self.all_metadata.get(key, {})
            # Stage 2 patch模式：使用patch_text（纯视觉特征，无位置信息）
            if sample['type'] == 'patches' and 'patch_text' in metadata_entry:
                caption = metadata_entry['patch_text']
            else:
                caption = metadata_entry.get('text', '')
            if not caption:
                caption = self._get_default_caption(sample['type'])
        else:
            caption = self._get_default_caption(sample['type'])
        
        return {
            "pixel_values": image,
            "caption": caption,
            "type": sample['type']
        }
    
    def _get_default_caption(self, sample_type):
        type_map = {
            'full_images': 'wind turbine blade with defect',
            'patches': 'damaged wind turbine blade surface with defect',
            'normal': 'normal wind turbine blade',
            'defect': 'wind turbine blade with damage'
        }
        return type_map.get(sample_type, 'wind turbine blade')


def train_single_stage(args):
    """
    单阶段训练：仅学习缺陷patch的纹理特征
    
    适用于img2img任务，需要提供原始叶片图片。
    """
    print("\n" + "=" * 70)
    print("单阶段训练 - 学习缺陷纹理")
    print("=" * 70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_components = load_model_components(args.model_path, device)
    unet = model_components['unet']
    
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    dataset = UnifiedBladeDataset(
        args.data_dir,
        mode="patches",
        size=args.resolution,
        use_detailed_caption=args.use_detailed_caption
    )
    
    if len(dataset) == 0:
        print("\n错误: 未找到缺陷patch！")
        print(f"请确保 {args.data_dir}/patches/ 目录下有图片文件")
        return None
    
    return train_model(
        unet=unet,
        dataset=dataset,
        model_components=model_components,
        args=args,
        stage_name="单阶段-缺陷纹理",
        output_dir=args.output_dir
    )


def train_two_stage(args):
    """
    两阶段训练：
    Stage 1: 学习完整叶片外观
    Stage 2: 学习缺陷特征
    """
    print("\n" + "=" * 70)
    print("两阶段训练 - 叶片外观 + 缺陷特征")
    print("=" * 70)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_components = load_model_components(args.model_path, device)
    
    stage1_dir = os.path.join(args.output_dir, "stage1")
    stage2_dir = os.path.join(args.output_dir, "stage2")
    
    print("\n" + "#" * 70)
    print("# Stage 1: 学习完整叶片外观")
    print("#" * 70)
    
    dataset_stage1 = UnifiedBladeDataset(
        args.data_dir,
        mode="full_images",
        size=args.resolution,
        use_detailed_caption=args.use_detailed_caption
    )
    
    if len(dataset_stage1) == 0:
        dataset_stage1 = UnifiedBladeDataset(
            args.data_dir,
            mode="all",
            size=args.resolution,
            use_detailed_caption=args.use_detailed_caption
        )
        print("  未找到full_images，使用all模式")
    
    unet1 = model_components['unet']
    lora_config1 = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    unet1 = get_peft_model(unet1, lora_config1)
    unet1.print_trainable_parameters()
    
    stage1_args = argparse.Namespace(
        learning_rate=args.stage1_lr,
        max_train_steps=args.stage1_steps,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    stage1_path = train_model(
        unet=unet1,
        dataset=dataset_stage1,
        model_components=model_components,
        args=stage1_args,
        stage_name="阶段1-叶片外观",
        output_dir=stage1_dir
    )
    
    del unet1
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n" + "#" * 70)
    print("# Stage 2: 学习缺陷特征")
    print("#" * 70)
    
    dataset_stage2 = UnifiedBladeDataset(
        args.data_dir,
        mode="patches",
        size=256,
        use_detailed_caption=args.use_detailed_caption
    )
    
    if len(dataset_stage2) == 0:
        print("  错误: 未找到缺陷patch！")
        return stage1_path
    
    unet2 = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")
    unet2.to(device)
    
    print(f"  加载Stage 1权重继续训练: {stage1_path}")
    unet2 = PeftModel.from_pretrained(unet2, stage1_path)
    for name, param in unet2.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    unet2.print_trainable_parameters()
    
    stage2_args = argparse.Namespace(
        learning_rate=args.stage2_lr,
        max_train_steps=args.stage2_steps,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    stage2_path = train_model(
        unet=unet2,
        dataset=dataset_stage2,
        model_components=model_components,
        args=stage2_args,
        stage_name="阶段2-缺陷特征",
        output_dir=stage2_dir
    )
    
    return stage2_path


def train_dual_lora(args):
    """
    双LoRA并行训练（Defect-LoRA风格）
    
    Background LoRA：仅用正常图片训练，学习载体外观
    Defect LoRA：仅用缺陷图片训练，学习缺陷特征
    两个LoRA完全独立，推理时加权融合
    """
    print("\n" + "=" * 70)
    print("双LoRA并行训练 - 背景 + 缺陷（独立训练）")
    print("=" * 70)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    bg_dir = os.path.join(args.output_dir, "background")
    defect_dir = os.path.join(args.output_dir, "defect")

    print("\n" + "#" * 70)
    print("# Background LoRA: 学习载体外观（仅正常图片）")
    print("#" * 70)

    dataset_bg = UnifiedBladeDataset(
        args.data_dir,
        mode="normal_only",
        size=args.resolution,
        use_detailed_caption=args.use_detailed_caption
    )

    if len(dataset_bg) == 0:
        dataset_bg = UnifiedBladeDataset(
            args.data_dir,
            mode="full_images",
            size=args.resolution,
            use_detailed_caption=args.use_detailed_caption
        )
        print("  未找到normal目录，使用full_images模式")

    if len(dataset_bg) == 0:
        print("  错误: 未找到正常图片！请确保有 normal/ 或 full_images/ 目录")
        return None, None

    model_components_bg = load_model_components(args.model_path, device)
    unet_bg = model_components_bg['unet']

    lora_config_bg = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    unet_bg = get_peft_model(unet_bg, lora_config_bg)
    unet_bg.print_trainable_parameters()

    bg_args = argparse.Namespace(
        learning_rate=args.bg_lr,
        max_train_steps=args.bg_steps,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    bg_path = train_model(
        unet=unet_bg,
        dataset=dataset_bg,
        model_components=model_components_bg,
        args=bg_args,
        stage_name="Background-LoRA",
        output_dir=bg_dir
    )

    del unet_bg, model_components_bg
    gc.collect()
    torch.cuda.empty_cache()

    print("\n" + "#" * 70)
    print("# Defect LoRA: 学习缺陷特征（仅缺陷图片）")
    print("#" * 70)

    dataset_defect = UnifiedBladeDataset(
        args.data_dir,
        mode="defect_only",
        size=256,
        use_detailed_caption=args.use_detailed_caption
    )

    if len(dataset_defect) == 0:
        dataset_defect = UnifiedBladeDataset(
            args.data_dir,
            mode="patches",
            size=256,
            use_detailed_caption=args.use_detailed_caption
        )
        print("  未找到defect目录，使用patches模式")

    if len(dataset_defect) == 0:
        print("  错误: 未找到缺陷图片！请确保有 defect/ 或 patches/ 目录")
        return bg_path, None

    model_components_def = load_model_components(args.model_path, device)
    unet_def = model_components_def['unet']

    lora_config_def = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    unet_def = get_peft_model(unet_def, lora_config_def)
    unet_def.print_trainable_parameters()

    defect_args = argparse.Namespace(
        learning_rate=args.defect_lr,
        max_train_steps=args.defect_steps,
        train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    defect_path = train_model(
        unet=unet_def,
        dataset=dataset_defect,
        model_components=model_components_def,
        args=defect_args,
        stage_name="Defect-LoRA",
        output_dir=defect_dir
    )

    print("\n" + "=" * 70)
    print("双LoRA训练完成!")
    print(f"  Background LoRA: {bg_path}")
    print(f"  Defect LoRA: {defect_path}")
    print("  推理时使用: --lora-bg <bg_path> --lora-defect <defect_path>")
    print("  可选参数: --alpha 0.7 --beta 1.0")
    print("=" * 70)

    return bg_path, defect_path


def load_model_components(model_path, device):
    """加载SD模型组件"""
    noise_scheduler = DDPMScheduler.from_pretrained(model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder")
    text_encoder.to(device)
    text_encoder.requires_grad_(False)
    
    vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
    vae.to(device)
    vae.requires_grad_(False)
    
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder="unet")
    unet.to(device)
    
    use_amp = torch.cuda.is_available()
    # 暂时禁用AMP以避免GradScaler错误
    use_amp = False
    if use_amp:
        vae = vae.to(dtype=torch.float16)
        text_encoder = text_encoder.to(dtype=torch.float16)
    
    return {
        'noise_scheduler': noise_scheduler,
        'tokenizer': tokenizer,
        'text_encoder': text_encoder,
        'vae': vae,
        'unet': unet,
        'use_amp': use_amp
    }


def train_model(unet, dataset, model_components, args, stage_name, output_dir):
    """通用训练函数"""
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    noise_scheduler = model_components['noise_scheduler']
    tokenizer = model_components['tokenizer']
    text_encoder = model_components['text_encoder']
    vae = model_components['vae']
    use_amp = model_components['use_amp']
    
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        unet = unet.to(dtype=torch.float16)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=min(200, args.max_train_steps // 10),
        num_training_steps=args.max_train_steps,
    )
    
    unet.train()
    vae.eval()
    text_encoder.eval()
    
    global_step = 0
    loss_sum = 0.0
    log_interval = 50
    save_interval = 500
    
    print(f"\n开始训练 (共 {args.max_train_steps} 步)...")
    
    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break
            
            pixel_values = batch["pixel_values"].to(device)
            captions = batch["caption"]
            if isinstance(captions, str):
                captions = [captions]
            
            with torch.no_grad():
                with autocast(enabled=use_amp):
                    latents = vae.encode(
                        pixel_values.to(dtype=torch.float16 if use_amp else torch.float32)
                    ).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor
                
                text_inputs = tokenizer(
                    captions,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                with autocast(enabled=use_amp):
                    encoder_hidden_states = text_encoder(text_input_ids)[0]
            
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device
            )
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # 计算预测 (使用autocast)
            with autocast(enabled=use_amp):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # 计算损失 (确保float32)
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                target = latents

            loss = torch.nn.functional.mse_loss(
                model_pred.float(), target.float(), reduction="mean"
            )

            scaler.scale(loss).backward()
            
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
            
            global_step += 1
            loss_val = loss.detach().item()
            loss_sum += loss_val
            
            if global_step % log_interval == 0:
                avg_loss = loss_sum / log_interval
                lr = lr_scheduler.get_last_lr()[0]
                mem_used = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                print(f"  Step {global_step}/{args.max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | GPU: {mem_used:.1f}GB")
                loss_sum = 0.0
            
            if global_step % save_interval == 0:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                unet.save_pretrained(save_path)
                print(f"  保存检查点: {save_path}")
    
    final_path = os.path.join(output_dir, "final")
    unet.save_pretrained(final_path)
    print(f"\n训练完成! 模型保存至: {final_path}")
    
    return final_path


def detect_training_mode(data_dir):
    """自动检测训练模式"""
    data_path = Path(data_dir)
    
    has_full_images = (data_path / "full_images").exists() and any((data_path / "full_images").glob("*.png"))
    has_patches = (data_path / "patches").exists() and any((data_path / "patches").glob("*.png"))
    has_normal = (data_path / "normal").exists() and any((data_path / "normal").glob("*.png"))
    has_defect = (data_path / "defect").exists() and any((data_path / "defect").glob("*.png"))
    
    if has_full_images or (has_normal and has_defect):
        return "two-stage"
    elif has_patches:
        return "single"
    else:
        return "single"


def main():
    parser = argparse.ArgumentParser(
        description="统一LoRA训练脚本 - 支持单阶段和两阶段训练",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="数据目录（包含metadata.jsonl）")
    parser.add_argument("--data-dirs", type=str, nargs='+',
                        help="多个数据目录")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="SD基础模型路径")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT,
                        help="输出目录")
    
    parser.add_argument("--mode", type=str, default="auto", choices=["auto", "single", "two-stage", "dual-lora"],
                        help="训练模式: auto/single/two-stage/dual-lora")
    parser.add_argument("--resolution", type=int, default=512,
                        help="图片分辨率")
    parser.add_argument("--use-detailed-caption", action="store_true", default=True,
                        help="使用metadata.jsonl中的详细caption")
    
    parser.add_argument("--train-batch-size", type=int, default=2,
                        help="训练batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    parser.add_argument("--stage1-steps", type=int, default=2000,
                        help="Stage 1训练步数")
    parser.add_argument("--stage1-lr", type=float, default=1e-4,
                        help="Stage 1学习率")
    parser.add_argument("--stage2-steps", type=int, default=3000,
                        help="Stage 2训练步数")
    parser.add_argument("--stage2-lr", type=float, default=5e-5,
                        help="Stage 2学习率（较小，避免遗忘）")
    
    parser.add_argument("--max-steps", type=int, default=3000,
                        help="单阶段训练步数")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="单阶段学习率")

    parser.add_argument("--bg-steps", type=int, default=2000,
                        help="双LoRA模式-背景LoRA训练步数")
    parser.add_argument("--bg-lr", type=float, default=1e-4,
                        help="双LoRA模式-背景LoRA学习率")
    parser.add_argument("--defect-steps", type=int, default=3000,
                        help="双LoRA模式-缺陷LoRA训练步数")
    parser.add_argument("--defect-lr", type=float, default=1e-4,
                        help="双LoRA模式-缺陷LoRA学习率")
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print("\n" + "=" * 70)
    print("统一LoRA训练")
    print("=" * 70)
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"LoRA配置: rank={args.lora_rank}, alpha={args.lora_alpha}")
    
    if args.mode == "auto":
        detected_mode = detect_training_mode(args.data_dir)
        print(f"自动检测模式: {detected_mode}")
        training_mode = detected_mode
    else:
        training_mode = args.mode
        print(f"训练模式: {training_mode}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    
    print("=" * 70)
    
    if training_mode == "single":
        if args.data_dirs:
            dataset = CombinedUnifiedDataset(
                args.data_dirs,
                mode="patches",
                size=args.resolution,
                use_detailed_caption=args.use_detailed_caption
            )
        else:
            dataset = UnifiedBladeDataset(
                args.data_dir,
                mode="patches",
                size=args.resolution,
                use_detailed_caption=args.use_detailed_caption
            )
        
        if len(dataset) == 0:
            print("\n错误: 未找到缺陷patch！")
            return
        
        args_single = argparse.Namespace(
            model_path=args.model_path,
            data_dir=args.data_dir,
            output_dir=os.path.join(args.output_dir, "single"),
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            learning_rate=args.learning_rate,
            max_train_steps=args.max_steps,
            seed=args.seed,
            use_detailed_caption=args.use_detailed_caption,
        )
        
        lora_path = train_single_stage(args_single)
        
    elif training_mode == "two-stage":
        args_two = argparse.Namespace(
            model_path=args.model_path,
            output_dir=os.path.join(args.output_dir, "two_stage"),
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            stage1_steps=args.stage1_steps,
            stage1_lr=args.stage1_lr,
            stage2_steps=args.stage2_steps,
            stage2_lr=args.stage2_lr,
            seed=args.seed,
            data_dir=args.data_dir,
            use_detailed_caption=args.use_detailed_caption,
        )
        
        lora_path = train_two_stage(args_two)

    elif training_mode == "dual-lora":
        args_dual = argparse.Namespace(
            model_path=args.model_path,
            output_dir=os.path.join(args.output_dir, "dual"),
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            bg_steps=args.bg_steps,
            bg_lr=args.bg_lr,
            defect_steps=args.defect_steps,
            defect_lr=args.defect_lr,
            seed=args.seed,
            data_dir=args.data_dir,
            use_detailed_caption=args.use_detailed_caption,
        )

        bg_path, defect_path = train_dual_lora(args_dual)
    
    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    if lora_path:
        print(f"LoRA权重: {lora_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
