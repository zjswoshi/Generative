"""
=================================================================
LoRA训练脚本 - 用于img2img生成（单阶段训练）
=================================================================

【脚本用途】
本脚本用于训练LoRA，使Stable Diffusion能够在img2img任务中
将正常叶片图片的指定区域替换为缺陷（通过mask控制）。

【与txt2img脚本的区别】
train_lora_txt2img.py (两阶段):
  - Stage 1: 学习正常叶片外观
  - Stage 2: 学习缺陷特征
  - 输出: LoRA包含"叶片外观+缺陷"知识
  - 用途: txt2img（直接根据文字生成完整叶片）

train_lora_img2img.py (单阶段):
  - 直接用缺陷patch训练
  - 只学: 缺陷的局部纹理特征
  - 不学: 叶片整体外观
  - 用途: img2img（局部替换，需要提供原始叶片图片）

【单阶段训练原理】
1. 输入: 裁剪好的缺陷patch图片（256x256）
   - 这些patch是从带缺陷的叶片上提取的小块
   - 只包含缺陷区域，不包含完整叶片
   
2. 训练: 让SD学习"这个patch的纹理、颜色、形状"
   - 不需要学习叶片外观
   - 只需要学会"如何生成这种缺陷纹理"
   
3. 输出: LoRA只包含"缺陷纹理知识"

【为什么用patch而不是完整图片？】
1. 计算效率: 256x256比512x512小4倍，训练更快
2. 聚焦学习: 只学缺陷纹理，不受叶片背景干扰
3. 灵活应用: 可以把patch"贴"到任何叶片上

【输入数据格式】
缺陷patch目录结构:
dataset/
├── images/          # 缺陷patch图片
│   ├── defect_001.png
│   ├── defect_002.png
│   └── ...
└── masks/           # 对应的mask（可选，用于知道缺陷范围）
    ├── defect_001.png
    └── ...

注意: images目录下的.png文件会被自动收集为训练样本
支持递归搜索（会自动查找所有images子目录）

【使用方法】
# 基本用法
python train_lora_img2img.py

# 指定数据目录和参数
python train_lora_img2img.py \
    --data-dir /path/to/defect/patches \
    --output-dir ./outputs/defect_patch_lora \
    --max-steps 3000 \
    --learning-rate 1e-4

# 组合多个数据集
python train_lora_img2img.py \
    --data-dirs ./patches/fengji ./patches/offshore ./patches/nordtank

【输出】
outputs/defect_patch_lora/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-2000/
├── checkpoint-3000/
└── final/                    # 最终权重（用于img2img生成）

【生成测试】
训练完成后，脚本会自动用测试prompt生成图片到 test_output/ 目录

【适用场景】
- img2img局部替换（如：正常叶片 + mask → 带缺陷叶片）
- 需要提供原始叶片图片作为基础
- 只需要生成缺陷纹理，不关心叶片外观

【img2img使用示例】
```python
from diffusers import StableDiffusionInpaintPipeline
from peft import PeftModel

# 加载SD Inpaint模型
pipe = StableDiffusionInpaintPipeline.from_pretrained("sd-2-1-base")

# 加载训练的LoRA
pipe.unet = PeftModel.from_pretrained(pipe.unet, "outputs/defect_patch_lora/final")
pipe.unet = pipe.unet.merge_and_unload()

# 加载正常叶片图片和mask
normal_image = Image.open("normal_blade.png")
mask = Image.open("defect_mask.png")

# 生成带缺陷的叶片
result = pipe(
    prompt="damaged wind turbine blade surface with defect",
    image=normal_image,
    mask_image=mask,
).images[0]

result.save("result.png")
```

【LoRA超参数说明】
- lora_rank: LoRA的秩（通常16-32）
  * r=4: 轻量级，可能欠拟合
  * r=16: 平衡（推荐）
  * r=32: 表达能力更强，可能过拟合
  
- lora_alpha: LoRA缩放因子，通常设置为lora_rank的值

- learning_rate: 学习率
  * 1e-4: 常用起始值
  * 5e-5: 更保守，避免过拟合
  * 1e-5: 微调，已有关键词知识
  
- max_train_steps: 训练步数
  * 1000: 快速测试
  * 2000-3000: 正常训练
  * 5000+: 精细训练

【为什么这个脚本不用两阶段？】
因为img2img任务中，叶片的外观是由输入图片决定的，不需要LoRA学习。
LoRA只需要学习"如何生成缺陷纹理"，单阶段就足够了。
"""

import os
import json
import argparse
import gc
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torchvision import transforms
from diffusers import DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from peft import LoraConfig, get_peft_model
from transformers import CLIPTextModel, CLIPTokenizer


MODEL_PATH = "/home/cn/yolo/AnomalyAny/sd-2-1-base"
DEFAULT_OUTPUT = "./outputs/defect_patch_lora"

# 缺陷类别到描述的映射（用于生成caption）
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


class DefectPatchDataset(Dataset):
    """
    缺陷Patch数据集类
    
    功能:
    - 递归搜索所有images目录
    - 收集所有.png文件作为训练样本
    - 可选支持mask（如果masks目录存在）
    
    特点:
    - 使用256x256分辨率（比完整图片小，计算更快）
    - 只关注缺陷纹理，不关注叶片背景
    
    参数:
    - data_dir: 缺陷patch根目录
    - size: resize目标尺寸，默认256
    """
    
    def __init__(self, data_dir, size=256):
        self.size = size
        self.data_dir = data_dir
        self.samples = []

        # 图片预处理
        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        # 递归查找所有images目录
        self._find_images(data_dir)
        print(f"  缺陷patch数量: {len(self.samples)}")

    def _find_images(self, directory):
        """
        递归查找所有images目录下的png文件
        
        支持的目录结构:
        - data_dir/images/*.png
        - data_dir/subdir/images/*.png
        - data_dir/dataset1/images/*.png
        """
        directory = Path(directory)

        # 查找images子目录
        images_dir = directory / "images"
        if images_dir.exists():
            # 收集所有png文件
            for fname in sorted(images_dir.glob("*.png")):
                masks_dir = directory / "masks"
                self.samples.append({
                    'image': str(fname),
                    'mask': str(masks_dir / fname.name) if masks_dir.exists() else None
                })

        # 递归搜索子目录
        for subdir in directory.iterdir():
            if subdir.is_dir() and subdir.name not in ['__pycache__', '.git']:
                self._find_images(subdir)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        返回单个样本
        返回: {
            "pixel_values": 预处理后的patch图片tensor,
            "caption": 统一的文字描述
        }
        """
        sample = self.samples[idx]
        image = Image.open(sample['image']).convert("RGB")
        image = self.image_transforms(image)

        # 所有缺陷patch使用统一的caption
        caption = "damaged wind turbine blade surface with defect"

        return {"pixel_values": image, "caption": caption}


class CombinedDefectDataset(Dataset):
    """
    组合缺陷数据集
    
    功能:
    - 从多个数据集目录组合样本
    - 用于训练时使用多个来源的缺陷patch
    
    参数:
    - data_dirs: 多个数据目录的列表
    - size: resize目标尺寸
    """
    
    def __init__(self, data_dirs, size=256):
        self.size = size
        
        # 图片预处理
        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        self.samples = []
        
        # 从每个目录收集样本
        for data_dir in data_dirs:
            images_dir = os.path.join(data_dir, "images")
            masks_dir = os.path.join(data_dir, "masks")

            if os.path.exists(images_dir):
                for fname in sorted(os.listdir(images_dir)):
                    if fname.endswith('.png'):
                        self.samples.append({
                            'image': os.path.join(images_dir, fname),
                            'mask': os.path.join(masks_dir, fname),
                            'source': os.path.basename(data_dir)
                        })

        print(f"  总缺陷patch数量: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample['image']).convert("RGB")
        image = self.image_transforms(image)

        caption = "damaged wind turbine blade surface with defect"

        return {"pixel_values": image, "caption": caption}


def train_defect_lora(args):
    """
    训练缺陷局部LoRA的核心函数
    
    参数:
    - args: 命令行参数
    
    训练流程:
    1. 加载预训练SD模型
    2. 为UNet添加LoRA适配器
    3. 用缺陷patch训练
    4. 保存LoRA权重
    
    注意:
    - 不需要两阶段（叶片外观由img2img的输入图片决定）
    - 只学缺陷纹理（256x256 patch）
    """
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("缺陷局部LoRA训练 - 用于img2img")
    print("=" * 60)
    print(f"模型路径: {args.model_path}")
    print(f"数据目录: {args.data_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练步数: {args.max_train_steps}")
    print("=" * 60)

    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 检查GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # 加载预训练模型组件
    noise_scheduler = DDPMScheduler.from_pretrained(args.model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.model_path, subfolder="text_encoder")
    text_encoder.to(device)
    text_encoder.requires_grad_(False)

    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
    vae.to(device)
    vae.requires_grad_(False)

    unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")
    unet.to(device)

    # 创建LoRA适配器
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # 混合精度训练
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        vae = vae.to(dtype=torch.float16)
        text_encoder = text_encoder.to(dtype=torch.float16)

    # 加载数据集
    print(f"\n加载数据集: {args.data_dir}")
    dataset = DefectPatchDataset(args.data_dir, size=args.resolution)

    # 检查数据集是否为空
    if len(dataset) == 0:
        print(f"\n错误: 数据集为空!")
        print(f"请确保 {args.data_dir}/images/ 目录下有.png文件")
        print(f"或者运行 extract_defect_patches.py 先提取缺陷patch")
        return

    # 如果指定了多个数据目录，使用组合数据集
    if args.data_dirs:
        dataset = CombinedDefectDataset(args.data_dirs, size=args.resolution)

    # 创建DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # 优化器和学习率调度器
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

    # 设置模型状态
    unet.train()
    vae.eval()
    text_encoder.eval()

    # 训练循环
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

            with torch.no_grad():
                # 编码图片为latent
                with autocast(enabled=use_amp):
                    latents = vae.encode(pixel_values.to(dtype=torch.float16 if use_amp else torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # 编码统一的caption
                text_inputs = tokenizer(
                    ["damaged wind turbine blade surface with defect"] * latents.shape[0],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_input_ids = text_inputs.input_ids.to(device)
                with autocast(enabled=use_amp):
                    encoder_hidden_states = text_encoder(text_input_ids)[0]

            # 添加噪声
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet预测噪声
            with autocast(enabled=use_amp):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # 确定训练目标
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = latents
                    
                # 计算损失
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度累积
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            loss_val = loss.detach().item()
            loss_sum += loss_val

            # 打印日志
            if global_step % log_interval == 0:
                avg_loss = loss_sum / log_interval
                lr = lr_scheduler.get_last_lr()[0]
                mem_used = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                print(f"  Step {global_step}/{args.max_train_steps} | Loss: {avg_loss:.4f} | LR: {lr:.2e} | GPU: {mem_used:.1f}GB")
                loss_sum = 0.0

            # 保存checkpoint
            if global_step % save_interval == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                unet.save_pretrained(save_path)
                print(f"  保存检查点: {save_path}")

    # 保存最终模型
    final_path = os.path.join(args.output_dir, "final")
    unet.save_pretrained(final_path)
    print(f"\n训练完成! 模型保存至: {final_path}")

    # 清理内存
    del vae, text_encoder, optimizer, lr_scheduler, scaler, unet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return final_path


def main():
    parser = argparse.ArgumentParser(description="缺陷局部LoRA训练 - 用于img2img生成")
    
    # 路径参数
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="SD基础模型路径")
    parser.add_argument("--data-dir", type=str, default="defect_patches",
                        help="缺陷patch数据目录（包含images子目录）")
    parser.add_argument("--data-dirs", type=str, nargs='+',
                        help="多个缺陷patch数据目录（用于组合数据集）")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT,
                        help="输出目录")
    
    # 训练参数
    parser.add_argument("--resolution", type=int, default=256,
                        help="Patch图片分辨率")
    parser.add_argument("--train-batch-size", type=int, default=4,
                        help="训练batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                        help="梯度累积步数")
    parser.add_argument("--max-train-steps", type=int, default=3000,
                        help="最大训练步数")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("缺陷局部LoRA训练 - 用于img2img局部替换")
    print("=" * 60)
    print("训练模式: 单阶段（直接学习缺陷patch）")
    print("训练目标: 学习缺陷的局部纹理特征")
    print("=" * 60)

    # 执行训练
    final_path = train_defect_lora(args)

    print("\n" + "=" * 60)
    print("LoRA训练完成!")
    print("=" * 60)
    print(f"LoRA权重: {final_path}")
    print("\n使用方法:")
    print("  1. 加载SD Inpaint模型")
    print("  2. 加载LoRA权重到UNet")
    print("  3. 提供正常叶片图片 + mask")
    print("  4. 生成带缺陷的叶片")
    print("=" * 60)


if __name__ == "__main__":
    main()
