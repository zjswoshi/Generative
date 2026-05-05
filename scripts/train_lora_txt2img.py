"""
=================================================================
LoRA训练脚本 - 用于txt2img生成（两阶段训练）
=================================================================

【脚本用途】
本脚本用于训练LoRA，使Stable Diffusion能够根据文字描述
直接生成完整的风机叶片图片（包含各种缺陷）。

【两阶段训练原理】
Stage 1（叶片外观学习）:
  - 输入: 大量正常风机叶片图片
  - 目标: 学会"什么是风机叶片"（叶片形状、颜色、纹理等）
  - 目的: 让SD知道如何生成真实的风机叶片
  
Stage 2（缺陷特征学习）:
  - 输入: 带缺陷的风机叶片图片
  - 目标: 在Stage 1基础上，学会"叶片上的缺陷长什么样"
  - 目的: 让SD能在叶片上生成各种缺陷
  - 注意: 使用更小的学习率，避免遗忘Stage 1学到的知识

【为什么需要两阶段？】
1. 如果直接用缺陷图片训练，SD可能学会"生成有缺陷的图片"
   但不知道"如何生成风机叶片的形状"
2. Stage 1先让SD学会"生成叶片外观"
3. Stage 2在基础上添加"缺陷知识"
4. 最终LoRA = 叶片外观 + 缺陷特征

【输入数据格式】
processed_dataset/
├── metadata.jsonl          # 每行: {"file_name": "xxx.png", "text": "caption"}
├── normal/                 # 正常叶片图片
│   ├── nordtank_00001.png
│   └── ...
└── defect/                # 缺陷叶片图片
    ├── fengji_crack_00001.png
    └── ...

【使用方法】
# 完整训练（两个阶段都运行）
python train_lora_txt2img.py

# 只运行Stage 2（使用已有的Stage 1权重）
python train_lora_txt2img.py --skip-stage1

# 自定义参数
python train_lora_txt2img.py \
    --stage1-steps 2000 \
    --stage2-steps 3000 \
    --stage1-lr 1e-4 \
    --stage2-lr 5e-5

【输出】
outputs/
├── lora_stage1_blade/
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   ├── checkpoint-1500/
│   ├── checkpoint-2000/
│   └── final/                    # Stage 1最终权重
└── lora_stage2_defect/
    ├── checkpoint-500/
    ├── checkpoint-1000/
    ├── checkpoint-2000/
    ├── checkpoint-3000/
    └── final/                    # Stage 2最终权重（用于生成）

【生成测试】
训练完成后，脚本会自动用测试prompt生成图片到 test_output/ 目录

【适用场景】
- 文字描述生成（如："带裂纹的风机叶片"）
- 不需要提供原始叶片图片
- 直接从零生成完整图片

【LoRA超参数说明】
- lora_rank: LoRA的秩，值越大表达能力越强（通常16-32）
- lora_alpha: LoRA的缩放因子，通常设置为lora_rank的值
- learning_rate: 学习率，Stage 2比Stage 1小以避免遗忘
- max_train_steps: 训练步数，越多越精细但也越慢
"""

import os
import json
import argparse
import gc
import random

import torch
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from torchvision import transforms

from diffusers import StableDiffusionPipeline, DDPMScheduler, AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler

from peft import LoraConfig, get_peft_model, PeftModel
from transformers import CLIPTextModel, CLIPTokenizer


DATASET_DIR = "/home/cn/yolo/AnomalyAny/processed_dataset"
MODEL_PATH = "/home/cn/yolo/AnomalyAny/sd-2-1-base"
STAGE1_OUTPUT = "./outputs/lora_stage1_blade"
STAGE2_OUTPUT = "./outputs/lora_stage2_defect"


class WindTurbineDataset(Dataset):
    """
    风机叶片数据集类
    
    功能:
    - 从metadata.jsonl读取图片列表
    - 支持三种模式: all(全部)、normal_only(仅正常)、defect_only(仅缺陷)
    - 自动resize和归一化图片
    
    参数:
    - data_dir: 数据目录（包含metadata.jsonl和normal/defect文件夹）
    - mode: 数据模式
      * "all": 使用所有图片（normal + defect）
      * "normal_only": 只用正常叶片
      * "defect_only": 只用缺陷叶片
    - size: 图片resize的目标尺寸
    """
    
    def __init__(self, data_dir, mode="all", size=512):
        self.size = size
        self.data_dir = data_dir
        self.mode = mode

        # 读取metadata.jsonl
        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        with open(metadata_path, 'r') as f:
            all_entries = [json.loads(line) for line in f if line.strip()]

        # 根据模式筛选数据
        if mode == "normal_only":
            # 只选文件名包含"normal"的图片
            self.entries = [e for e in all_entries if "normal" in e["file_name"]]
        elif mode == "defect_only":
            # 只选文件名包含"defect"的图片
            self.entries = [e for e in all_entries if "defect" in e["file_name"]]
        else:
            # 使用所有图片
            self.entries = all_entries

        print(f"数据集模式: {mode}, 样本数: {len(self.entries)}")

        # 图片预处理: Resize → ToTensor → Normalize
        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # 归一化到[-1, 1]
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        """
        返回单个样本
        返回: {
            "pixel_values": 预处理后的图片tensor,
            "caption": 图片对应的文字描述
        }
        """
        entry = self.entries[idx]
        img_path = os.path.join(self.data_dir, entry["file_name"])
        caption = entry["text"]

        image = Image.open(img_path).convert("RGB")
        image = self.image_transforms(image)

        return {"pixel_values": image, "caption": caption}


def train_lora(args):
    """
    训练LoRA的核心函数
    
    参数:
    - args: 命令行参数，包含所有训练配置
    
    训练流程:
    1. 加载预训练SD模型（VAE、Text Encoder、UNet）
    2. 为UNet添加LoRA适配器
    3. 创建DataLoader
    4. 循环训练:
       - 编码图片为latent
       - 添加噪声
       - UNet预测噪声
       - 计算损失并反向传播
       - 定期保存checkpoint
    5. 保存最终LoRA权重
    """
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print(f"LoRA训练 - {args.stage_name}")
    print("=" * 70)
    print(f"模型路径: {args.model_path}")
    print(f"数据集: {args.dataset_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"数据模式: {args.mode}")
    print(f"LoRA rank: {args.lora_rank}, alpha: {args.lora_alpha}")
    print(f"学习率: {args.learning_rate}")
    print(f"训练步数: {args.max_train_steps}")
    print("=" * 70)

    # 设置随机种子保证可重复性
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
    text_encoder.requires_grad_(False)  # Text Encoder不训练

    vae = AutoencoderKL.from_pretrained(args.model_path, subfolder="vae")
    vae.to(device)
    vae.requires_grad_(False)  # VAE不训练

    unet = UNet2DConditionModel.from_pretrained(args.model_path, subfolder="unet")
    unet.to(device)

    # 加载或创建LoRA适配器
    if args.resume_lora:
        print(f"加载已有LoRA权重继续训练: {args.resume_lora}")
        unet = PeftModel.from_pretrained(unet, args.resume_lora)
        for name, param in unet.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        unet.print_trainable_parameters()
    else:
        # 创建LoRA配置
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

    # 混合精度训练加速
    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        vae = vae.to(dtype=torch.float16)
        text_encoder = text_encoder.to(dtype=torch.float16)

    # 准备数据集和DataLoader
    dataset = WindTurbineDataset(args.dataset_dir, mode=args.mode, size=args.resolution)
    dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # 丢弃最后一个不完整的batch
    )

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=1e-2,
    )
    lr_scheduler = get_scheduler(
        "cosine",  # 余弦退火学习率
        optimizer=optimizer,
        num_warmup_steps=min(200, args.max_train_steps // 10),  # 预热步数
        num_training_steps=args.max_train_steps,
    )

    # 设置模型状态
    unet.train()   # UNet训练模式
    vae.eval()     # VAE eval模式（不更新参数）
    text_encoder.eval()  # Text Encoder eval模式

    # 训练循环
    global_step = 0
    loss_sum = 0.0
    log_interval = 50      # 每50步打印一次
    save_interval = 500    # 每500步保存一次

    print(f"\n开始训练 (共 {args.max_train_steps} 步)...")

    while global_step < args.max_train_steps:
        for batch in dataloader:
            if global_step >= args.max_train_steps:
                break

            pixel_values = batch["pixel_values"].to(device)
            captions = batch["caption"]

            with torch.no_grad():
                # 编码图片为latent（压缩表示）
                with autocast(enabled=use_amp):
                    latents = vae.encode(pixel_values.to(dtype=torch.float16 if use_amp else torch.float32)).latent_dist.sample()
                    latents = latents * vae.config.scaling_factor

                # 编码文字描述
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

            # 添加噪声（DDPM的前向过程）
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # UNet预测噪声
            with autocast(enabled=use_amp):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # 根据scheduler类型确定目标
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = latents
                    
                # 计算MSE损失
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

            # 反向传播
            scaler.scale(loss).backward()

            # 梯度累积（当batch太小时使用）
            if (global_step + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()

            global_step += 1
            loss_val = loss.detach().item()
            loss_sum += loss_val

            # 打印训练日志
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

    # 清理GPU内存
    del vae, text_encoder, optimizer, lr_scheduler, scaler, unet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return final_path


def test_generation(model_path, lora_path, output_dir, prompts):
    """
    用训练好的LoRA生成测试图片
    
    参数:
    - model_path: SD基础模型路径
    - lora_path: 训练好的LoRA权重路径
    - output_dir: 输出目录
    - prompts: 测试用的文字描述列表
    """
    print("\n生成测试图像...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载SD pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    # 加载LoRA权重并合并
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.unet = pipe.unet.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)

    # 生成图片
    for name, prompt in prompts:
        for seed in [42, 123, 456]:
            generator = torch.Generator('cuda').manual_seed(seed)
            image = pipe(prompt, generator=generator, num_inference_steps=30).images[0]
            path = os.path.join(output_dir, f"{name}_s{seed}.png")
            image.save(path)
            print(f"  {path}")

    # 清理
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    """
    主函数: 执行两阶段LoRA训练
    
    流程:
    1. Stage 1: 用正常叶片训练（学会生成叶片外观）
    2. Stage 2: 用缺陷叶片训练（学会生成缺陷）
    """
    parser = argparse.ArgumentParser(description="两阶段LoRA训练 - 用于txt2img生成")
    
    # 通用参数
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR,
                        help="processed_dataset目录路径")
    parser.add_argument("--model-path", type=str, default=MODEL_PATH,
                        help="SD基础模型路径")
    parser.add_argument("--resolution", type=int, default=512,
                        help="图片分辨率")
    parser.add_argument("--train-batch-size", type=int, default=2,
                        help="训练batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4,
                        help="梯度累积步数")
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (秩)")
    parser.add_argument("--lora-alpha", type=int, default=16,
                        help="LoRA alpha (缩放因子)")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    
    # Stage 1参数
    parser.add_argument("--stage1-output", type=str, default=STAGE1_OUTPUT,
                        help="Stage 1输出目录")
    parser.add_argument("--stage1-steps", type=int, default=2000,
                        help="Stage 1训练步数")
    parser.add_argument("--stage1-lr", type=float, default=1e-4,
                        help="Stage 1学习率")
    parser.add_argument("--skip-stage1", action="store_true",
                        help="跳过Stage 1（使用已有的Stage 1权重）")
    
    # Stage 2参数
    parser.add_argument("--stage2-output", type=str, default=STAGE2_OUTPUT,
                        help="Stage 2输出目录")
    parser.add_argument("--stage2-steps", type=int, default=3000,
                        help="Stage 2训练步数")
    parser.add_argument("--stage2-lr", type=float, default=5e-5,
                        help="Stage 2学习率（比Stage 1小，避免遗忘）")
    parser.add_argument("--skip-stage2", action="store_true",
                        help="跳过Stage 2")
    
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("两阶段LoRA训练方案 - 用于txt2img生成")
    print("=" * 70)
    print("Stage 1: 仅用正常叶片图片 → 学会'什么是风机叶片'")
    print("Stage 2: 在阶段1基础上用缺陷图片 → 学会'叶片上的缺陷长什么样'")
    print("=" * 70)

    # ========== Stage 1: 学习正常叶片外观 ==========
    if not args.skip_stage1:
        print("\n" + "#" * 70)
        print("# Stage 1: 学习正常风机叶片外观")
        print("#" * 70)

        stage1_args = argparse.Namespace(
            stage_name="阶段1-正常叶片",
            model_path=args.model_path,
            dataset_dir=args.dataset_dir,
            output_dir=args.stage1_output,
            mode="normal_only",  # Stage 1只用正常叶片
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            learning_rate=args.stage1_lr,
            max_train_steps=args.stage1_steps,
            seed=args.seed,
            resume_lora=None,
        )
        
        stage1_path = train_lora(stage1_args)
        
        # 测试生成
        test_prompts = [
            ("normal_blade", "a normal wind turbine blade, clean surface"),
        ]
        test_generation(args.model_path, stage1_path, "test_output/stage1", test_prompts)

    # ========== Stage 2: 学习缺陷特征 ==========
    if not args.skip_stage2:
        print("\n" + "#" * 70)
        print("# Stage 2: 学习叶片缺陷特征")
        print("#" * 70)
        
        # Stage 2在Stage 1基础上继续训练
        resume_stage1 = os.path.join(args.stage1_output, "final")
        
        stage2_args = argparse.Namespace(
            stage_name="阶段2-缺陷叶片",
            model_path=args.model_path,
            dataset_dir=args.dataset_dir,
            output_dir=args.stage2_output,
            mode="defect_only",  # Stage 2只用缺陷叶片
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            learning_rate=args.stage2_lr,  # 使用更小的学习率
            max_train_steps=args.stage2_steps,
            seed=args.seed,
            resume_lora=resume_stage1 if not args.skip_stage1 else None,
        )
        
        stage2_path = train_lora(stage2_args)
        
        # 测试生成
        test_prompts = [
            ("blade_with_crack", "a wind turbine blade with crack damage"),
            ("blade_with_hole", "a wind turbine blade with hole penetration"),
            ("blade_with_erosion", "a wind turbine blade with erosion damage"),
        ]
        test_generation(args.model_path, stage2_path, "test_output/stage2", test_prompts)

    print("\n" + "=" * 70)
    print("两阶段LoRA训练全部完成!")
    print("=" * 70)
    print(f"Stage 1 权重: {args.stage1_output}/final")
    print(f"Stage 2 权重: {args.stage2_output}/final")
    print("=" * 70)


if __name__ == "__main__":
    main()
