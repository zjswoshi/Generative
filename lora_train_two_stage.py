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
    def __init__(self, data_dir, mode="all", size=512):
        self.size = size
        self.data_dir = data_dir
        self.mode = mode

        metadata_path = os.path.join(data_dir, "metadata.jsonl")
        with open(metadata_path, 'r') as f:
            all_entries = [json.loads(line) for line in f if line.strip()]

        if mode == "normal_only":
            self.entries = [e for e in all_entries if "normal" in e["file_name"]]
        elif mode == "defect_only":
            self.entries = [e for e in all_entries if "defect" in e["file_name"]]
        else:
            self.entries = all_entries

        print(f"数据集模式: {mode}, 样本数: {len(self.entries)}")

        self.image_transforms = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.data_dir, entry["file_name"])
        caption = entry["text"]

        image = Image.open(img_path).convert("RGB")
        image = self.image_transforms(image)

        return {"pixel_values": image, "caption": caption}


def train_lora(args):
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

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")

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

    if args.resume_lora:
        print(f"加载已有LoRA权重继续训练: {args.resume_lora}")
        unet = PeftModel.from_pretrained(unet, args.resume_lora)
        for name, param in unet.named_parameters():
            if "lora_" in name:
                param.requires_grad = True
        unet.print_trainable_parameters()
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        )
        unet = get_peft_model(unet, lora_config)
        unet.print_trainable_parameters()

    use_amp = torch.cuda.is_available()
    scaler = GradScaler(enabled=use_amp)
    if use_amp:
        vae = vae.to(dtype=torch.float16)
        text_encoder = text_encoder.to(dtype=torch.float16)

    dataset = WindTurbineDataset(args.dataset_dir, mode=args.mode, size=args.resolution)
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

            with torch.no_grad():
                with autocast(enabled=use_amp):
                    latents = vae.encode(pixel_values.to(dtype=torch.float16 if use_amp else torch.float32)).latent_dist.sample()
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
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            with autocast(enabled=use_amp):
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    target = latents
                loss = torch.nn.functional.mse_loss(model_pred.float(), target.float(), reduction="mean")

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
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                unet.save_pretrained(save_path)
                print(f"  保存检查点: {save_path}")

    final_path = os.path.join(args.output_dir, "final")
    unet.save_pretrained(final_path)
    print(f"\n训练完成! 模型保存至: {final_path}")

    del vae, text_encoder, optimizer, lr_scheduler, scaler, unet
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return final_path


def test_generation(model_path, lora_path, output_dir, prompts):
    print("\n生成测试图像...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)

    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.unet = pipe.unet.merge_and_unload()

    os.makedirs(output_dir, exist_ok=True)

    for name, prompt in prompts:
        for seed in [42, 123, 456]:
            generator = torch.Generator('cuda').manual_seed(seed)
            image = pipe(prompt, generator=generator, num_inference_steps=30).images[0]
            path = os.path.join(output_dir, f"{name}_s{seed}.png")
            image.save(path)
            print(f"  {path}")

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="两阶段LoRA训练")
    parser.add_argument("--dataset-dir", type=str, default=DATASET_DIR)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--stage1-output", type=str, default=STAGE1_OUTPUT)
    parser.add_argument("--stage2-output", type=str, default=STAGE2_OUTPUT)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-stage1", action="store_true", help="跳过阶段1(使用已有权重)")
    parser.add_argument("--skip-stage2", action="store_true", help="跳过阶段2")
    parser.add_argument("--stage1-steps", type=int, default=2000, help="阶段1训练步数")
    parser.add_argument("--stage2-steps", type=int, default=3000, help="阶段2训练步数")
    parser.add_argument("--stage1-lr", type=float, default=1e-4, help="阶段1学习率")
    parser.add_argument("--stage2-lr", type=float, default=5e-5, help="阶段2学习率(更小,避免遗忘)")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("两阶段LoRA训练方案")
    print("=" * 70)
    print("阶段1: 仅用正常叶片图片 → 学会'什么是风机叶片'")
    print("阶段2: 在阶段1基础上用缺陷图片 → 学会'叶片上的缺陷长什么样'")
    print("=" * 70)

    # ========== 阶段1: 学习正常叶片外观 ==========
    if not args.skip_stage1:
        print("\n" + "#" * 70)
        print("# 阶段1: 学习正常风机叶片外观")
        print("#" * 70)

        stage1_args = argparse.Namespace(
            stage_name="阶段1-正常叶片",
            model_path=args.model_path,
            dataset_dir=args.dataset_dir,
            output_dir=args.stage1_output,
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.stage1_lr,
            max_train_steps=args.stage1_steps,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            mode="normal_only",
            resume_lora=None,
            seed=args.seed,
        )
        stage1_lora_path = train_lora(stage1_args)

        test_prompts_s1 = [
            ("normal", "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects"),
        ]
        test_generation(
            args.model_path,
            stage1_lora_path,
            os.path.join(args.stage1_output, "test_samples"),
            test_prompts_s1,
        )
    else:
        stage1_lora_path = os.path.join(args.stage1_output, "final")
        print(f"\n跳过阶段1, 使用已有权重: {stage1_lora_path}")

    # ========== 阶段2: 学习缺陷特征 ==========
    if not args.skip_stage2:
        print("\n" + "#" * 70)
        print("# 阶段2: 学习风机叶片缺陷特征")
        print("#" * 70)

        stage2_args = argparse.Namespace(
            stage_name="阶段2-缺陷特征",
            model_path=args.model_path,
            dataset_dir=args.dataset_dir,
            output_dir=args.stage2_output,
            resolution=args.resolution,
            train_batch_size=args.train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.stage2_lr,
            max_train_steps=args.stage2_steps,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            mode="defect_only",
            resume_lora=stage1_lora_path,
            seed=args.seed,
        )
        stage2_lora_path = train_lora(stage2_args)

        test_prompts_s2 = [
            ("normal", "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects"),
            ("crack", "a photo of a white fiberglass wind turbine blade surface with crack, damaged area"),
            ("erosion", "a photo of a white fiberglass wind turbine blade surface with erosion, damaged area"),
            ("coating_damage", "a photo of a white fiberglass wind turbine blade surface with coating damage, damaged area"),
            ("contamination", "a photo of a white fiberglass wind turbine blade surface with contamination, damaged area"),
            ("peeling", "a photo of a white fiberglass wind turbine blade surface with peeling, damaged area"),
            ("hole", "a photo of a white fiberglass wind turbine blade surface with hole, damaged area"),
        ]
        test_generation(
            args.model_path,
            stage2_lora_path,
            os.path.join(args.stage2_output, "test_samples"),
            test_prompts_s2,
        )

    print("\n" + "=" * 70)
    print("两阶段训练全部完成!")
    print(f"阶段1权重(正常叶片): {args.stage1_output}/final")
    print(f"阶段2权重(含缺陷): {args.stage2_output}/final")
    print("=" * 70)


if __name__ == "__main__":
    main()
