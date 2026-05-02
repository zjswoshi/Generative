import argparse
import gc
import json
import os
import random
import time

import numpy as np
import torch
from diffusers import PNDMScheduler
from peft import PeftModel

from clip_pipeline_attend_and_excite import RelationalAttendAndExcitePipeline
from utils.ptp_utils import register_attention_control, AttentionStore


POSITION_VARIANTS = [
    "on the left side",
    "on the right side",
    "in the center",
    "near the leading edge",
    "near the trailing edge",
    "near the root",
    "near the tip",
    "on the upper surface",
    "on the lower surface",
    "along the span",
]

SEVERITY_VARIANTS = [
    "minor",
    "slight",
    "moderate",
    "severe",
    "significant",
    "extensive",
    "small",
    "large",
    "tiny",
    "widespread",
]

APPEARANCE_VARIANTS = [
    "visible",
    "clearly visible",
    "subtle",
    "obvious",
    "barely noticeable",
    "prominent",
    "faint",
    "distinct",
    "sharp",
    "blurred",
]

LIGHTING_VARIANTS = [
    "under natural daylight",
    "in bright sunlight",
    "in overcast conditions",
    "under inspection lighting",
    "with side lighting",
    "under diffuse light",
    "in shadow",
    "with raking light",
    "under UV inspection light",
    "in harsh direct light",
]

TEXTURE_VARIANTS = [
    "with rough texture",
    "with smooth surrounding surface",
    "with fibrous edges",
    "with chipped paint",
    "with exposed material",
    "with discolored area",
    "with raised edges",
    "with sunken surface",
    "with flaking coating",
    "with worn appearance",
]


DEFECT_CONFIGS = [
    {
        "name": "crack",
        "name_cn": "裂纹",
        "defect_words": ["crack", "fracture", "split", "fissure", "hairline crack"],
        "normal_prompt": "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects",
        "token_indices": [9],
    },
    {
        "name": "erosion",
        "name_cn": "侵蚀",
        "defect_words": ["erosion", "corrosion", "wear", "pitting", "material loss"],
        "normal_prompt": "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects",
        "token_indices": [9],
    },
    {
        "name": "coating_damage",
        "name_cn": "涂层损伤",
        "defect_words": ["coating damage", "paint damage", "peeling paint", "flaking coating", "chipped coating"],
        "normal_prompt": "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects",
        "token_indices": [9],
    },
    {
        "name": "surface_damage",
        "name_cn": "表面损伤",
        "defect_words": ["surface damage", "gouge", "scratch", "abrasion", "scuff"],
        "normal_prompt": "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects",
        "token_indices": [9],
    },
    {
        "name": "contamination",
        "name_cn": "污染",
        "defect_words": ["contamination", "oil stain", "dirt", "discoloration", "soiling"],
        "normal_prompt": "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects",
        "token_indices": [9],
    },
    {
        "name": "peeling",
        "name_cn": "剥落",
        "defect_words": ["peeling", "delamination", "flaking", "blistering", "separation"],
        "normal_prompt": "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects",
        "token_indices": [9],
    },
    {
        "name": "hole",
        "name_cn": "孔洞",
        "defect_words": ["hole", "perforation", "penetration", "puncture", "breakthrough"],
        "normal_prompt": "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects",
        "token_indices": [9],
    },
    {
        "name": "burn_damage",
        "name_cn": "雷击",
        "defect_words": ["burn damage", "lightning strike", "scorch mark", "charred area", "thermal damage"],
        "normal_prompt": "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects",
        "token_indices": [9],
    },
    {
        "name": "delamination",
        "name_cn": "分层",
        "defect_words": ["delamination", "layer separation", "internal separation", "bond failure", "laminate split"],
        "normal_prompt": "a photo of a normal white fiberglass wind turbine blade surface, clean and intact, no defects",
        "token_indices": [9],
    },
]


def get_defect_token_index(pipe, prompt, defect_word):
    text_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    tokens = pipe.tokenizer.convert_ids_to_tokens(text_inputs.input_ids[0])
    defect_tokens = pipe.tokenizer(defect_word, add_special_tokens=False)
    defect_ids = defect_tokens["input_ids"]

    for i in range(len(tokens) - len(defect_ids) + 1):
        match = True
        for j, did in enumerate(defect_ids):
            if text_inputs.input_ids[0][i + j].item() != did:
                match = False
                break
        if match:
            return i

    return 9


def build_prompts(config, rng):
    defect_word = rng.choice(config["defect_words"])
    position = rng.choice(POSITION_VARIANTS)
    severity = rng.choice(SEVERITY_VARIANTS)
    appearance = rng.choice(APPEARANCE_VARIANTS)
    lighting = rng.choice(LIGHTING_VARIANTS)
    texture = rng.choice(TEXTURE_VARIANTS)

    anomaly_prompt = (
        f"a photo of a white fiberglass wind turbine blade surface "
        f"{position} with {severity} {defect_word}, {appearance} {texture}, {lighting}"
    )

    detailed_prompt = (
        f"a detailed inspection photo of a white fiberglass wind turbine blade surface "
        f"showing {severity} {defect_word} {position}, the defect is {appearance} "
        f"and {texture}, captured {lighting}, industrial inspection image"
    )

    return anomaly_prompt, detailed_prompt, defect_word


def load_pipeline(model_path, lora_path, device, lora_scale=0.8):
    print("加载基础模型...")
    pipe = RelationalAttendAndExcitePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        scheduler=PNDMScheduler.from_pretrained(model_path, subfolder="scheduler"),
    )
    print(f"加载LoRA权重: {lora_path} (scale={lora_scale})")
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe.unet = pipe.unet.merge_and_unload()
    pipe = pipe.to(device)
    print("模型加载完成")
    return pipe


def generate_image(pipe, config, output_path, seed, num_steps, guidance_scale,
                    anomaly_prompt, detailed_prompt, token_indices, noise_strength=0.0):
    controller = AttentionStore()
    register_attention_control(pipe, controller)
    generator = torch.Generator('cuda').manual_seed(seed)

    start_time = time.time()
    try:
        outputs = pipe(
            prompt=anomaly_prompt,
            attention_store=controller,
            indices_to_alter=token_indices,
            attention_res=16,
            guidance_scale=guidance_scale,
            generator=generator,
            num_inference_steps=num_steps,
            max_iter_to_alter=num_steps // 2,
            run_standard_sd=False,
            thresholds={0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor=20,
            scale_range=(1.0, 0.5),
            smooth_attentions=True,
            sigma=0.5,
            kernel_size=3,
            sd_2_1=True,
            normal_prompt=config["normal_prompt"],
            detailed_prompt=detailed_prompt,
        )

        if isinstance(outputs, tuple):
            image = outputs[0]
        elif isinstance(outputs, list):
            image = outputs[0]
        else:
            image = outputs.images[0]
        while isinstance(image, list):
            image = image[0]

        if noise_strength > 0:
            import PIL.Image as PILImage
            arr = np.array(image).astype(np.float32)
            noise = np.random.normal(0, noise_strength * 255, arr.shape)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            image = PILImage.fromarray(arr)

        elapsed = time.time() - start_time
        image.save(output_path)

        arr_check = np.array(image)
        is_black = arr_check.mean() < 5
        is_white = arr_check.mean() > 250
        return image, elapsed, is_black, is_white

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  生成出错: {e}")
        return None, elapsed, True, False


def main():
    parser = argparse.ArgumentParser(description="AnomalyAny批量缺陷生成 (多样性增强+防记忆)")
    parser.add_argument("--model-path", type=str, default="/home/cn/yolo/AnomalyAny/sd-2-1-base")
    parser.add_argument("--lora-path", type=str, default="./outputs/lora_stage2_defect/final")
    parser.add_argument("--output-dir", type=str, default="/home/cn/yolo/AnomalyAny/dataset_blade_defect_10k")
    parser.add_argument("--total", type=int, default=10000, help="总生成数量")
    parser.add_argument("--defect", type=str, default=None, help="指定缺陷类型,逗号分隔(如crack,erosion)")
    parser.add_argument("--steps", type=int, default=0, help="0=随机25~50")
    parser.add_argument("--guidance", type=float, default=0, help="0=随机6~12")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-strength", type=float, default=0.02,
                        help="输出图像添加轻微噪声(0~0.05), 防止像素级记忆")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    pipe = load_pipeline(args.model_path, args.lora_path, device)

    if args.defect:
        defect_names = [d.strip() for d in args.defect.split(',')]
        configs = [c for c in DEFECT_CONFIGS if c["name"] in defect_names]
        if not configs:
            print(f"未找到缺陷类型: {args.defect}")
            print(f"可用类型: {[c['name'] for c in DEFECT_CONFIGS]}")
            return
        print(f"指定缺陷类型: {[c['name'] for c in configs]}")
    else:
        configs = DEFECT_CONFIGS

    num_defects = len(configs)
    per_defect = args.total // num_defects
    total = per_defect * num_defects

    print(f"\n{'='*70}")
    print(f"批量生成配置 (多样性增强+防记忆):")
    print(f"  总数量: {total}")
    print(f"  缺陷类型: {num_defects} 种, 每种 {per_defect} 张")
    print(f"  内循环迭代: 12次")
    print(f"  Prompt变化: 缺陷词/位置/严重度/外观/光照/纹理 6维随机组合")
    print(f"  Guidance: {'随机6~12' if args.guidance == 0 else args.guidance}")
    print(f"  Steps: {'随机25~50' if args.steps == 0 else args.steps}")
    print(f"  防记忆噪声: {args.noise_strength}")
    print(f"  输出目录: {args.output_dir}")
    est_time = total * 45 / 2 / 3600
    print(f"  预计耗时: ~{est_time:.0f} 小时 (双GPU, 按45s/张估算)")
    print(f"{'='*70}")

    os.makedirs(args.output_dir, exist_ok=True)

    rng = random.Random(args.seed)
    base_seed = args.seed
    count = 0
    success = 0
    black_count = 0
    error_count = 0
    start_time_all = time.time()

    for config in configs:
        defect_dir = os.path.join(args.output_dir, config["name"])
        os.makedirs(defect_dir, exist_ok=True)

        for i in range(per_defect):
            count += 1

            seed = base_seed + count * 137 + i * 31
            output_path = os.path.join(defect_dir, f"{config['name']}_{i:05d}.png")

            if os.path.exists(output_path):
                success += 1
                continue

            anomaly_prompt, detailed_prompt, defect_word = build_prompts(config, rng)

            token_idx = get_defect_token_index(pipe, anomaly_prompt, defect_word)
            token_indices = [token_idx]

            num_steps = args.steps if args.steps > 0 else rng.randint(25, 50)
            guidance_scale = args.guidance if args.guidance > 0 else rng.uniform(6.0, 12.0)

            print(f"\n[{count}/{total}] {config['name_cn']}({config['name']}) #{i}")
            print(f"  seed={seed}, steps={num_steps}, guidance={guidance_scale:.1f}, token_idx={token_idx}")
            print(f"  prompt: {anomaly_prompt[:80]}...")

            image, elapsed, is_black, is_white = generate_image(
                pipe, config, output_path,
                seed=seed,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                anomaly_prompt=anomaly_prompt,
                detailed_prompt=detailed_prompt,
                token_indices=token_indices,
                noise_strength=args.noise_strength,
            )

            if image is not None and not is_black and not is_white:
                success += 1
                print(f"  OK {elapsed:.1f}s")
            else:
                if is_black:
                    black_count += 1
                    print(f"  黑图 {elapsed:.1f}s")
                else:
                    error_count += 1
                    print(f"  失败 {elapsed:.1f}s")

            if count % 50 == 0:
                elapsed_all = time.time() - start_time_all
                rate = success / elapsed_all * 3600 if elapsed_all > 0 else 0
                remaining = (total - count) / rate * 3600 if rate > 0 else 0
                print(f"\n--- 进度: {count}/{total} ({count/total*100:.1f}%) "
                      f"成功:{success} 黑图:{black_count} 失败:{error_count} "
                      f"速率:{rate:.0f}张/h 预计剩余:{remaining/3600:.1f}h ---\n")

                progress_file = os.path.join(args.output_dir, "_progress.json")
                with open(progress_file, 'w') as f:
                    json.dump({"success": success, "black": black_count,
                               "error": error_count, "total": total, "count": count}, f)

                gc.collect()
                torch.cuda.empty_cache()

    elapsed_all = time.time() - start_time_all
    print(f"\n{'='*70}")
    print(f"生成完成!")
    print(f"  成功: {success}")
    print(f"  黑图: {black_count}")
    print(f"  失败: {error_count}")
    print(f"  总计: {count}")
    print(f"  总耗时: {elapsed_all/3600:.1f} 小时")
    print(f"  平均速率: {success/elapsed_all*3600:.0f} 张/小时")
    print(f"  保存在: {args.output_dir}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
