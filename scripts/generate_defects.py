"""
=================================================================
通用缺陷生成脚本
=================================================================

【脚本用途】
通过参数控制，支持txt2img和img2img两种生成模式，
同时支持普通SD和AnomalyAny两种管道。

【支持的生成模式】

模式1: txt2img（文字生图）
  - 使用文字描述生成缺陷图片
  - 不需要输入图片

模式2: img2img（图生图）
  - 在输入图片的mask区域生成缺陷
  - 需要提供原图和mask

【支持的管道】

管道1: sd（普通SD）
  - 使用Stable Diffusion Inpaint Pipeline
  - 适合快速生成

管道2: anomalyany（CLIP增强）
  - 使用Attend-and-Excite技术
  - 更好的局部控制

【命令行参数】

必选参数:
  --mode [txt2img|img2img]  生成模式
  --pipe [sd|anomalyany]    使用哪个管道

txt2img模式必选:
  --prompt                 文字描述

img2img模式必选:
  --image                  输入图片路径
  --mask                   mask图片路径

可选参数:
  --lora-path              LoRA权重路径
  --output                 输出路径，默认"output.png"
  --seed                   随机种子，默认42
  --steps                  推理步数，默认30
  --guidance               guidance scale，默认7.5

【使用方法】

示例1: SD txt2img生成
python generate_defects.py \
    --mode txt2img \
    --pipe sd \
    --prompt "a wind turbine blade with crack damage" \
    --output ./output.png

示例2: SD img2img生成
python generate_defects.py \
    --mode img2img \
    --pipe sd \
    --image normal_blade.png \
    --mask defect_mask.png \
    --output result.png

示例3: AnomalyAny txt2img生成
python generate_defects.py \
    --mode txt2img \
    --pipe anomalyany \
    --prompt "a wind turbine blade with crack damage" \
    --output ./output.png

示例4: AnomalyAny img2img生成
python generate_defects.py \
    --mode img2img \
    --pipe anomalyany \
    --image normal_blade.png \
    --mask defect_mask.png \
    --output result.png

示例5: 使用LoRA生成
python generate_defects.py \
    --mode txt2img \
    --pipe sd \
    --lora-path ./outputs/lora_stage2_defect/final \
    --prompt "a wind turbine blade with crack damage" \
    --output result.png

【管道说明】

SD管道:
  - 使用StableDiffusionInpaintPipeline
  - 通过mask控制生成区域
  - 快速，适合大批量生成

AnomalyAny管道:
  - 使用Attend-and-Excite技术
  - CLIP引导更好的局部控制
  - 更精确的缺陷生成

【注意事项】
1. img2img模式会自动resize图片到512x512
2. mask必须是灰度图，白色区域为要生成的位置
3. AnomalyAny管道需要较新的diffusers版本
4. LoRA会合并到模型权重中
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import torch
from PIL import Image
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from diffusers import StableDiffusionInpaintPipeline, PNDMScheduler
from peft import PeftModel


def load_dual_lora_pipeline(model_path, lora_bg_path=None, lora_defect_path=None,
                            alpha=0.7, beta=1.0, pipe_type="sd", device="cuda"):
    """
    加载双LoRA加权融合Pipeline（Defect-LoRA风格）
    
    W = W_base + alpha * W_defect + beta * W_bg
    
    参数:
        model_path: SD模型路径
        lora_bg_path: 背景LoRA权重路径
        lora_defect_path: 缺陷LoRA权重路径
        alpha: 缺陷LoRA权重（控制缺陷强度）
        beta: 背景LoRA权重（控制背景保真度）
        pipe_type: 管道类型 "sd" 或 "anomalyany"
        device: 设备
    """
    from safetensors.torch import load_file, save_file
    import tempfile
    
    print(f"加载双LoRA融合模型 (α={alpha}, β={beta}):")
    print(f"  背景LoRA: {lora_bg_path}")
    print(f"  缺陷LoRA: {lora_defect_path}")
    
    if lora_bg_path and lora_defect_path:
        bg_path = Path(lora_bg_path)
        defect_path = Path(lora_defect_path)
        
        bg_file = bg_path / "adapter_model.safetensors" if bg_path.is_dir() else bg_path
        defect_file = defect_path / "adapter_model.safetensors" if defect_path.is_dir() else defect_path
        
        bg_state = load_file(str(bg_file))
        defect_state = load_file(str(defect_file))
        
        merged_state = {}
        all_keys = set(bg_state.keys()) | set(defect_state.keys())
        
        for key in all_keys:
            bg_val = bg_state.get(key, torch.zeros_like(defect_state.get(key, torch.zeros(1))))
            defect_val = defect_state.get(key, torch.zeros_like(bg_state.get(key, torch.zeros(1))))
            
            if key in bg_state and key in defect_state:
                merged_state[key] = beta * bg_val + alpha * defect_val
            elif key in bg_state:
                merged_state[key] = beta * bg_val
            else:
                merged_state[key] = alpha * defect_val
        
        bg_config = Path(lora_bg_path) / "adapter_config.json" if Path(lora_bg_path).is_dir() else None
        merged_dir = tempfile.mkdtemp(prefix="dual_lora_")
        
        save_file(merged_state, os.path.join(merged_dir, "adapter_model.safetensors"))
        
        if bg_config and bg_config.exists():
            import shutil
            shutil.copy(str(bg_config), os.path.join(merged_dir, "adapter_config.json"))
        
        print(f"  融合权重已保存至临时目录: {merged_dir}")
        lora_path = merged_dir
    elif lora_bg_path:
        lora_path = lora_bg_path
    elif lora_defect_path:
        lora_path = lora_defect_path
    else:
        lora_path = None
    
    if pipe_type == "sd":
        return load_sd_pipeline(model_path, lora_path, device)
    else:
        return load_anomalyany_pipeline(model_path, lora_path, device)


def load_sd_pipeline(model_path, lora_path=None, device="cuda"):
    """
    加载SD Inpaint Pipeline
    
    参数:
        model_path: SD模型路径
        lora_path: LoRA权重路径（可选）
        device: 设备
        
    返回:
        StableDiffusionInpaintPipeline
    """
    print(f"加载SD Inpaint模型: {model_path}")
    
    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    
    if lora_path and Path(lora_path).exists():
        print(f"加载LoRA: {lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        pipe.unet = pipe.unet.merge_and_unload()
    
    return pipe


def load_anomalyany_pipeline(model_path, lora_path=None, device="cuda"):
    """
    加载AnomalyAny Pipeline (Attend-and-Excite)
    
    参数:
        model_path: SD模型路径
        lora_path: LoRA权重路径（可选）
        device: 设备
        
    返回:
        RelationalAttendAndExcitePipeline
    """
    print(f"加载AnomalyAny模型: {model_path}")
    
    from clip_pipeline_attend_and_excite import RelationalAttendAndExcitePipeline
    from utils.ptp_utils import register_attention_control, AttentionStore
    
    pipe = RelationalAttendAndExcitePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        scheduler=PNDMScheduler.from_pretrained(model_path, subfolder="scheduler"),
    )
    pipe = pipe.to(device, dtype=torch.float16)
    
    if lora_path and Path(lora_path).exists():
        print(f"加载LoRA: {lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
        pipe.unet = pipe.unet.merge_and_unload()
    
    return pipe


def get_token_indices(prompt, tokenizer):
    """
    获取prompt中关键词的token索引
    
    参数:
        prompt: 文字提示
        tokenizer: 分词器
        
    返回:
        list: token索引列表
    """
    tokens = tokenizer(prompt)['input_ids']
    token_idx_to_word = {
        idx: tokenizer.decode(t).lower().strip()
        for idx, t in enumerate(tokens)
        if 0 < idx < len(tokens) - 1
    }
    
    defect_keywords = [
        'crack', 'damage', 'contamination', 'dirt', 'scratch', 'hole', 'burn',
        'peeling', 'fracture', 'debris', 'imperfection', 'delamination',
        'erosion', 'corrosion', 'defect', 'stain', 'scar', 'gap', 'split',
        'tear', 'break', 'corruption', 'fault', 'flaw', 'deterioration'
    ]
    
    indices = []
    for idx, word in token_idx_to_word.items():
        if any(kw in word for kw in defect_keywords):
            indices.append(idx)

    if not indices:
        indices = [len(tokens) - 2]

    # 只返回最重要的一个关键词（第一个匹配的）
    return [indices[0]] if indices else [len(tokens) - 2]


def generate_txt2img_sd(pipe, prompt, seed, steps, guidance):
    """SD txt2img生成"""
    generator = torch.Generator('cuda').manual_seed(seed)
    
    result = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=guidance,
    )
    
    return result.images[0]


def generate_img2img_sd(pipe, image, mask, prompt, seed, steps, guidance):
    """SD img2img生成"""
    generator = torch.Generator('cuda').manual_seed(seed)
    
    result = pipe(
        prompt=prompt,
        image=image,
        mask_image=mask,
        generator=generator,
        num_inference_steps=steps,
        guidance_scale=guidance,
    )
    
    return result.images[0]


def generate_txt2img_anomalyany(pipe, prompt, seed, steps, guidance):
    """AnomalyAny txt2img生成"""
    from utils.ptp_utils import register_attention_control, AttentionStore
    
    controller = AttentionStore()
    register_attention_control(pipe, controller)
    
    tokens = pipe.tokenizer(prompt)['input_ids']
    token_indices = get_token_indices(prompt, pipe.tokenizer)
    
    token_idx_to_word = {
        idx: pipe.tokenizer.decode(t)
        for idx, t in enumerate(tokens)
        if 0 < idx < len(tokens) - 1
    }
    
    print(f"  增强关键词: {[token_idx_to_word.get(i, i) for i in token_indices]}")
    
    normal_prompt = prompt.replace(" with ", " that is ")
    
    generator = torch.Generator('cuda').manual_seed(seed)
    
    outputs = pipe(
        prompt=prompt,
        attention_store=controller,
        indices_to_alter=token_indices,
        attention_res=16,
        guidance_scale=guidance,
        generator=generator,
        num_inference_steps=steps,
        max_iter_to_alter=steps // 2,
        run_standard_sd=False,
        thresholds={0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor=20,
        scale_range=(1.0, 0.5),
        smooth_attentions=True,
        sigma=0.5,
        kernel_size=3,
        sd_2_1=True,
        normal_prompt=normal_prompt,
        detailed_prompt=prompt,
    )
    
    if isinstance(outputs, tuple):
        return outputs[0]
    elif isinstance(outputs, list):
        return outputs[0]
    else:
        return outputs.images[0]


def generate_img2img_anomalyany(pipe, image, mask, prompt, seed, steps, guidance):
    """AnomalyAny img2img生成"""
    from utils.ptp_utils import register_attention_control, AttentionStore
    
    controller = AttentionStore()
    register_attention_control(pipe, controller)
    
    tokens = pipe.tokenizer(prompt)['input_ids']
    token_indices = get_token_indices(prompt, pipe.tokenizer)
    
    token_idx_to_word = {
        idx: pipe.tokenizer.decode(t)
        for idx, t in enumerate(tokens)
        if 0 < idx < len(tokens) - 1
    }
    
    print(f"  增强关键词: {[token_idx_to_word.get(i, i) for i in token_indices]}")
    
    normal_prompt = prompt.replace(" with ", " that is ")
    
    generator = torch.Generator('cuda').manual_seed(seed)
    
    # Resize到512
    original_size = image.size
    image = image.resize((512, 512), Image.Resampling.LANCZOS)
    mask = mask.resize((512, 512), Image.Resampling.LANCZOS)
    
    outputs = pipe(
        prompt=prompt,
        attention_store=controller,
        indices_to_alter=token_indices,
        attention_res=16,
        guidance_scale=guidance,
        generator=generator,
        num_inference_steps=steps,
        max_iter_to_alter=steps // 2,
        run_standard_sd=False,
        thresholds={0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor=20,
        scale_range=(1.0, 0.5),
        smooth_attentions=True,
        sigma=0.5,
        kernel_size=3,
        sd_2_1=True,
        normal_prompt=normal_prompt,
        detailed_prompt=prompt,
        init_image=image,
        mask_image=mask,
    )
    
    result = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
    if hasattr(result, 'images'):
        result = result.images[0]
    elif isinstance(result, list):
        result = result[0]
    
    # 恢复原始大小
    result = result.resize(original_size, Image.Resampling.LANCZOS)
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="通用缺陷生成脚本 - 支持txt2img和img2img",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # SD txt2img
  python generate_defects.py --mode txt2img --pipe sd \\
      --prompt "wind turbine blade with crack"
  
  # SD img2img
  python generate_defects.py --mode img2img --pipe sd \\
      --image normal.png --mask mask.png
  
  # AnomalyAny txt2img
  python generate_defects.py --mode txt2img --pipe anomalyany \\
      --prompt "wind turbine blade with crack"
  
  # AnomalyAny img2img
  python generate_defects.py --mode img2img --pipe anomalyany \\
      --image normal.png --mask mask.png
        """
    )
    
    # 必选参数
    parser.add_argument("--mode", type=str, required=True, choices=["txt2img", "img2img"],
                        help="生成模式: txt2img(文字生图) 或 img2img(图生图)")
    parser.add_argument("--pipe", type=str, required=True, choices=["sd", "anomalyany"],
                        help="管道: sd(普通SD) 或 anomalyany(CLIP增强)")
    
    # 模型参数
    parser.add_argument("--model-path", type=str, 
                        default="/home/cn/yolo/AnomalyAny/sd-2-1-base",
                        help="SD模型路径")
    parser.add_argument("--lora-path", type=str, default=None,
                        help="LoRA权重路径（可选）")
    parser.add_argument("--lora-bg", type=str, default=None,
                        help="背景LoRA权重路径（双LoRA模式）")
    parser.add_argument("--lora-defect", type=str, default=None,
                        help="缺陷LoRA权重路径（双LoRA模式）")
    parser.add_argument("--alpha", type=float, default=0.7,
                        help="缺陷LoRA权重（控制缺陷强度，默认0.7）")
    parser.add_argument("--beta", type=float, default=1.0,
                        help="背景LoRA权重（控制背景保真度，默认1.0）")
    
    # txt2img参数
    parser.add_argument("--prompt", type=str,
                        help="文字描述（txt2img模式必选）")
    
    # img2img参数
    parser.add_argument("--image", type=str,
                        help="输入图片路径（img2img模式必选）")
    parser.add_argument("--mask", type=str,
                        help="mask图片路径（img2img模式必选）")
    
    # 生成参数
    parser.add_argument("--output", type=str, default="output.png",
                        help="输出路径")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--steps", type=int, default=30,
                        help="推理步数")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="guidance scale")
    
    args = parser.parse_args()
    
    # 验证参数
    if args.mode == "txt2img" and not args.prompt:
        parser.error("--prompt 在txt2img模式下必选")
    
    if args.mode == "img2img" and (not args.image or not args.mask):
        parser.error("--image 和 --mask 在img2img模式下必选")
    
    # 加载模型
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.lora_bg or args.lora_defect:
        pipe = load_dual_lora_pipeline(
            args.model_path,
            lora_bg_path=args.lora_bg,
            lora_defect_path=args.lora_defect,
            alpha=args.alpha,
            beta=args.beta,
            pipe_type=args.pipe,
            device=device
        )
    elif args.pipe == "sd":
        pipe = load_sd_pipeline(args.model_path, args.lora_path, device)
    else:
        pipe = load_anomalyany_pipeline(args.model_path, args.lora_path, device)
    
    # 生成图片
    print("\n" + "=" * 60)
    print(f"生成配置:")
    print(f"  模式: {args.mode}")
    print(f"  管道: {args.pipe}")
    print(f"  种子: {args.seed}")
    print(f"  步数: {args.steps}")
    print(f"  Guidance: {args.guidance}")
    print("=" * 60)
    
    if args.mode == "txt2img":
        print(f"\n生成txt2img: {args.prompt}")
        
        if args.pipe == "sd":
            result = generate_txt2img_sd(pipe, args.prompt, args.seed, args.steps, args.guidance)
        else:
            result = generate_txt2img_anomalyany(pipe, args.prompt, args.seed, args.steps, args.guidance)
            
    else:  # img2img
        print(f"\n生成img2img:")
        print(f"  图片: {args.image}")
        print(f"  Mask: {args.mask}")
        
        image = Image.open(args.image).convert("RGB")
        mask = Image.open(args.mask).convert("L")
        
        prompt = args.prompt if args.prompt else "damaged wind turbine blade"
        
        if args.pipe == "sd":
            result = generate_img2img_sd(pipe, image, mask, prompt, args.seed, args.steps, args.guidance)
        else:
            result = generate_img2img_anomalyany(pipe, image, mask, prompt, args.seed, args.steps, args.guidance)
    
    # 保存结果
    result.save(args.output)
    print(f"\n保存至: {args.output}")
    print("=" * 60)
    print("✅ 生成完成!")


if __name__ == "__main__":
    main()
