"""
AnomalyAny - Unseen Visual Anomaly Generation
============================================

This script demonstrates how to use the AnomalyAny pipeline for generating anomalies.

Usage:
------
1. Basic usage with default settings:
   python ANOMALY_ANY_USAGE.py

2. The script will generate an anomaly image based on the prompts.

Key Parameters:
--------------
- prompt: The anomaly prompt describing the desired anomaly
- normal_prompt: A normal version of the object
- detailed_prompt: A detailed description of the anomaly
- token_indices: The indices of tokens to alter in the prompt
- num_inference_steps: Number of denoising steps (default: 30)
- guidance_scale: Guidance scale for generation (default: 7.5)
"""

import pprint
import torch
from PIL import Image
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add current directory to path
sys.path.insert(0, '/home/cn/yolo/AnomalyAny')

from clip_pipeline_attend_and_excite import RelationalAttendAndExcitePipeline
from utils.ptp_utils import register_attention_control, AttentionStore


def generate_anomaly(
    anomaly_prompt: str,
    normal_prompt: str,
    detailed_prompt: str,
    token_indices: list,
    output_path: str = "./outputs/anomaly_result.png",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42
):
    """
    Generate an anomaly image using AnomalyAny.

    Args:
        anomaly_prompt: Prompt describing the anomaly (e.g., "a photo of a table that is faded")
        normal_prompt: Normal version of the object (e.g., "a photo of a table")
        detailed_prompt: Detailed anomaly description
        token_indices: Token indices to alter (usually the anomaly word)
        output_path: Where to save the result
        num_inference_steps: Quality vs speed tradeoff
        guidance_scale: How closely to follow the prompt
        seed: Random seed for reproducibility

    Returns:
        PIL.Image: The generated anomaly image
    """

    print("=" * 60)
    print("AnomalyAny - Anomaly Generation")
    print("=" * 60)

    # 1. Load model
    print("\n[1/5] Loading Stable Diffusion model...")
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Use local SD 2.1 model (converted from checkpoint)
    model_path = "/home/cn/yolo/AnomalyAny/sd-2-1-base"

    # Use PNDM scheduler for compatibility (SD 2.1 default is Euler which has issues)
    from diffusers import PNDMScheduler
    pipe = RelationalAttendAndExcitePipeline.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        scheduler=PNDMScheduler.from_pretrained(model_path, subfolder="scheduler"),
    )
    pipe = pipe.to(device)
    print("  ✓ Model loaded (SD 2.1 Base with PNDM scheduler)")

    # 2. Setup prompts
    print("\n[2/5] Setting up prompts...")
    print(f"  Anomaly: {anomaly_prompt}")
    print(f"  Normal: {normal_prompt}")

    # Get token indices
    tokens = pipe.tokenizer(anomaly_prompt)['input_ids']
    token_idx_to_word = {
        idx: pipe.tokenizer.decode(t)
        for idx, t in enumerate(tokens)
        if 0 < idx < len(tokens) - 1
    }
    print(f"  Altering token #{token_indices[0]}: '{token_idx_to_word[token_indices[0]]}'")

    # 3. Setup controller
    print("\n[3/5] Setting up attention controller...")
    controller = AttentionStore()
    register_attention_control(pipe, controller)
    print("  ✓ Controller ready")

    # 4. Generate
    print("\n[4/5] Generating anomaly image...")
    print(f"  Steps: {num_inference_steps}, Guidance: {guidance_scale}")

    generator = torch.Generator('cuda').manual_seed(seed)
    import time
    start = time.time()

    outputs = pipe(
        prompt=anomaly_prompt,
        attention_store=controller,
        indices_to_alter=token_indices,
        attention_res=16,
        guidance_scale=guidance_scale,
        generator=generator,
        num_inference_steps=num_inference_steps,
        max_iter_to_alter=num_inference_steps // 2,
        run_standard_sd=False,
        thresholds={0: 0.05, 10: 0.5, 20: 0.8},
        scale_factor=20,
        scale_range=(1.0, 0.5),
        smooth_attentions=True,
        sigma=0.5,
        kernel_size=3,
        sd_2_1=True,  # Using SD 2.1 model
        normal_prompt=normal_prompt,
        detailed_prompt=detailed_prompt
    )

    elapsed = time.time() - start
    print(f"  ✓ Done in {elapsed:.1f}s")

    # 5. Save
    print("\n[5/5] Saving image...")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Handle different output formats
    if isinstance(outputs, tuple):
        image = outputs[0]
    elif isinstance(outputs, list):
        image = outputs[0]
    else:
        image = outputs.images[0]

    # Ensure image is not a list
    while isinstance(image, list):
        image = image[0]

    image.save(output_path)
    print(f"  ✓ Saved to: {output_path}")

    print("\n" + "=" * 60)
    print("✅ Generation Complete!")
    print("=" * 60)

    return image


if __name__ == "__main__":
    # Example: Generate a faded table
    generate_anomaly(
        anomaly_prompt="a wrench that is rusty",
        normal_prompt="a wrench",
        detailed_prompt="a metal wrench with rust spots and corrosion marks visible on the metallic surface due to oxidation and prolonged exposure to moisture",
        token_indices=[4],  # Index of "rusty" (after filtering special tokens)
        output_path="./outputs/anomaly_wrench_rusty.png",
        num_inference_steps=30,
        guidance_scale=7.5,
        seed=42
    )
