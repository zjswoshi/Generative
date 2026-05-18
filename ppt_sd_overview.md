
# Stable Diffusion 三种核心模式详解

## 目录
1. [Text-to-Image (txt2img) - 文字生图](#1-text-to-image-txt2img---文字生图)
2. [Image-to-Image (img2img) - 图生图](#2-image-to-image-img2img---图生图)
3. [Inpainting - 局部重绘](#3-inpainting---局部重绘)

---

## 1. Text-to-Image (txt2img) - 文字生图

### 整体流程
```
Text Prompt → CLIP Text Encoder → Text Embedding
                                           ↓
Random Noise Latent → UNet → Noise Prediction → Scheduler → Clean Latent → VAE Decoder → Generated Image
```

### 代码位置
`StableDiffusionPipeline` (Diffusers官方)

### 关键步骤详解

#### 1.1 文本编码
```python
# 将输入文本转换为词嵌入
text_inputs = tokenizer(prompt, return_tensors="pt")
text_embeddings = text_encoder(**text_inputs).last_hidden_state
```
- 输入：文本描述
- 输出：77×768 的词嵌入向量

#### 1.2 初始化随机噪声
```python
# 生成初始噪声
latents = torch.randn(
    (batch_size, num_channels_latent, height // 8, width // 8),
    generator=generator,
    device=device,
    dtype=dtype
)
```

#### 1.3 去噪循环（核心）
```python
for t in timesteps:
    # 缩放 latent
    latent_model_input = scheduler.scale_model_input(latents, t)
    
    # UNet 预测噪声
    noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
    
    # CFG (Classifier-Free Guidance)
    if do_classifier_free_guidance:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    
    # Scheduler 更新 latent
    latents = scheduler.step(noise_pred, t, latents).prev_sample
```

#### 1.4 VAE 解码
```python
# 将 latent 解码为像素空间
image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
```

### CFG 公式
```
noise_pred = noise_pred_uncond + guidance_scale × (noise_pred_text - noise_pred_uncond)
```
- `guidance_scale=7.5`：强烈遵循文本提示
- `guidance_scale=1`：几乎随机
- `guidance_scale=20`：过度拟合

---

## 2. Image-to-Image (img2img) - 图生图

### 整体流程
```
Input Image → VAE Encoder → Clean Latent → Add Noise → Noisy Latent
                                                              ↓
                                                     UNet → Scheduler → Clean Latent → VAE Decoder → Generated Image
```

### 核心思想
**先加噪，再去噪**

### 代码位置
`StableDiffusionImg2ImgPipeline`

### 关键步骤详解

#### 2.1 图像编码
```python
# 将输入图像转换为 latent
init_image = image.resize((width, height))
init_image = preprocess(init_image).to(device, dtype)

init_latent = vae.encode(init_image).latent_dist.sample(generator=generator)
init_latent = init_latent * vae.config.scaling_factor
```

#### 2.2 决定加噪程度
```python
# denoising_strength 控制加噪程度
# 0.0 = 几乎不变，1.0 = 完全重绘
t_start = int(num_inference_steps * (1 - denoising_strength))
timesteps = scheduler.timesteps[t_start:]
```

#### 2.3 加噪
```python
# 给原始 latent 加噪声
noise = torch.randn_like(init_latent)
latents = scheduler.add_noise(init_latent, noise, timesteps[0])
```

#### 2.4 去噪循环
```python
for i, t in enumerate(timesteps):
    # 和 txt2img 相同的去噪流程
    ...
```

### 加噪公式
```
noisy_latent = clean_latent × sqrt(1 - β_t) + noise × sqrt(β_t)
```

---

## 3. Inpainting - 局部重绘

### 整体流程
```
Input Image + Mask → VAE Encoder → Clean Latents (源 + 去噪)
                                                              ↓
                    ┌─────────────────────────────────────────┐
                    │  每个去噪步：                            │
                    │  1. UNet 预测噪声                       │
                    │  2. Scheduler 更新去噪 latents          │
                    │  3. Masked Blending：                   │
                    │     latents = (去噪后 × mask) + (原始加噪 × (1-mask))│
                    └─────────────────────────────────────────┘
                                                              ↓
                                        VAE Decoder → Inpainted Image
```

### 核心思想
**只修改 mask 指定的区域，其他区域保持原样**

### 代码位置
`StableDiffusionInpaintPipeline`

### 关键步骤详解

#### 3.1 准备两个 latents
```python
# latents_source：保存原始图像，不做去噪
latents_source = image2latent(vae, init_image, width, height, device, generator)

# latents：用于去噪
latents = latents_source.clone()
latents = scheduler.add_noise(latents, noise, t_start)

# 读取并预处理 mask
latent_mask, _ = read_mask(mask_image, dest_size=(width//8, height//8))
```

#### 3.2 核心：Masked Noise Blending（在每个去噪步后）
```python
for i, t in enumerate(timesteps):
    # 1. 标准 SD 去噪
    latents = scheduler.step(noise_pred, t, latents).prev_sample
    
    # 2. 给原始图像也加同步噪声
    noise_source_latents = scheduler.add_noise(latents_source, torch.randn_like(latents), t)
    
    # 3. 混合！（关键！）
    if mask_image is not None:
        latents = (latents * latent_mask) + (noise_source_latents * (1 - latent_mask))
```

### Mask 预处理
```python
def read_mask(mask_path, dest_size=(64, 64)):
    """
    转换为二值化 mask
    - 白色区域(值=1)：需要重绘的区域
    - 黑色区域(值=0)：保持原样的区域
    """
    org_mask = Image.open(mask_path).convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = np.array(mask)
    mask[mask != 0] = 255
    mask = mask / 255
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    return torch.from_numpy(mask).half().to('cuda'), org_mask
```

### 为什么要同步加噪？
```
问题：直接混合会导致边界不连贯
    - 去噪中的 latents：有中等噪声
    - 原始 latents：完全干净
    - 边界处：噪声水平不一致，会产生伪影

解决方案：给原始 latents 也加相同时间步的噪声
    - noise_source_latents = add_noise(latents_source, t)
    - 两边噪声水平一致，边界更平滑
```

---

## 对比表格

| 模式 | 输入 | 输出 | 核心技术 |
|------|------|------|---------|
| txt2img | 文本 | 新图像 | 随机噪声初始化 |
| img2img | 图像 + 文本 | 相似但修改的图像 | 图像加噪后去噪 |
| inpainting | 图像 + mask + 文本 | 局部修改的图像 | Masked Blending |

---

## 关键代码文件
- `clip_pipeline_attend_and_excite.py`：本项目的实现，包含完整去噪循环
- `scripts/generate_defects.py`：三种模式的统一入口

