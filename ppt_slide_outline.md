
# PPT 整体大纲

## 学术风格，逻辑清晰

---

## Slide 1: 封面
- **标题**：Stable Diffusion 与 AnomalyAny 原理详解
- **副标题**：从 txt2img, img2img, inpainting 到 Attend-and-Excite
- **作者**：[你的名字]
- **日期**：[日期]

---

## Slide 2: 目录
1. Stable Diffusion 基础原理
2. Text-to-Image (txt2img)
3. Image-to-Image (img2img)
4. Inpainting（局部重绘）
5. AnomalyAny 论文简介
6. AnomalyAny Pipeline 详解
7. 总结与对比

---

## Part 1: Stable Diffusion 基础

### Slide 3: Stable Diffusion 整体架构
- **三大核心组件**
  - Text Encoder：CLIP ViT-L/14
  - UNet：条件去噪网络
  - VAE：图像编码/解码
- **生成流程**
  ```
  Text → Text Encoder → Text Embedding
  Random Noise → UNet → Noise Prediction → Scheduler → Clean Latent → VAE → Image
  ```
- **代码文件**：`clip_pipeline_attend_and_excite.py`

---

### Slide 4: CFG (Classifier-Free Guidance) - 让模型听话
- **问题**：如何让模型更遵循文本提示？
- **CFG 公式**
  ```
  noise_pred = noise_pred_uncond + guidance_scale × (noise_pred_text - noise_pred_uncond)
  ```
- **参数说明**
  - `guidance_scale=1`：几乎随机
  - `guidance_scale=7.5`：默认，平衡
  - `guidance_scale=20`：过度拟合
- **代码位置**：`clip_pipeline_attend_and_excite.py` 第 1641-1643 行

---

## Part 2: 三种生成模式

### Slide 5: Text-to-Image (txt2img) - 文字生图
- **输入**：文本描述
- **输出**：全新生成的图像
- **流程**
  1. 文本编码
  2. 随机噪声初始化
  3. 去噪循环（50-100 步）
     - UNet 预测噪声
     - CFG
     - Scheduler 更新
  4. VAE 解码
- **核心代码**：`StableDiffusionPipeline`
- **示例**：
  ```python
  prompt = "a wind turbine blade with scratch damage"
  image = pipe(prompt).images[0]
  ```

---

### Slide 6: Image-to-Image (img2img) - 图生图
- **输入**：原始图像 + 文本描述
- **输出**：相似但修改后的图像
- **核心思想**：先加噪，再去噪
- **关键参数**：`denoising_strength`
  - 0.0 = 几乎不变
  - 0.5 = 中等修改
  - 1.0 = 完全重绘
- **流程**
  1. 编码原始图像为 latent
  2. 加噪声到 t_start
  3. 从 t_start 去噪到 t=0
  4. VAE 解码
- **加噪公式**
  ```
  noisy_latent = clean_latent × sqrt(1 - β_t) + noise × sqrt(β_t)
  ```
- **核心代码**：`StableDiffusionImg2ImgPipeline`

---

### Slide 7: Inpainting - 局部重绘（1/2）
- **输入**：原始图像 + Mask + 文本描述
- **输出**：Mask 区域修改后的图像
- **核心思想**：只修改 mask 指定的区域
- **Mask 说明**
  - 白色区域（值=1）：需要重绘
  - 黑色区域（值=0）：保持原样
- **两个 Latents**
  - `latents_source`：保存原始图像，不做去噪
  - `latents`：用于去噪

---

### Slide 8: Inpainting - 局部重绘（2/2）
- **核心技术：Masked Noise Blending**
  - 在每个去噪步后混合
  ```python
  noise_source_latents = scheduler.add_noise(latents_source, noise, t)
  latents = (latents * latent_mask) + (noise_source_latents * (1 - latent_mask))
  ```
- **为什么要同步加噪？**
  - 问题：直接混合干净和去噪图像，边界不连贯
  - 解决：给原始图像也加相同时间步的噪声
  - 结果：两边噪声水平一致，边界更平滑
- **核心代码**：`StableDiffusionInpaintPipeline`
- **代码位置**：`clip_pipeline_attend_and_excite.py` 第 1647-1649 行

---

## Part 3: AnomalyAny

### Slide 9: AnomalyAny 论文简介
- **会议**：CVPR 2025
- **标题**：*Unseen Visual Anomaly Generation*
- **作者**：Han Sun, Yunkang Cao, Hao Dong, Olga Fink
- **核心思想**
  - 只需单个正常样本 + 文本描述
  - 即可生成多样化、真实的未见过的异常样本
- **两大核心技术**
  1. Attention-Guided Anomaly Optimization
  2. Prompt-Guided Anomaly Refinement

---

### Slide 10: 核心技术：Attend-and-Excite
- **原始问题**：SD 有时候会忽略文本中的关键词
- **Attend-and-Excite 解决方案**
  1. 监控文本中关键词的注意力分数
  2. 如果注意力过低，通过迭代优化提升
  3. 确保所有主体概念在图像中都被正确表达
- **在 AnomalyAny 中的应用**
  - 确保缺陷/异常关键词（"defect", "scratch"）被正确生成
- **核心代码**：`_compute_max_attention_per_index()`, `_compute_loss()`

---

### Slide 11: 在 SD 流程中哪个步骤进行操作？
- **对比表格**
  | 标准 SD 去噪循环 | AnomalyAny 去噪循环 |
  |----------------|-------------------|
  | 1. UNet 预测噪声 | 1. **[Attend-and-Excite 优化]**（内循环） |
  | 2. CFG |   - 计算 L_att, L_img |
  | 3. Scheduler 更新 |   - 更新 latents, prompt embeddings |
  |                  | 2. **[早期阶段额外优化]**（前 10 步） |
  |                  | 3. 标准 SD 去噪 |
  |                  |   - UNet, CFG, Scheduler |
  |                  |   - **[Masked Blending]**（inpainting） |

- **关键结论**：AnomalyAny 在「标准 SD 去噪」之前插入了两层优化！

---

### Slide 12: AnomalyAny 去噪循环详解（伪代码）
- **完整流程**
  ```python
  for i, t in enumerate(timesteps):
      # [第一层优化] 内循环（12次）
      for q in range(12):
          L_att = compute_att_loss()
          L_img = compute_clip_loss()  # 后期开启
          latents = update_latent(0.1*L_img + 0.5*L_att)
          prompt_embeds = update_latent(L_prompt)
      
      # [第二层优化] 早期额外优化（前 10 步）
      if i &lt; 10:
          latents = iterative_refinement_step(latents)
      
      # 标准 SD 去噪
      noise_pred = unet(latents, t)
      noise_pred = cfg(noise_pred)
      latents = scheduler.step(noise_pred, t, latents)
      
      # Inpainting：Masked Blending
      if mask:
          latents = latents*mask + noise_source*(1-mask)
  ```
- **代码位置**：`clip_pipeline_attend_and_excite.py` 第 1076 行起

---

### Slide 13: 关键函数：`_update_latent()` - 核心！
- **作用**：根据损失更新 latents
- **核心步骤**
  1. 计算损失对 latents 的梯度
     ```python
     grad_cond = torch.autograd.grad(
         loss.requires_grad_(True),
         [latents],
         retain_graph=True
     )[0]
     ```
  2. 梯度下降
     ```python
     latents = latents - step_size * grad_cond
     ```
- **关键点**
  - latents 必须 `requires_grad=True`
  - 完全在 latent 空间操作，不破坏 SD 流程
- **代码位置**：`clip_pipeline_attend_and_excite.py` 第 732-791 行

---

## Part 4: 总结

### Slide 14: 三种模式对比总结
| 模式 | 输入 | 输出 | 核心技术 |
|------|------|------|---------|
| txt2img | 文本 | 新图像 | 随机噪声初始化 |
| img2img | 图像 + 文本 | 相似但修改的图像 | 图像加噪后去噪 |
| inpainting | 图像 + mask + 文本 | 局部修改图像 | Masked Blending |

---

### Slide 15: AnomalyAny 总结
- **位置**：每个 SD 去噪步之前
- **优化**
  1. 内循环优化（12 次）：latents, prompt embeddings
  2. 早期额外优化（前 10 步）：迭代优化注意力
- **关键创新**
  - 完全在 latent 空间操作
  - latents 变为可训练变量
  - 同时优化图像和文本
- **代码文件**
  - `/workspace/clip_pipeline_attend_and_excite.py`
  - `/workspace/scripts/generate_defects.py`

---

### Slide 16: Q&amp;A
- **提问与解答**
- **感谢**

---

## 附录

### 补充 Slide A1: 代码文件索引
- `/workspace/clip_pipeline_attend_and_excite.py`：核心 Pipeline
- `/workspace/clip_loss.py`：CLIP 损失函数
- `/workspace/scripts/generate_defects.py`：统一生成入口

### 补充 Slide A2: 参考文献
- Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" (CVPR 2022)
- Sun et al. "Unseen Visual Anomaly Generation" (CVPR 2025)
- Attend-and-Excite paper

