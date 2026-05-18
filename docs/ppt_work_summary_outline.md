
# 工作周总结 PPT 大纲
# 学术风格，16页，三者并重

---

## Slide 1: 封面
- **标题**：工作周总结与研究进展
- **副标题**：Stable Diffusion 原理理解、问题分析与后续规划
- **汇报人**：[您的姓名]
- **日期**：2026年5月18日
- **项目**：异常样本生成与人脸修复

---

## Slide 2: 工作概述与目标
### 本周核心目标
1. 生成「人物玩手机」的真实数据
2. 解决现有数据集中的人脸马赛克修复问题

### 本周工作流程
1. 问题发现与理论分析
2. 模型对比实验
3. 人脸修复系统实现
4. 后续规划

---

## Slide 3: 问题发现：LoRA 方案的局限性
### 初始方案
- 基于开源「人物玩手机」数据集
- 对 SD2.1 进行 LoRA 微调
- 期望生成真实的「人物玩手机」数据

### 实验结果
- ❌ 无论标准 SD 还是 AnomalyAny 都无法达到预期
- ❌ SD2.1 生成的图片总是带有动漫风格
- ❌ 真实感不足

### 转向：深入理论分析

---

## Slide 4: 理论复盘（1/3）：SD 整体架构
### Stable Diffusion 三大核心组件
1. **Text Encoder**：CLIP ViT-L/14
   - 将文本转换为 embedding
2. **UNet**：条件去噪网络
   - 核心：交叉注意力层注入文本条件
3. **VAE**：变分自编码器
   - 压缩图像到 latent 空间

### 关键文件
- [clip_pipeline_attend_and_excite.py](file:///workspace/clip_pipeline_attend_and_excite.py)

---

## Slide 5: 理论复盘（2/3）：Text-to-Image Pipeline
### 文字生图完整流程
1. **文本编码**
   - Prompt → CLIP Text Encoder → Text Embeddings
2. **噪声初始化**
   - 随机高斯噪声作为起点
3. **去噪循环**（50-100步）
   - UNet 预测噪声
   - CFG 应用
   - Scheduler 更新
4. **VAE 解码**
   - Clean Latent → VAE → 最终图像

---

## Slide 6: 理论复盘（3/3）：Img2Img 与 Inpainting
### Image-to-Image 核心思想
- 先加噪，后去噪
- 关键参数：denoising_strength

### Inpainting 核心技术
- **Masked Noise Blending**
```python
noise_source_latents = scheduler.add_noise(latents_source, noise, t)
latents = (latents * latent_mask) + (noise_source_latents * (1 - latent_mask))
```
- **同步加噪**：避免边界生硬

---

## Slide 7: 深入理解：AnomalyAny Pipeline
### AnomalyAny 在 SD 流程中的位置
- **在每个标准去噪步之前**插入优化
- 两层优化：
  1. 内循环优化（×12次）
  2. 早期额外优化（前10步）

### 三大损失函数
1. **L_att**：注意力损失
2. **L_img**：CLIP损失（后期开启）
3. **L_prompt**：文本嵌入损失

---

## Slide 8: 核心函数分析：_update_latent()
### 作用：根据损失优化 latents
### 核心代码
```python
grad_cond = torch.autograd.grad(
    loss.requires_grad_(True),
    [latents],
    retain_graph=True
)[0]
latents = latents - step_size * grad_cond
```
### 关键点
- **latents 变为可训练变量**
- **完全在 latent 空间操作**
- **不改变 SD 预训练权重**

**位置**：[clip_pipeline_attend_and_excite.py](file:///workspace/clip_pipeline_attend_and_excite.py#L732-L790)

---

## Slide 9: 核心结论（1/2）：模型上限问题
### LoRA vs AnomalyAny 本质对比
| 维度 | LoRA | AnomalyAny |
|------|------|------------|
| 本质 | 训练（改变权重） | 测试时优化 |
| 数据需求 | 10-100张 | 0张（纯文本） |
| 知识来源 | 训练样本教给模型 | 完全依赖基础模型 |
| 创造能力 | 有限 | ❌ 不能创造新知识 |

### 关键结论
- **LoRA 与 AnomalyAny 都是锦上添花**
- **它们的上限完全由基础模型决定**

---

## Slide 10: 核心结论（2/2）：SD2.1 的问题
### SD2.1 天生缺陷
- **训练数据中动漫内容占比过高**
- **生成图片总是带有动漫风格**
- **工业/真实感任务不适合**

### 验证
- 测试 RV1.5、SDXL、ZImage → 真实感均显著优于 SD2.1

---

## Slide 11: 模型对比实验
### 实验设置
- 仅使用原始 text-to-image pipeline
- 相同 prompt：「人物玩手机」
- 不使用 LoRA 或 AnomalyAny

### 实验结果
| 模型 | 真实感评分 | 动漫风格程度 | 备注 |
|------|-----------|------------|------|
| SD2.1 | ⭐ | 高 | 基线 |
| Realistic Vision 1.5 | ⭐⭐⭐⭐⭐ | 低 | 显著更好 |
| SDXL | ⭐⭐⭐⭐ | 中 | 显著更好 |
| ZImage | ⭐⭐⭐⭐⭐ | 低 | 显著更好 |

---

## Slide 12: 人脸修复工作（1/2）：问题与方案
### 问题背景
- 目标数据集中大部分人脸被打马赛克
- 需要高质量修复

### 当前方案
- **受限于新模型架构不熟悉**
- 暂用 SD2.1 inpainting + ZImage 二阶段处理

### 具体流程
1. 马赛克检测：统计连续灰色区域
2. 阶段1：SD2.1 inpainting 修复
3. 阶段2：ZImage 轻微写实化处理

---

## Slide 13: 人脸修复工作（2/2）：现有问题与改进方向
### 现有问题
1. **马赛克检测有漏检**
   - 仅靠简单统计，鲁棒性不足
   - 解决方案：二次修复
2. **SD2.1 修复结果仍有动漫风**
   - 二阶段 ZImage 可部分缓解
   - 但如果阶段1太差，会被放大

### 改进方向
- **优先方向**：直接使用 ZImage/Flux 的 inpainting
- 更好的马赛克检测算法

---

## Slide 14: 后续规划（1/2）：近期任务（1-2周）
### 任务1：验证新模型的 inpainting 能力
- ZImage inpainting 测试
- Flux inpainting 测试
- 对比 SD2.1 vs ZImage vs Flux
- **时间**：1-2天

### 任务2：改进人脸修复系统
- 实现 ZImage/Flux inpainting
- 改进马赛克检测算法
- 优化二阶段流程
- **时间**：2-3天

### 任务3（可选）：训练新模型的 LoRA
- 如果新模型效果已很好，收集少量数据微调 LoRA
- **时间**：3-5天

---

## Slide 15: 后续规划（2/2）：长期探索（1-2月）
### AnomalyAny 迁移研究
1. **SDXL 迁移**（⭐⭐⭐ 优先）
   - 仍然是 UNet 架构，迁移相对容易
   - **时间**：1-2周
2. **Flux 迁移**（⭐⭐ 高级研究）
   - 纯 Transformer 架构（DiT）
   - 需要重新设计注意力提取
   - **时间**：3-4周

### 策略
- **并行探索**：新模型直接用 vs AnomalyAny 迁移
- **如果新模型 LoRA 已足够好，可放弃 AnomalyAny**

---

## Slide 16: 总结与 Q&amp;A
### 本周总结
1. 深入理解了 SD 三种 Pipeline 与 AnomalyAny
2. 得出核心结论：基础模型决定上限
3. 实现了人脸修复原型
4. 验证了新模型的优势

### 核心要点
- ✅ SD2.1 不适合真实感任务
- ✅ LoRA 与 AnomalyAny 都是锦上添花
- ✅ 优先使用更好的基础模型

### Q&amp;A
- 感谢老师/评委
- 提问与解答

---

## 附录（可选补充）
### 附录 A：关键文件索引
- [clip_pipeline_attend_and_excite.py](file:///workspace/clip_pipeline_attend_and_excite.py)
- [utils/ptp_utils.py](file:///workspace/utils/ptp_utils.py)
- [scripts/train_lora_unified.py](file:///workspace/scripts/train_lora_unified.py)

### 附录 B：参考文献
- Rombach et al. (CVPR 2022)
- Sun et al. (CVPR 2025)
