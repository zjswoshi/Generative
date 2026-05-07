# 风机叶片缺陷生成管线 - 脚本说明

## 📁 目录结构

```
scripts/
├── 数据准备
│   └── prepare_hq_datasets.py     # VLM增强的高质量数据集生成
│
├── LoRA训练
│   └── train_lora_unified.py      # 统一LoRA训练（单阶段/两阶段/自动）
│
├── 数据平衡
│   └── balance_dataset.py         # 类别均衡（自动生成补充样本）
│
└── 缺陷生成
    └── generate_defects.py        # 通用缺陷生成（支持多种模式）
```

---

## 🎯 完整管线流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                     1. 数据准备阶段                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ prepare_hq_datasets.py                                              │
│ 从YOLO标注中提取完整图片+mask+patch，VLM生成详细caption            │
│                                                                     │
│ 输入: YOLO格式数据集 + label-map                                    │
│ 输出: hq_output/                                                    │
│   ├── full_images/      完整叶片图片（调整大小）                    │
│   ├── full_masks/       缺陷区域mask                                │
│   ├── patches/          缺陷patch                                   │
│   └── metadata.jsonl    详细caption（含位置/大小/特征/严重度）      │
└─────────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────────┐
│                     2. LoRA训练阶段                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ train_lora_unified.py (--mode auto)                                 │
│                                                                     │
│ 自动检测数据源，选择训练模式：                                      │
│ ├── 有 full_images → 两阶段训练                                     │
│ │   ├── Stage 1: full_images → 学习叶片外观 (512px, lr=1e-4)       │
│ │   └── Stage 2: patches → 学习缺陷特征 (256px, lr=5e-5)          │
│ └── 只有 patches → 单阶段训练                                       │
│     └── patches → 学习缺陷纹理 (默认512px, lr=1e-4)                │
│                                                                     │
│ 输出: outputs/lora_unified/                                         │
│   ├── two_stage/stage1/final     两阶段Stage1权重                   │
│   ├── two_stage/stage2/final     两阶段Stage2权重（用于txt2img）    │
│   └── single/final               单阶段权重（用于img2img）          │
└─────────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────────┐
│                   3. 数据平衡阶段 (可选)                             │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ balance_dataset.py                                                  │
│ 自动诊断类别分布，生成补充少数类样本                                │
│                                                                     │
│ 输入: 不均衡的数据集                                                │
│ 输出: 均衡的数据集 + 生成的补充样本                                 │
└─────────────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────────┐
│                     4. 缺陷生成阶段                                  │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ generate_defects.py (通用生成)                                      │
│ ├── 支持txt2img和img2img两种模式                                    │
│ ├── 支持SD和AnomalyAny两种管道                                      │
│ └── 支持LoRA权重加载                                                │
│ 输出: 生成的缺陷图片                                                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 📖 脚本详细说明

### 1️⃣ prepare_hq_datasets.py - VLM增强的高质量数据集生成

**功能**: 从YOLO标注数据集中提取完整图片+缺陷mask+缺陷patch，并使用VLM生成详细的结构化caption

**核心特性**:
- 同时输出**完整图片 + 缺陷mask + 缺陷patch**三件套
- 集成 **Ollama VLM**（默认 qwen3.5:27b）自动分析缺陷
- VLM双输入分析：完整图片（带红框标注）+ 缺陷patch
- 生成包含 **Location / Size / Visual Features / Severity** 的结构化描述
- 内置28种风机叶片缺陷类型的专业描述字典
- 自动检测中文输出并回退到默认描述

**使用方法**:

```bash
# 基础用法（不使用VLM，使用默认描述）
python prepare_hq_datasets.py \
    --dataset /path/to/yolo_dataset \
    --label-map "0:DQ,1:TL,2:LW" \
    --output-dir ./hq_output

# VLM增强模式（推荐）
python prepare_hq_datasets.py \
    --dataset /path/to/yolo_dataset \
    --label-map "0:DQ,1:TL,2:LW" \
    --output-dir ./hq_output \
    --use-vlm

# 高级用法（自定义参数）
python prepare_hq_datasets.py \
    --dataset /path/to/dataset \
    --label-map "0:DQ,1:TL,2:LW,3:BX" \
    --output-dir ./hq_output \
    --patch-size 512 \
    --resolution 512 \
    --use-vlm \
    --vlm-model qwen3.5:27b \
    --vlm-host http://localhost:11434
```

**输出格式**:
```
hq_output/
├── metadata.jsonl              # 元数据（详细描述）
├── full_images/                # 完整叶片图片（调整大小）
│   ├── blade_00001.png
│   └── blade_00002.png
├── full_masks/                 # 缺陷mask（与full_images对应）
│   ├── blade_00001_mask.png
│   └── blade_00002_mask.png
└── patches/                    # 缺陷patch
    ├── defect_00001.png
    └── defect_00002.png
```

**元数据格式**:
```json
{
  "file_name": "full_images/blade_00001.png",
  "mask_name": "full_masks/blade_00001_mask.png",
  "patch_name": "patches/defect_00001.png",
  "text": "wind turbine blade with paint peeling near leading edge, moderate severity, red paint loss exposing white primer over 10cm area",
  "defect_type": "DQ",
  "location": "leading edge, near tip",
  "size": "moderate (~10cm)",
  "visual_features": "red paint peeling exposing white primer, jagged edges",
  "severity": "moderate"
}
```

**参数说明**:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--dataset` | 必填 | 数据集路径（YOLO格式） |
| `--label-map` | 必填 | 缺陷类别映射，格式: `0:DQ,1:TL,2:LW` |
| `--output-dir` | 必填 | 输出目录 |
| `--dataset-name` | 自动推断 | 数据集名称 |
| `--patch-size` | 512 | 缺陷patch大小 |
| `--resolution` | 512 | 完整图片分辨率 |
| `--margin-ratio` | 0.15 | 缺陷区域边距比例 |
| `--max-samples` | 无 | 最大样本数量 |
| `--use-vlm` | False | 启用VLM生成详细描述 |
| `--vlm-model` | qwen3.5:27b | VLM模型名称 |
| `--vlm-host` | http://localhost:11434 | Ollama服务地址 |

---

### 2️⃣ train_lora_unified.py - 统一LoRA训练

**功能**: 统一的LoRA训练入口，支持单阶段、两阶段和自动检测三种模式

**训练模式**:

| 模式 | 参数 | 训练内容 | 适用场景 |
|------|------|---------|---------|
| 自动 | `--mode auto` | 检测数据源自动选择 | 推荐 |
| 单阶段 | `--mode single` | 仅 patches → 缺陷纹理 | img2img 局部替换 |
| 两阶段 | `--mode two-stage` | full_images → patches | txt2img 完整生成 |

**使用方法**:

```bash
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
    --stage1-steps 2000 \
    --stage2-steps 3000

# 组合多个数据集
python train_lora_unified.py \
    --data-dirs ./hq_fengji ./hq_offshore ./hq_nordtank \
    --mode two-stage
```

**输出**:
```
outputs/lora_unified/
├── single/                     # 单阶段输出
│   ├── checkpoint-500/
│   ├── checkpoint-1000/
│   └── final/
└── two_stage/                  # 两阶段输出
    ├── stage1/
    │   ├── checkpoint-500/
    │   └── final/
    └── stage2/
        ├── checkpoint-500/
        ├── checkpoint-1000/
        └── final/
```

**参数说明**:

通用参数:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | ./hq_output | 数据目录 |
| `--data-dirs` | 无 | 多个数据目录 |
| `--mode` | auto | 训练模式: auto/single/two-stage |
| `--resolution` | 512 | 图片分辨率 |
| `--use-detailed-caption` | True | 使用metadata中的详细caption |
| `--lora-rank` | 16 | LoRA rank |
| `--lora-alpha` | 16 | LoRA alpha |
| `--train-batch-size` | 2 | 训练batch size |
| `--gradient-accumulation-steps` | 4 | 梯度累积步数 |

两阶段参数:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--stage1-steps` | 2000 | Stage 1训练步数 |
| `--stage1-lr` | 1e-4 | Stage 1学习率 |
| `--stage2-steps` | 3000 | Stage 2训练步数 |
| `--stage2-lr` | 5e-5 | Stage 2学习率（较小，避免遗忘） |

单阶段参数:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--max-steps` | 3000 | 训练步数 |
| `--learning-rate` | 1e-4 | 学习率 |

---

### 3️⃣ balance_dataset.py - 类别均衡

**功能**: 自动诊断数据集类别分布，识别少数类，调用生成脚本补充样本

**使用方法**:

```bash
# 只分析不生成
python balance_dataset.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage" \
    --dry-run

# 分析并生成补充样本
python balance_dataset.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage" \
    --generate-mode txt2img \
    --pipe anomalyany \
    --lora-path ./outputs/lora_path \
    --output-dir ./balanced

# 使用img2img模式
python balance_dataset.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage" \
    --generate-mode img2img \
    --pipe sd \
    --output-dir ./balanced
```

**参数说明**:

| 参数 | 说明 |
|------|------|
| `--target-ratio` | 目标类别比例（0-1），各类别样本数将为最大类的该比例 |
| `--min-samples` | 每类最小样本数 |
| `--dry-run` | 只分析不生成 |
| `--generate-mode` | 生成模式（txt2img或img2img） |

---

### 4️⃣ generate_defects.py - 通用缺陷生成

**功能**: 通用的缺陷生成脚本，支持多种模式和管道

**支持的模式**:

#### txt2img模式（文字生图）

```bash
# SD txt2img
python generate_defects.py \
    --mode txt2img \
    --pipe sd \
    --prompt "a wind turbine blade with crack damage" \
    --output ./output.png

# AnomalyAny txt2img（CLIP增强）
python generate_defects.py \
    --mode txt2img \
    --pipe anomalyany \
    --prompt "a wind turbine blade with crack damage" \
    --lora-path ./outputs/lora_unified/two_stage/stage2/final \
    --output ./output.png
```

#### img2img模式（图生图）

```bash
# SD img2img
python generate_defects.py \
    --mode img2img \
    --pipe sd \
    --image normal_blade.png \
    --mask defect_mask.png \
    --output result.png

# AnomalyAny img2img（CLIP增强）
python generate_defects.py \
    --mode img2img \
    --pipe anomalyany \
    --image normal_blade.png \
    --mask defect_mask.png \
    --prompt "wind turbine blade with crack damage" \
    --lora-path ./outputs/lora_unified/single/final \
    --output result.png
```

**参数说明**:

通用参数:

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--mode` | 必填 | 生成模式（txt2img或img2img） |
| `--pipe` | 必填 | 生成管道（sd或anomalyany） |
| `--lora-path` | 无 | LoRA权重路径 |
| `--seed` | 42 | 随机种子 |
| `--steps` | 30 | 推理步数 |
| `--guidance` | 7.5 | guidance scale |

txt2img参数:

| 参数 | 说明 |
|------|------|
| `--prompt` | 文字描述 |

img2img参数:

| 参数 | 说明 |
|------|------|
| `--image` | 输入图片路径 |
| `--mask` | mask图片路径（白色区域为生成位置） |
| `--prompt` | 文字描述（可选） |

---

## 🔧 管道选择指南

### 何时使用txt2img vs img2img？

| 场景 | 推荐模式 | 说明 |
|------|---------|------|
| 需要生成完整叶片图片 | txt2img | 直接从文字生成完整叶片 |
| 在现有叶片上添加缺陷 | img2img | 保持叶片背景，只替换mask区域 |
| 大量快速生成 | txt2img | 只需要文字，不需要mask |
| 精确控制缺陷位置 | img2img | 通过mask控制生成区域 |

### 何时使用SD vs AnomalyAny？

| 管道 | 特点 | 适用场景 |
|------|------|---------|
| **SD** | 快速生成 | 大批量生成，速度优先 |
| **AnomalyAny** | CLIP引导，更精确 | 质量优先，需要更好局部控制 |

### LoRA选择

| LoRA | 训练模式 | 适用场景 | 说明 |
|------|---------|---------|------|
| 两阶段LoRA | `--mode two-stage` | txt2img | 学习完整叶片+缺陷知识 |
| 单阶段LoRA | `--mode single` | img2img | 只学习缺陷纹理 |

---

## 📊 典型工作流

### 工作流1: VLM增强 + 两阶段训练 + 生成（推荐）

```bash
# 1. 准备高质量数据集（VLM增强）
python prepare_hq_datasets.py \
    --dataset /path/to/raw_dataset \
    --label-map "0:DQ,1:TL,2:LW" \
    --output-dir ./hq_output \
    --use-vlm

# 2. 自动训练LoRA
python train_lora_unified.py \
    --data-dir ./hq_output \
    --mode auto

# 3. 生成缺陷图片
python generate_defects.py \
    --mode txt2img \
    --pipe anomalyany \
    --prompt "wind turbine blade with paint peeling near leading edge" \
    --lora-path ./outputs/lora_unified/two_stage/stage2/final \
    --output ./generated.png
```

### 工作流2: img2img专用训练 + 生成

```bash
# 1. 准备数据集（含patches）
python prepare_hq_datasets.py \
    --dataset /path/to/dataset \
    --label-map "0:DQ" \
    --output-dir ./hq_output

# 2. 单阶段训练
python train_lora_unified.py \
    --data-dir ./hq_output \
    --mode single

# 3. img2img生成
python generate_defects.py \
    --mode img2img \
    --pipe sd \
    --image normal_blade.png \
    --mask defect_mask.png \
    --lora-path ./outputs/lora_unified/single/final \
    --output result.png
```

### 工作流3: 类别均衡（补充少数类）

```bash
# 分析并生成
python balance_dataset.py \
    --dataset /path/to/dataset \
    --label-map "0:DQ,1:TL" \
    --generate-mode txt2img \
    --pipe sd \
    --lora-path ./outputs/lora_path \
    --output-dir ./balanced_dataset
```

---

## ⚙️ 训练参数建议

### 单阶段训练

| 参数 | 建议值 | 说明 |
|------|--------|------|
| resolution | 256 | patch较小，256即可 |
| learning_rate | 1e-4 | 默认值 |
| max_steps | 2000-3000 | 视数据量调整 |

### 两阶段训练

| 参数 | Stage 1 | Stage 2 | 说明 |
|------|---------|---------|------|
| resolution | 512 | 256 | Stage1学全图，Stage2学patch |
| learning_rate | 1e-4 | 5e-5 | Stage2更小，避免遗忘 |
| max_steps | 1500-2000 | 2000-3000 | 视数据量调整 |

### LoRA超参数

| 参数 | 建议值 | 说明 |
|------|--------|------|
| lora_rank | 16 | 推荐值，可调整范围4-32 |
| lora_alpha | =lora_rank | 通常设为rank值 |

---

## 📝 提示

1. **VLM增强**: 推荐启用 `--use-vlm`，生成的caption质量显著高于默认模板
2. **LoRA训练**: 建议先在小规模数据上测试，再大规模训练
3. **类别平衡**: `--dry-run` 先分析，再决定生成数量
4. **生成速度**: SD比AnomalyAny快，但质量略低
5. **自动模式**: `--mode auto` 会根据数据目录结构自动选择最佳训练模式

---

## 🐛 常见问题

**Q: VLM连接失败？**
A: 确保Ollama服务已启动且模型已下载：`ollama pull qwen3.5:27b`

**Q: 提示"数据集为空"？**
A: 检查数据集路径和label-map参数是否正确

**Q: 生成图片质量差？**
A: 尝试增加 `--steps` 和调整 `--guidance`，或使用AnomalyAny管道

**Q: 内存不足？**
A: 减小 `--resolution` 和 `--train-batch-size`

**Q: 需要LoRA吗？**
A: 不是必须的，但LoRA能显著提高生成质量

---

## 📚 更多资源

- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- LoRA: https://github.com/microsoft/LoRA
- AnomalyAny: https://github.com/niceimageio/AnomalyAny
- diffusers: https://github.com/huggingface/diffusers
- Ollama: https://ollama.ai
