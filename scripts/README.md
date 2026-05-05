# 风机叶片缺陷生成管线 - 脚本说明

## 📁 目录结构

```
scripts/
├── 数据准备工具
│   ├── prepare_datasets.py        # 数据集预处理
│   └── extract_defect_patches.py  # 缺陷Patch提取
│
├── 数据平衡工具
│   └── balance_dataset.py        # 类别均衡（自动生成补充样本）
│
├── LoRA训练
│   ├── train_lora_txt2img.py    # 两阶段LoRA训练（用于txt2img）
│   └── train_lora_img2img.py    # 单阶段LoRA训练（用于img2img）
│
└── 缺陷生成
    └── generate_defects.py       # 通用缺陷生成（支持多种模式）
```

---

## 🎯 完整管线流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                     1. 数据准备阶段                            │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ prepare_datasets.py                                        │
│ 将原始YOLO/COCO数据集转换为统一的训练格式                    │
│                                                              │
│ 输入: 多个原始数据集 (支持YOLO/COCO格式)                     │
│ 输出: processed_dataset/ (normal/ + defect/ + metadata.jsonl) │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────────┐
│ extract_defect_patches.py                                    │
│ 从数据集中提取缺陷区域作为Patch，用于img2img训练                 │
│                                                              │
│ 输入: 原始数据集 (YOLO/COCO格式)                             │
│ 输出: defect_patches/ (images/ + masks/)                   │
└──────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────────────┐
│                    2. LoRA训练阶段                              │
└─────────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ train_lora_txt2img.py (两阶段训练)                            │
│ ├── Stage 1: 学习正常叶片外观                               │
│ └── Stage 2: 学习缺陷特征                                   │
│ 输出: outputs/lora_stage2_defect/final (用于txt2img生成)        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ train_lora_img2img.py (单阶段训练)                         │
│ 直接学习缺陷Patch纹理                                      │
│ 输出: outputs/defect_patch_lora/final (用于img2img生成)      │
└───────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────┐
│                   3. 数据平衡阶段 (可选)                    │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ balance_dataset.py                                         │
│ 自动诊断类别分布，生成补充少数类样本                         │
│                                                              │
│ 输入: 不均衡的数据集                                         │
│ 输出: 均衡的数据集 + 生成的补充样本                            │
└─────────────────────────────────────────────────────────────┘

                              ↓

┌─────────────────────────────────────────────────────────────┐
│                   4. 缺陷生成阶段                            │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ generate_defects.py (通用生成)                              │
│ ├── 支持txt2img和img2img两种模式                            │
│ ├── 支持SD和AnomalyAny两种管道                            │
│ └── 支持LoRA权重加载                                        │
│ 输出: 生成的缺陷图片                                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 📖 脚本详细说明

### 1️⃣ prepare_datasets.py - 数据集预处理

**功能**: 将任意格式的YOLO/COCO数据集转换为统一的LoRA训练格式

**支持格式**:
- YOLO格式: `train/labels/*.txt`
- COCO格式: `train/_annotations.coco.json`

**使用方法**:

```bash
# 处理YOLO数据集
python prepare_datasets.py \
    --dataset /path/to/yolo_dataset \
    --label-map "0:crack,1:damage,2:erosion" \
    --output-dir ./processed_dataset \
    --dataset-name windturbine

# 处理COCO数据集
python prepare_datasets.py \
    --dataset /path/to/coco_dataset \
    --label-map "1:crack,2:damage" \
    --output-dir ./processed_dataset
```

**输出格式**:
```
processed_dataset/
├── metadata.jsonl          # 每行: {"file_name": "xxx.png", "text": "caption"}
├── normal/                 # 正常样本图片
└── defect/                # 缺陷样本图片
```

---

### 2️⃣ extract_defect_patches.py - 缺陷Patch提取

**功能**: 从标注数据集中提取缺陷区域（带mask），用于img2img训练

**使用方法**:

```bash
# 提取缺陷Patch
python extract_defect_patches.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage" \
    --output-dir ./defect_patches \
    --patch-size 256 \
    --context-padding 0.3
```

**输出格式**:
```
defect_patches/
├── dataset_name/
│   ├── train/
│   │   ├── images/           # 裁剪的缺陷Patch
│   │   └── masks/            # 对应的mask
│   └── metadata.json
```

**参数说明**:
- `--patch-size`: Patch大小，默认256
- `--context-padding`: 缺陷周围上下文比例，默认0.3（30%）

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
- `--target-ratio`: 目标类别比例（0-1），各类别样本数将为最大类的该比例
- `--min-samples`: 每类最小样本数
- `--dry-run`: 只分析不生成
- `--generate-mode`: 生成模式（txt2img或img2img）

---

### 4️⃣ train_lora_txt2img.py - 两阶段LoRA训练（txt2img）

**功能**: 两阶段训练LoRA，先学叶片外观，再学缺陷特征，用于txt2img生成

**训练阶段**:
- Stage 1: 用正常叶片学习"什么是叶片外观"
- Stage 2: 用缺陷叶片学习"缺陷长什么样"（在Stage 1基础上）

**使用方法**:

```bash
# 完整两阶段训练
python train_lora_txt2img.py \
    --stage1-steps 2000 \
    --stage2-steps 3000 \
    --stage1-lr 1e-4 \
    --stage2-lr 5e-5

# 跳过Stage 1（使用已有权重）
python train_lora_txt2img.py --skip-stage1

# 自定义参数
python train_lora_txt2img.py \
    --stage1-steps 1500 \
    --stage2-steps 2000 \
    --lora-rank 16
```

**输出**:
```
outputs/
├── lora_stage1_blade/final     # Stage 1权重
└── lora_stage2_defect/final     # Stage 2权重（用于txt2img生成）
```

**参数说明**:
- `--stage1-steps`: Stage 1训练步数，默认2000
- `--stage2-steps`: Stage 2训练步数，默认3000
- `--lora-rank`: LoRA rank，默认16
- `--stage1-lr`: Stage 1学习率，默认1e-4
- `--stage2-lr`: Stage 2学习率，默认5e-5（更小，避免遗忘Stage 1知识）

---

### 5️⃣ train_lora_img2img.py - 单阶段LoRA训练（img2img）

**功能**: 单阶段训练LoRA，直接学习缺陷Patch纹理，用于img2img生成

**特点**:
- 使用Patch训练（256x256）
- 只学缺陷纹理，不学叶片外观
- 适合img2img局部替换任务

**使用方法**:

```bash
# 基本用法
python train_lora_img2img.py \
    --data-dir ./defect_patches \
    --output-dir ./outputs/defect_patch_lora \
    --max-train-steps 3000

# 组合多个数据集
python train_lora_img2img.py \
    --data-dirs ./patches/dataset1 ./patches/dataset2 \
    --output-dir ./outputs/defect_patch_lora

# 自定义参数
python train_lora_img2img.py \
    --data-dir ./defect_patches \
    --output-dir ./outputs/defect_patch_lora \
    --lora-rank 16 \
    --learning-rate 1e-4 \
    --max-train-steps 3000
```

**参数说明**:
- `--data-dir`: 缺陷Patch数据目录
- `--patch-size`: Patch大小，默认256
- `--lora-rank`: LoRA rank，默认16
- `--learning-rate`: 学习率，默认1e-4

---

### 6️⃣ generate_defects.py - 通用缺陷生成

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
    --lora-path ./outputs/lora_stage2_defect/final \
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
    --lora-path ./outputs/lora_path \
    --output result.png
```

**参数说明**:

通用参数:
- `--mode`: 生成模式（txt2img或img2img）
- `--pipe`: 生成管道（sd或anomalyany）
- `--lora-path`: LoRA权重路径
- `--seed`: 随机种子，默认42
- `--steps`: 推理步数，默认30
- `--guidance`: guidance scale，默认7.5

txt2img参数:
- `--prompt`: 文字描述

img2img参数:
- `--image`: 输入图片路径
- `--mask`: mask图片路径（白色区域为生成位置）
- `--prompt`: 文字描述（可选）

---

## 🔧 管道选择

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

| LoRA | 训练脚本 | 适用模式 | 说明 |
|------|---------|---------|------|
| 两阶段LoRA | train_lora_txt2img.py | txt2img | 学习完整叶片+缺陷知识 |
| Patch LoRA | train_lora_img2img.py | img2img | 只学习缺陷纹理 |

---

## 📊 典型工作流

### 工作流1: 完整训练 + 生成（推荐）

```bash
# 1. 准备数据集
python prepare_datasets.py \
    --dataset /path/to/raw_dataset \
    --label-map "0:crack,1:damage,2:erosion" \
    --output-dir ./processed_dataset

# 2. 训练LoRA（两阶段）
python train_lora_txt2img.py \
    --stage1-steps 2000 \
    --stage2-steps 3000 \
    --output-dir ./outputs

# 3. 生成缺陷图片
python generate_defects.py \
    --mode txt2img \
    --pipe sd \
    --prompt "wind turbine blade with crack" \
    --lora-path ./outputs/lora_stage2_defect/final \
    --output ./generated.png
```

### 工作流2: img2img专用训练 + 生成

```bash
# 1. 提取Patch
python extract_defect_patches.py \
    --dataset /path/to/dataset \
    --label-map "0:crack" \
    --output-dir ./patches

# 2. 训练Patch LoRA
python train_lora_img2img.py \
    --data-dir ./patches \
    --output-dir ./outputs/patch_lora

# 3. img2img生成
python generate_defects.py \
    --mode img2img \
    --pipe sd \
    --image normal_blade.png \
    --mask defect_mask.png \
    --lora-path ./outputs/patch_lora/final \
    --output result.png
```

### 工作流3: 类别均衡（补充少数类）

```bash
# 分析并生成
python balance_dataset.py \
    --dataset /path/to/dataset \
    --label-map "0:crack,1:damage" \
    --generate-mode txt2img \
    --pipe sd \
    --lora-path ./outputs/lora_path \
    --output-dir ./balanced_dataset
```

---

## ⚙️ 常用参数说明

### LoRA训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lora-rank` | 16 | LoRA秩，值越大表达能力越强，也更容易过拟合 |
| `--lora-alpha` | 16 | LoRA缩放因子，通常设为rank值 |
| `--learning-rate` | 1e-4 | 学习率 |
| `--max-train-steps` | 3000 | 训练步数 |

### 生成参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--seed` | 42 | 随机种子，用于复现 |
| `--steps` | 30 | 推理步数，越高质量越好越慢 |
| `--guidance` | 7.5 | 文本引导强度 |
| `--resolution` | 512 | 图片分辨率 |

---

## 📝 提示

1. **LoRA训练**: 建议先在小规模数据上测试，再大规模训练
2. **类别平衡**: `--dry-run`先分析，再决定生成数量
3. **生成速度**: SD比AnomalyAny快，但质量略低
4. **Patch大小**: 256适合细节，512适合质量优先
5. **Context padding**: extract_patches的padding影响上下文信息量

---

## 🐛 常见问题

**Q: 提示"数据集为空"?**
A: 检查数据集路径和label-map参数是否正确

**Q: 生成图片质量差?**
A: 尝试增加`--steps`和调整`--guidance`

**Q: 内存不足?**
A: 减小`--patch-size`和`--train-batch-size`

**Q: 需要LoRA吗?**
A: 不是必须的，但LoRA能显著提高生成质量

---

## 📚 更多资源

- Stable Diffusion: https://github.com/CompVis/stable-diffusion
- LoRA: https://github.com/microsoft/LoRA
- AnomalyAny: https://github.com/niceimageio/AnomalyAny
- diffusers: https://github.com/huggingface/diffusers
