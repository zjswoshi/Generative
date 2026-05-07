<div align="center">

# AnomalyAny — 风机叶片缺陷智能生成与检测系统

基于 CVPR 2025 论文 *"Unseen Visual Anomaly Generation"* 的工程化实现

[论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Unseen_Visual_Anomaly_Generation_CVPR_2025_paper.pdf) · [项目主页](https://hansunhayden.github.io/AnomalyAny.github.io/) · [脚本详细文档](scripts/README.md) · [论文原始文档](docs/README.md)

</div>

---

## 项目简介

本项目在 AnomalyAny 学术框架基础上，构建了面向**风机叶片缺陷**的端到端生成-检测系统，解决真实缺陷数据极度稀缺的问题。

核心能力：
- 仅需**一张正常样本图片**，即可生成文本描述对应的逼真异常图像
- 集成 **VLM（视觉语言模型）** 自动生成高质量训练数据描述
- 支持 **txt2img / img2img** 双模式，**SD / AnomalyAny** 双管道
- 从异常生成到 YOLOv11-seg 实例分割检测的完整闭环

---

## 项目结构

```
AnomalyAny/
│
├── README.md                                    ← 本文件（项目总览）
├── docs/README.md                               ← AnomalyAny 论文原始文档
├── scripts/README.md                            ← 风机管线脚本详细文档
│
├── ═══ 核心生成引擎 ═══
│   ├── clip_pipeline_attend_and_excite.py       CLIP增强的Attend-and-Excite生成Pipeline
│   ├── clip_loss.py                             CLIP损失函数（方向/全局/纹理）
│   └── clip_utils/text_templates.py             CLIP文本模板
│
├── ═══ 评估指标 ═══
│   └── metrics/
│       ├── blip_captioning_and_clip_similarity.py
│       ├── compute_clip_similarity.py
│       └── imagenet_utils.py
│
├── ═══ 工具库 ═══
│   └── utils/
│       ├── ptp_utils.py                         Prompt-to-Prompt注意力工具
│       ├── gaussian_smoothing.py                高斯/均值平滑
│       ├── mask_utils.py                        椭圆Mask生成
│       ├── fg_extraction.py                     前景提取
│       └── vis_utils.py                         注意力热力图可视化
│
├── ═══ 风机叶片缺陷生成管线 ═══
│   └── scripts/
│       ├── prepare_hq_datasets.py               VLM增强的高质量数据集生成
│       ├── train_lora_unified.py                统一LoRA训练（单阶段/两阶段/自动）
│       ├── balance_dataset.py                   类别均衡
│       └── generate_defects.py                  通用缺陷生成（4种模式）
│
├── ═══ 数据集归档 ═══
│   └── dataset_archive/
│       ├── classified/processed_dataset/        (11505条)
│       └── classified/processed_dataset_v17/    (25127条)
│
├── ═══ 辅助工具 ═══
│   └── resplit_dataset.py                       分层抽样数据集重分割
│
└── ═══ 下游检测 ═══
    └── train_yolo_stable.py                     YOLOv11-seg实例分割训练
```

---

## 端到端流程

```
原始数据 (YOLO/COCO)
    │
    ├──→ resplit_dataset.py ──→ 分层抽样 80:10:10 分割
    │
    ├──→ prepare_hq_datasets.py ──→ 高质量数据集
    │    ├── full_images/       完整叶片图片
    │    ├── full_masks/        缺陷区域mask
    │    ├── patches/           缺陷patch
    │    └── metadata.jsonl     VLM生成的详细caption
    │         │
    │         ↓
    │    train_lora_unified.py (--mode auto)
    │    ├── 有 full_images → 两阶段训练（叶片外观 → 缺陷特征）
    │    └── 只有 patches  → 单阶段训练（缺陷纹理）
    │         │
    │         ↓
    │    generate_defects.py (4种模式生成)
    │         │
    │         ↓
    │    balance_dataset.py (可选，类别均衡)
    │
    └────→ train_yolo_stable.py ──→ YOLOv11-seg 实例分割
```

---

## 快速开始

### 环境安装

```bash
conda env create -f env.yml
```

需要 Python 3.7+、CUDA 11.6+。

### 1. 数据准备

```bash
python scripts/prepare_hq_datasets.py \
    --dataset /path/to/yolo_dataset \
    --label-map "0:DQ,1:TL,2:LW" \
    --output-dir ./hq_output \
    --use-vlm
```

使用 `--use-vlm` 启用 VLM（Ollama qwen3.5:27b）自动生成包含 Location/Size/Visual Features/Severity 的结构化 caption。不启用时使用内置的28种缺陷类型默认描述。

### 2. LoRA 训练

```bash
# 自动检测模式（推荐）
python scripts/train_lora_unified.py \
    --data-dir ./hq_output \
    --mode auto

# 两阶段训练
python scripts/train_lora_unified.py \
    --data-dir ./hq_output \
    --mode two-stage \
    --stage1-steps 2000 \
    --stage2-steps 3000

# 单阶段训练
python scripts/train_lora_unified.py \
    --data-dir ./hq_output \
    --mode single \
    --max-steps 3000
```

### 3. 缺陷生成

```bash
# txt2img + AnomalyAny（CLIP引导，质量优先）
python scripts/generate_defects.py \
    --mode txt2img \
    --pipe anomalyany \
    --prompt "wind turbine blade with paint peeling near leading edge" \
    --lora-path ./outputs/lora_unified/two_stage/stage2/final \
    --output ./output.png

# img2img + SD（局部替换，精确控制）
python scripts/generate_defects.py \
    --mode img2img \
    --pipe sd \
    --image normal_blade.png \
    --mask defect_mask.png \
    --lora-path ./outputs/lora_unified/single/final \
    --output ./result.png
```

### 4. 下游检测训练

```bash
python train_yolo_stable.py
```

---

## 生成模式选择

| | SD Pipeline | AnomalyAny Pipeline |
|---|---|---|
| **txt2img** | 快速文字生图，大批量 | CLIP引导，更精确 |
| **img2img** | 局部替换，保持背景 | CLIP引导+局部替换，最高质量 |

---

## 支持的缺陷类型（28种）

| 代码 | 描述 | 代码 | 描述 |
|------|------|------|------|
| Normal | 正常 | BQ | 补漆痕 |
| WR | 污染/油污 | DQ | 脱漆 |
| TL | 脱层 | BX | 腐蚀 |
| LJ | 雷击 | KL | 边缘开裂 |
| LW | 裂纹 | JSQ | 接闪器 |
| SH | 结构损伤 | HH | 划痕 |
| LVJPS | 铝尖破损 | LVJ | 铝尖完好 |
| LJZS | 雷击烧痕 | FYZ | 雨罩 |
| FYZTL | 雨罩脱落 | ZG/ZG1/ZG2/ZG3 | 增效件 |
| ZGTL | 增效件脱落 | SZBJ | 数字标记 |
| JTBJ | 箭头标记 | CTBJ | 条纹标记 |
| SLBJ | 沙漏标记 | SJBJ | 三角标记 |
| SZJBJ | 十字标记 | HHBJ | 混合标记 |

---

## 技术依赖

| 类别 | 技术 |
|------|------|
| 生成框架 | 🤗Diffusers + Stable Diffusion 2.1 |
| 注意力控制 | Prompt-to-Prompt + Attend-and-Excite |
| CLIP | OpenCLIP ViT-L/14 + OpenAI CLIP ViT-B/16 + RN50 |
| VLM | Ollama qwen3.5:27b |
| 视觉语言 | Salesforce BLIP |
| LoRA | PEFT |
| 检测 | Ultralytics YOLOv11-seg |
| 深度学习 | PyTorch + CUDA + AMP |

---

## 致谢

- 生成管线基于 [🤗Diffusers](https://github.com/huggingface/diffusers)，融合了 [Prompt-to-Prompt](https://github.com/google/prompt-to-prompt/) 和 [Attend-and-Excite](https://github.com/yuval-alaluf/Attend-and-Excite) 的实现
- CLIP 损失部分参考了 [StyleCLIP](https://github.com/orpatashnik/StyleCLIP)

## 引用

```bibtex
@inproceedings{sun2025unseen,
  title={Unseen Visual Anomaly Generation},
  author={Sun, Han and Cao, Yunkang and Dong, Hao and Fink, Olga},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={25508--25517},
  year={2025}
}
```
