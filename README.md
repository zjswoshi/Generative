<div align="center">

# Industrial Defect Generation — 工业缺陷智能生成与检测系统

基于 CVPR 2025 论文 *"Unseen Visual Anomaly Generation"* 的通用工业异常样本生成框架

[论文](https://openaccess.thecvf.com/content/CVPR2025/papers/Sun_Unseen_Visual_Anomaly_Generation_CVPR_2025_paper.pdf) · [项目主页](https://hansunhayden.github.io/AnomalyAny.github.io/) · [脚本详细文档](scripts/README.md) · [论文原始文档](docs/README.md)

</div>

---

## 项目简介

本项目将 AnomalyAny（CVPR 2025）学术框架工程化，构建了一套**通用工业缺陷智能生成与检测系统**，可适配风电叶片、金属零件、锂电池、纺织布料、PCB、钢铁冶金等多种工业场景。

核心能力：
- **任意工业场景**：准备好标注数据，快速适配新缺陷类型
- **VLM 智能标注**：自动生成高质量结构化描述，无需人工撰写 Caption
- **灵活生成模式**：4 种模式矩阵（txt2img/img2img × SD/AnomalyAny），覆盖从快速批生成到精准编辑
- **端到端闭环**：缺陷生成 → LoRA 微调 → 检测训练 → 实例分割

---

## 项目结构

```
IndustrialDefectGeneration/
│
├── README.md                                    ← 本文件（项目总览）
├── docs/README.md                               ← AnomalyAny 论文原始文档
├── scripts/README.md                            ← 管线脚本详细文档
│
├── ═══ 核心生成引擎 ═══
│   ├── clip_pipeline_attend_and_excite.py       CLIP增强的Attend-and-Excite生成Pipeline
│   ├── clip_loss.py                             CLIP损失函数（方向/全局/纹理）
│   └── clip_utils/text_templates.py              CLIP文本模板
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
├── ═══ 工业缺陷生成管线 ═══
│   └── scripts/
│       ├── prepare_hq_datasets.py               VLM增强的高质量数据集生成
│       ├── train_lora_unified.py               统一LoRA训练（单阶段/两阶段/自动）
│       ├── balance_dataset.py                   类别均衡
│       └── generate_defects.py                  通用缺陷生成（4种模式）
│
├── ═══ 数据集归档 ═══
│   └── dataset_archive/
│       ├── classified/processed_dataset/         (11505条)
│       └── classified/processed_dataset_v17/    (25127条)
│
├── ═══ 辅助工具 ═══
│   └── resplit_dataset.py                       分层抽样数据集重分割
│
└── ═══ 下游检测 ═══
    └── train_yolo_stable.py                    YOLOv11-seg实例分割训练
```

---

## 适用场景

| 行业 | 缺陷类型示例 |
|------|-------------|
| 🌀 风电叶片 | 脱漆、脱层、腐蚀、裂纹、雷击损伤 |
| 🔩 金属零件 | 划痕、凹坑、锈蚀、裂纹、断裂 |
| 🔋 锂电池 | 极片缺陷、隔膜破损、电解液泄漏 |
| 🧵 纺织布料 | 断经、断纬、油污、色差、破洞 |
| 📦 PCB电路板 | 短路、开路、焊点缺陷、元件偏移 |
| 🏭 钢铁冶金 | 表面裂纹、夹杂物、划伤、氧化皮 |
| 🚗 汽车零部件 | 涂装缺陷、装配误差、表面瑕疵 |
| 💎 玻璃制品 | 气泡、划痕、结石、裂纹 |

---

## 端到端流程

```
原始标注数据 (YOLO/COCO)
    │
    ├──→ resplit_dataset.py ──→ 分层抽样 80:10:10 分割
    │
    ├──→ prepare_hq_datasets.py ──→ 高质量数据集
    │    ├── full_images/       完整样本图片 + mask
    │    ├── patches/           缺陷区域patch
    │    └── metadata.jsonl     VLM生成的详细caption
    │         │
    │         ↓
    │    train_lora_unified.py (--mode auto)
    │    ├── 有 full_images → 两阶段训练（正常外观 → 缺陷特征）
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
    --dataset /path/to/industrial_dataset \
    --label-map "0:scratch,1:pitting,2:crack" \
    --output-dir ./hq_output \
    --use-vlm
```

使用 `--use-vlm` 启用 VLM（Ollama qwen3.5:27b）自动生成包含 Location/Size/Visual Features/Severity 的结构化 caption。

### 2. LoRA 训练

```bash
# 自动检测模式（推荐）
python scripts/train_lora_unified.py \
    --data-dir ./hq_output \
    --mode auto

# 两阶段训练（适用于 txt2img 完整生成）
python scripts/train_lora_unified.py \
    --data-dir ./hq_output \
    --mode two-stage \
    --stage1-steps 2000 \
    --stage2-steps 3000

# 单阶段训练（适用于 img2img 局部替换）
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
    --prompt "metal surface with scratch damage" \
    --lora-path ./outputs/lora_unified/two_stage/stage2/final \
    --output ./result.png

# img2img + SD（局部替换，精确控制）
python scripts/generate_defects.py \
    --mode img2img \
    --pipe sd \
    --image normal_product.png \
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

## 内置缺陷类型示例（以风电叶片为例）

内置的 28 种缺陷描述模板，可按需替换为其他工业场景的缺陷定义。

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
