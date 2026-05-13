"""
=================================================================
高质量风机叶片缺陷数据集生成脚本
=================================================================

【脚本用途】
为Stable Diffusion微调生成高质量的训练数据集。

【核心流程】
1. 从YOLO标注中提取：
   - 完整叶片图片 + 缺陷mask（用于训练"完整叶片+缺陷位置"的关系）
   - 缺陷patch（用于学习局部纹理）
2. VLM双输入分析：
   - 输入1: 完整图片 + mask → 分析缺陷位置和上下文
   - 输入2: 缺陷patch → 分析视觉特征
3. 综合生成包含Location/Size/Visual Features/Severity的caption

【输出结构】
output_dir/
├── metadata.jsonl              # 元数据（完整描述）
├── full_images/                # 完整叶片图片（调整大小）
│   ├── blade_00001.png
│   └── blade_00002.png
├── full_masks/                  # 缺陷mask（与full_images对应）
│   ├── blade_00001_mask.png
│   └── blade_00002_mask.png
└── patches/                     # 缺陷patch
    ├── defect_00001.png
    └── defect_00002.png

【元数据格式】
{
  "file_name": "full_images/blade_00001.png",      # 完整叶片图片
  "mask_name": "full_masks/blade_00001_mask.png",   # 缺陷mask（用于VLM定位）
  "patch_name": "patches/defect_00001.png",         # 缺陷patch
  "text": "wind turbine blade with paint peeling near leading edge,
          red paint loss exposing white primer over 10cm area, moderate severity",
  "defect_type": "DQ",
  "location": "leading edge, near tip",
  "size": "moderate (~10cm)",
  "visual_features": "red paint peeling exposing white primer, jagged edges",
  "severity": "moderate"
}

【VLM输入说明】
VLM会接收：
1. 完整叶片图片（标注了缺陷位置）
2. 缺陷区域的裁剪patch
同时分析，确保生成的caption既包含位置信息，又包含视觉特征

【使用方法】
# 基础用法
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
    --use-vlm \
    --vlm-sample 50

【Caption生成规则】
生成的caption包含以下维度：
- Location: 缺陷在叶片上的位置（leading edge, trailing edge, tip, root, surface）
- Size: 缺陷大小（minor <5cm, moderate 5-15cm, severe >15cm）
- Visual Features: 视觉特征（颜色、纹理、形状等）
- Severity: 严重程度（minor, moderate, severe）

示例输出：
"wind turbine blade with moderate paint peeling near leading edge tip,
red paint loss exposing white primer over 10cm area, jagged edges, moderate severity"

【注意事项】
1. 需要YOLO格式标注（.txt文件包含bbox坐标）
2. 需要Ollama服务运行qwen3.5:27b模型
3. 掩码为白色背景（255）上的黑色缺陷区域（0）

【与其他脚本的关系】
- 输出可用于 train_lora_txt2img.py 进行LoRA训练
- 掩码可用于后续的img2img微调
"""

import os
import json
import argparse
import base64
import time
import math
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image, ImageDraw, ImageFilter
from tqdm import tqdm
import numpy as np
import requests


BLADE_DEFECT_INFO = {
    'Normal': 'normal blade surface without any defects or damage',
    'BQ': 'painting trace or touch-up mark on blade surface after repair',
    'WR': 'surface contamination with black stains, dust accumulation or oil pollution',
    'DQ': 'paint peeling with red paint loss exposing white primer underneath',
    'TL': 'gel coat delamination exposing brown or yellow fiberglass layer without damage',
    'BX': 'fiberglass corrosion with exposed and damaged fiberglass layer, common at leading/trailing edges',
    'LJ': 'lightning strike damage with charred burn marks causing cracks or breakage',
    'KL': 'leading and trailing edge cracking with blade surface splitting apart',
    'LW': 'crack damage with horizontal, vertical or irregular cracks, jagged rough edges',
    'JSQ': 'lightning receptor device installed on blade surface, usually black or gray circular device',
    'SZBJ': 'number marking printed on blade surface, usually 1-2 digits',
    'ZG': 'power optimization component type 0 attached to blade, various shapes possible',
    'SH': 'structural damage with spar cap fracture or large area shell tearing',
    'LVJPS': 'aluminum tip broken with tip shell missing or detached, cracked edges',
    'LJZS': 'lightning burn marks near receptor area, black charred spots from lightning strike',
    'JTBJ': 'arrow marking on blade surface',
    'CTBJ': 'stripe or elongated marking on blade surface',
    'SLBJ': 'hourglass shaped marking on blade surface',
    'SJBJ': 'triangle shaped marking on blade surface',
    'SZJBJ': 'cross or plus shaped marking on blade surface',
    'HHBJ': 'mixed marking combining patterns and numbers on blade surface',
    'FYZ': 'rain shield protective cover installed on blade',
    'FYZTL': 'rain shield detached or showing damage',
    'ZG1': 'power optimization component type 1 attached to blade',
    'ZG2': 'power optimization component type 2 attached to blade',
    'HH': 'scratch marks with thin long black lines, smooth texture (vs crack LW)',
    'ZG3': 'power optimization component type 3 attached to blade',
    'LVJ': 'intact aluminum tip at blade tail, metallic material in good condition',
    'ZGTL': 'power optimization component detached or showing obvious damage'
}


def parse_label_map(label_map_str):
    """解析标签映射字符串"""
    label_map = {}
    pairs = label_map_str.split(',')
    for pair in pairs:
        if ':' in pair:
            class_id, label = pair.split(':', 1)
            label_map[int(class_id.strip())] = label.strip()
    return label_map


def get_margin_ratio_for_defect(bbox, mask_area_threshold=0.02, small_margin=0.5, normal_margin=0.15):
    """
    根据缺陷大小动态调整边距比例

    小缺陷（mask面积占比<mask_area_threshold）使用更大的边距，
    以便在patch中保留更多正常纹理信息，提升纹理质量。

    参数:
        bbox: (x_center, y_center, width, height) 归一化坐标
        mask_area_threshold: 小缺陷面积阈值（占图像比例）
        small_margin: 小缺陷使用的边距比例
        normal_margin: 正常缺陷使用的边距比例

    返回:
        float: 边距比例
    """
    bw = bbox[2]
    bh = bbox[3]
    bbox_area_ratio = bw * bh

    if bbox_area_ratio < mask_area_threshold:
        return small_margin
    return normal_margin


def expand_bbox(bbox, img_width, img_height, margin_ratio=0.1, max_size=None):
    """
    扩展bbox并添加边距

    参数:
        bbox: (x_center, y_center, width, height) 归一化坐标
        img_width: 图片宽度
        img_height: 图片高度
        margin_ratio: 边距比例
        max_size: 最大边长

    返回:
        (x1, y1, x2, y2) 像素坐标
    """
    x_center, y_center, w, h = bbox

    # 确保宽高为正
    w = abs(w)
    h = abs(h)

    # 转换为像素坐标
    x_center_px = x_center * img_width
    y_center_px = y_center * img_height
    w_px = w * img_width
    h_px = h * img_height

    # 添加边距
    margin_x = w_px * margin_ratio
    margin_y = h_px * margin_ratio

    x1 = max(0, x_center_px - w_px / 2 - margin_x)
    y1 = max(0, y_center_px - h_px / 2 - margin_y)
    x2 = min(img_width, x_center_px + w_px / 2 + margin_x)
    y2 = min(img_height, y_center_px + h_px / 2 + margin_y)

    # 确保 x1 <= x2, y1 <= y2
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return int(x1), int(y1), int(x2), int(y2)


def extract_defect_patch(img, bbox, target_size=512, margin_ratio=0.15):
    """
    从图片中提取缺陷区域，保持宽高比

    参数:
        img: PIL.Image对象
        bbox: (x_center, y_center, width, height) 归一化坐标
        target_size: 输出图片大小
        margin_ratio: 边距比例

    返回:
        PIL.Image: 裁剪后的缺陷图片（保持宽高比，空白处填充）
    """
    img_width, img_height = img.size
    x1, y1, x2, y2 = expand_bbox(bbox, img_width, img_height, margin_ratio)

    patch = img.crop((x1, y1, x2, y2))

    if patch.size[0] > 0 and patch.size[1] > 0:
        # 保持宽高比resize，使用LANZOS插值
        patch_w, patch_h = patch.size
        ratio = min(target_size / patch_w, target_size / patch_h)
        new_w = int(patch_w * ratio)
        new_h = int(patch_h * ratio)
        patch_resized = patch.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # 创建正方形画布，空白处填充灰色
        patch = Image.new('RGB', (target_size, target_size), (128, 128, 128))
        paste_x = (target_size - new_w) // 2
        paste_y = (target_size - new_h) // 2
        patch.paste(patch_resized, (paste_x, paste_y))

    return patch


def resize_patch_preserve_aspect(patch, short_side=128, max_long_side=512):
    """
    保持宽高比resize patch

    参数:
        patch: PIL.Image对象
        short_side: 短边统一到这个尺寸
        max_long_side: 长边最大不超过这个尺寸

    返回:
        PIL.Image: 保持宽高比的patch
    """
    patch_w, patch_h = patch.size

    if patch_w >= patch_h:
        # 横长型：短边是height
        ratio = short_side / patch_h
        new_h = short_side
        new_w = min(int(patch_w * ratio), max_long_side)
    else:
        # 竖长型：短边是width
        ratio = short_side / patch_w
        new_w = short_side
        new_h = min(int(patch_h * ratio), max_long_side)

    return patch.resize((new_w, new_h), Image.Resampling.LANCZOS)


def extract_patch_by_mask(img, mask, target_size=512, min_mask_ratio=0.02):
    """
    使用mask从图片中提取缺陷patch，保持宽高比

    参数:
        img: PIL.Image对象
        mask: PIL.Image对象 (L模式，白色=缺陷区域)
        target_size: 输出图片大小（已废弃，使用short_side/max_long_side）
        min_mask_ratio: mask面积占图像面积的比例阈值，低于此值则返回None

    返回:
        PIL.Image or None: 裁剪后的缺陷图片（保持宽高比，长边最多512），如果mask太小则返回None
    """
    mask_arr = np.array(mask)
    white_y, white_x = np.where(mask_arr > 128)

    if len(white_y) == 0:
        return None

    # 检查mask面积占比
    mask_ratio = len(white_y) / mask_arr.size
    if mask_ratio < min_mask_ratio:
        return None

    # 取最大连通区域的边界
    from scipy.ndimage import label, find_objects
    labeled_arr, num_features = label(mask_arr > 128)
    if num_features == 0:
        return None

    # 找到最大的连通区域
    largest_region = None
    largest_size = 0
    for i in range(1, num_features + 1):
        region_size = np.sum(labeled_arr == i)
        if region_size > largest_size:
            largest_size = region_size
            largest_region = i

    # 取最大连通区域的边界
    slices = find_objects(labeled_arr == largest_region)[0]
    y1, y2 = slices[0].start, slices[0].stop
    x1, x2 = slices[1].start, slices[1].stop

    # 提取mask区域
    patch = img.crop((x1, y1, x2, y2))

    if patch.size[0] > 0 and patch.size[1] > 0:
        # 保持宽高比resize，短边128，长边按比例最多512
        patch_resized = resize_patch_preserve_aspect(patch, short_side=128, max_long_side=512)
        return patch_resized

    return None


def generate_mask(img_size, bbox, mask_value=255, bg_value=0):
    """
    生成缺陷区域的二值掩码（支持bbox和多边形）

    参数:
        img_size: (width, height)
        bbox:
            - bbox模式：(x_center, y_center, width, height) 归一化坐标
            - polygon模式：list of [x1,y1,x2,y2,...] 归一化多边形坐标
        mask_value: 掩码值（缺陷区域，白色=255）
        bg_value: 背景值（黑色=0）

    返回:
        PIL.Image: 二值掩码
    """
    img_w, img_h = img_size
    mask = Image.new('L', img_size, bg_value)
    draw = ImageDraw.Draw(mask)

    if isinstance(bbox, tuple) and len(bbox) == 4:
        # bbox模式
        x1, y1, x2, y2 = expand_bbox(bbox, img_size[0], img_size[1], margin_ratio=0)
        draw.rectangle([x1, y1, x2, y2], fill=mask_value)
    else:
        # polygon模式：绘制多边形
        coords = list(bbox)
        if len(coords) >= 4 and len(coords) % 2 == 0:
            # 转换归一化坐标到像素坐标
            polygon_points = []
            for i in range(0, len(coords), 2):
                x_px = coords[i] * img_w
                y_px = coords[i + 1] * img_h
                polygon_points.append([x_px, y_px])

            draw.polygon(polygon_points, fill=mask_value)

    return mask


def extract_defect_mask(img, bbox, target_size=512, margin_ratio=0.15):
    """
    提取缺陷掩码（调整为指定大小）
    
    参数:
        img: PIL.Image对象
        bbox: (x_center, y_center, width, height) 归一化坐标
        target_size: 输出掩码大小
        
    返回:
        PIL.Image: 调整后的掩码（白色=缺陷=255，黑色=背景=0）
    """
    img_width, img_height = img.size
    x1, y1, x2, y2 = expand_bbox(bbox, img_width, img_height, margin_ratio)
    
    mask = Image.new('L', img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=255)
    
    mask_patch = mask.crop((x1, y1, x2, y2))
    
    if mask_patch.size[0] > 0 and mask_patch.size[1] > 0:
        mask_patch = mask_patch.resize((target_size, target_size), Image.Resampling.NEAREST)
    
    return mask_patch


def image_to_base64(image):
    """将PIL.Image转换为base64字符串"""
    import io
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')


def create_annotated_image(full_image: Image, bbox, defect_type: str) -> Image:
    """
    在完整图片上标注缺陷位置（红色框）
    
    用于VLM分析时指明缺陷位置
    """
    annotated = full_image.copy()
    draw = ImageDraw.Draw(annotated)
    
    x1, y1, x2, y2 = expand_bbox(bbox, full_image.size[0], full_image.size[1], margin_ratio=0.05)
    draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
    
    return annotated


class OllamaVLMEnhancer:
    """使用Ollama VLM生成详细描述"""
    
    def __init__(self, model_name: str = "qwen3.5:27b", ollama_host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.api_url = f"{ollama_host}/api/generate"
        self.host = ollama_host
        self._check_connection()
    
    def _contains_chinese(self, text: str) -> bool:
        """检测文本是否包含中文字符"""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':
                return True
        return False
    
    def _check_language(self, response: str, max_retries: int = 2) -> str:
        """
        检查响应语言，如果包含中文则尝试清理或返回None
        
        参数:
            response: VLM响应文本
            max_retries: 最大重试次数
            
        返回:
            清理后的响应文本，如果是中文则返回None
        """
        if not response:
            return None
        
        if self._contains_chinese(response):
            print(f"    ⚠️ 检测到中文输出，尝试重新生成...")
            return None
        
        return response
    
    def _check_connection(self):
        """检查Ollama连接"""
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            if response.status_code == 200:
                models = [m['name'] for m in response.json().get('models', [])]
                if self.model_name in models:
                    print(f"  ✅ VLM '{self.model_name}' 已就绪")
                else:
                    print(f"  ⚠️ 未找到模型 '{self.model_name}'")
                    self.model_name = None
        except Exception as e:
            print(f"  ⚠️ 无法连接Ollama: {e}")
            self.model_name = None
    
    def describe_defect_comprehensive(self, full_image: Image, defect_patch: Image, 
                                      bbox, defect_type: str, mask: Image = None) -> str:
        """
        综合分析：同时输入完整图片+缺陷patch，直接输出 SD 微调用 caption
        
        参数:
            full_image: 完整叶片图片（带红色框标注缺陷位置）
            defect_patch: 缺陷区域的裁剪图（能看到细节）
            bbox: 缺陷的边界框
            defect_type: 缺陷类型
            mask: 缺陷mask（可选）
            
        返回:
            str: SD 微调用的标准化 caption
        """
        if not self.model_name:
            return self._default_caption(defect_type)
        
        defect_desc = BLADE_DEFECT_INFO.get(defect_type, f'{defect_type} defect')
        x_center, y_center, width, height = bbox
        
        x_pos = "left" if x_center < 0.33 else ("center" if x_center < 0.66 else "right")
        y_pos = "tip" if y_center < 0.33 else ("middle" if y_center < 0.66 else "root")
        
        prompt = f"""You are creating training captions for Stable Diffusion fine-tuning.
You will receive two images: (1) a complete wind turbine blade with defect marked by red box, and (2) a close-up of the defect area.

Look at BOTH images carefully and describe EXACTLY what you see:

For the BLADE (image 1): What is the blade's overall color and surface condition?
For the DEFECT (image 2): Describe specifically:
- What exact colors are visible in the defect (e.g., "black charred", "white primer", "brown yellow fiberglass")
- What is the defect's shape and size? (e.g., "long vertical crack", "circular patch", "small localized damage")
- What texture does the defect have? (e.g., "rough jagged", "smooth glossy", "dusty")
- How severe does the damage appear visually?

IMPORTANT:
- Do NOT give generic descriptions - describe what you ACTUALLY SEE
- Include specific color descriptions (black, white, red, brown, yellow, gray, etc.)
- Include specific texture words (jagged, smooth, rough, cracked, burnt, etc.)
- End with "professional blade inspection, high detail, realistic"
- Output ONLY the caption, nothing else
- Length: 80-200 characters

Good examples:
"wind turbine blade with weathered white surface, long vertical crack exposing brown yellow fiberglass, rough jagged edges, severe damage, professional blade inspection, high detail, realistic"
"wind turbine blade surface with red paint, circular paint peeling area exposing white primer, rough texture, moderate damage, professional blade inspection, high detail, realistic"

Now describe what you see in these images:"""
        
        try:
            full_base64 = image_to_base64(full_image)
            patch_base64 = image_to_base64(defect_patch)
            
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [full_base64, patch_base64],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200
                    }
                },
                timeout=180
            )
            
            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '').strip()
                
                if not raw_response:
                    return self._default_caption(defect_type)
                
                if self._contains_chinese(raw_response):
                    return self._default_caption(defect_type)
                
                caption = self._clean_caption(raw_response, defect_type)
                if caption:
                    return caption
                    
        except Exception as e:
            pass
        
        return self._default_caption(defect_type)
    
    def describe_defect_patch(self, patch_image: Image, defect_type: str) -> dict:
        """
        描述缺陷patch的局部特征

        返回:
            dict: {
                'visual_features': '缺陷的视觉特征描述',
                'size': '缺陷大小描述',
                'severity': '严重程度'
            }
        """
        if not self.model_name:
            return self._default_defect_info(defect_type)

        defect_desc = BLADE_DEFECT_INFO.get(defect_type, f'{defect_type} defect')

        prompt = f"""Analyze this wind turbine blade defect area image.

Defect type: {defect_desc}

Please provide detailed description:
1. **Visual Features**: Defect colors, textures, shapes, edge characteristics
2. **Size**: Relative defect size (minor <5cm, moderate 5-15cm, severe >15cm)
3. **Severity**: Defect severity level (minor/moderate/severe)

Output format (JSON, IMPORTANT: Use ENGLISH for all values):
{{
    "visual_features": "Visual features description of the defect",
    "size": "Size assessment",
    "severity": "Severity level"
}}

Output JSON only, no other content."""

        try:
            img_base64 = image_to_base64(patch_image)

            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [img_base64],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200
                    }
                },
                timeout=120
            )

            if response.status_code == 200:
                result = response.json()
                caption = result.get('response', '').strip()

                if self._contains_chinese(caption):
                    print(f"    ⚠️ Patch分析：检测到中文输出，使用默认信息")
                    return self._default_defect_info(defect_type)

                return self._parse_vlm_response(caption, defect_type)
        except Exception as e:
            pass

        return self._default_defect_info(defect_type)

    def describe_patch_caption(self, full_image: Image, patch_image: Image,
                                bbox, defect_type: str) -> str:
        """
        为缺陷patch生成caption（包含位置信息）

        用于Stage 2训练 - patch_text需要包含位置信息，这样SD才能学到
        "这个缺陷纹理在叶片的XX位置"

        参数:
            full_image: 完整叶片图片（带红框标注缺陷位置）
            patch_image: 缺陷patch图
            bbox: 缺陷边界框 (x_center, y_center, width, height) 归一化坐标
            defect_type: 缺陷类型

        返回:
            str: 适用于patch训练的caption（包含位置+视觉特征）
        """
        if not self.model_name:
            return self._default_patch_caption(defect_type)

        defect_desc = BLADE_DEFECT_INFO.get(defect_type, f'{defect_type} defect')

        # 根据bbox计算位置
        x_center, y_center, width, height = bbox

        # 横向位置：left/center/right
        if x_center < 0.35:
            h_pos = "left"
        elif x_center < 0.65:
            h_pos = "center"
        else:
            h_pos = "right"

        # 纵向位置：tip/middle/root
        if y_center < 0.35:
            v_pos = "tip"
        elif y_center < 0.65:
            v_pos = "middle"
        else:
            v_pos = "root"

        # 根据defect_type推断常见位置
        defect_location_map = {
            'KL': 'leading or trailing edge',
            'LW': 'blade surface',
            'LJZS': 'blade surface',
            'WR': 'blade surface',
            'LVJPS': 'blade tip',
            'HH': 'blade surface',
            'DQ': 'leading edge or trailing edge',
            'SH': 'blade surface',
            'TL': 'blade surface',
            'LJ': 'blade surface',
        }
        inferred_location = defect_location_map.get(defect_type, 'blade surface')

        # 组合位置描述
        location_desc = f"{inferred_location}, {h_pos} side, {v_pos} area"

        prompt = f"""You are creating training captions for Stable Diffusion fine-tuning.

You will receive TWO images:
1. A complete wind turbine blade with a red box marking the defect location
2. A close-up image of the defect area

Look at BOTH images to understand:
- WHERE on the blade this defect is located (the red box shows the defect position)
- WHAT the defect looks like in detail (the close-up image shows the texture)

Position context: This defect is on the {location_desc} of the blade.

Your task: Describe the defect with BOTH location context AND visual details.
Start with the blade location, then describe the defect.

IMPORTANT RULES:
- Start with the blade position/location (e.g., "leading edge area:", "blade surface near tip:", "trailing edge section:")
- Then describe the visual defect in detail
- Include specific colors (black, white, red, brown, yellow, gray, etc.)
- Include specific textures (jagged, smooth, rough, cracked, burnt, etc.)
- Describe the defect shape and extent (2cm crack, 5cm patch, etc.)
- End with "professional inspection, high detail, realistic"
- Total length: 80-180 characters
- Output ONLY the caption, no explanations

Good examples:
"leading edge area: 3cm long vertical crack exposing dark charred material, rough jagged edges, severe damage, professional inspection, high detail, realistic"
"blade tip section: paint peeling exposing white primer over 4cm area, irregular edges, moderate damage, professional inspection, high detail, realistic"
"trailing edge zone: lightning burn marks with black charred spots, smooth surrounding surface, severe damage, professional inspection, high detail, realistic"

Now describe the defect in this image:"""

        try:
            full_base64 = image_to_base64(full_image)
            patch_base64 = image_to_base64(patch_image)

            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [full_base64, patch_base64],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 200
                    }
                },
                timeout=180
            )

            if response.status_code == 200:
                result = response.json()
                raw_response = result.get('response', '').strip()

                if not raw_response:
                    return self._default_patch_caption(defect_type)

                if self._contains_chinese(raw_response):
                    return self._default_patch_caption(defect_type)

                caption = self._clean_patch_caption(raw_response, defect_type)
                if caption:
                    return caption

        except Exception as e:
            pass

        return self._default_patch_caption(defect_type)

    def _clean_patch_caption(self, raw_response: str, defect_type: str) -> str:
        """清理patch caption输出"""
        caption = raw_response.strip()
        caption = caption.strip('"').strip("'").strip()

        lines = caption.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                caption = line
                break

        # 检查开头是否有效（defect类型相关）
        valid_start = any(caption.lower().startswith(x) for x in [
            'paint', 'jagged', 'gel coat', 'crack', 'lightning', 'scratch',
            'stain', 'contamination', 'shell', 'fracture', 'split', 'broken',
            'burn', 'delamination', 'peeling', 'charring', 'charred', 'exposing',
            'yellow', 'brown', 'white', 'black', 'red', 'gray', 'grey', 'dark',
            'leading', 'trailing', 'aluminum', 'horizontal', 'vertical', 'smooth',
            'large', 'small', 'thin', 'thick', 'irregular', 'rough', 'exposed'
        ])
        if not valid_start:
            return self._default_patch_caption(defect_type)

        # 保留位置信息 - 保留"leading edge", "trailing edge", "blade tip"等结构化位置描述
        # 只移除模糊的绝对位置描述
        import re

        # 保留这些位置关键词（它们是结构化的defect类型描述）
        keep_location_patterns = [
            r'\bleading\s+edge\b',
            r'\btrailing\s+edge\b',
            r'\bblade\s+tip\b',
            r'\bblade\s+root\b',
            r'\bblade\s+surface\b',
            r'\bblade\s+section\b',
            r'\bedge\s+area\b',
            r'\bedge\s+zone\b',
        ]

        # 移除这些模糊位置描述
        remove_location_patterns = [
            r'\bon\s+(left|right|center|middle|upper|lower)\s*(side|tip|root|edge)?',
            r'\bat\s+(left|right|center|middle|upper|lower)\s*(side|tip|root|edge)?',
            r'\bnear\s+(left|right|center|tip|root|edge)',
            r'\b(left|right|center|middle|upper|lower)\s*(side|tip|root)\b',
            r'\b(root|tip)\s+(area|region|section)\b',
        ]

        for pattern in keep_location_patterns:
            if re.search(pattern, caption, re.IGNORECASE):
                # 找到了保留的位置模式，确保它完整保留
                pass

        for pattern in remove_location_patterns:
            caption = re.sub(pattern, '', caption, flags=re.IGNORECASE)

        # 清理多余空格和标点
        caption = re.sub(r'\s+', ' ', caption)
        caption = caption.strip(' ,.')

        # 如果caption过短（<60字符）或几乎是缺陷描述的重复，则认为VLM可能返回了模板
        if len(caption) < 70:
            defect_desc = BLADE_DEFECT_INFO.get(defect_type, '').lower()
            caption_words = set(caption.lower().replace(',', ' ').split())
            desc_words = set(defect_desc.replace(',', ' ').replace('(', ' ').split())
            # 计算重叠率
            overlap = len(caption_words & desc_words) / max(len(caption_words), 1)
            if overlap > 0.6:  # 如果60%以上的词都来自缺陷描述，说明是模板
                return self._default_patch_caption(defect_type)

        if len(caption) > 160:
            caption = caption[:160]

        # 标准化结尾
        if not any(caption.rstrip().endswith(x) for x in ['realistic', 'realistic"']):
            caption = caption.rstrip(',').strip()
            if not caption.endswith('realistic'):
                caption = caption + ", professional inspection, high detail, realistic"

        return caption

    def _default_patch_caption(self, defect_type: str, bbox=None) -> str:
        """生成默认的patch caption（包含位置信息）"""
        location_map = {
            'KL': 'leading edge area',
            'LW': 'blade surface',
            'LJZS': 'blade surface',
            'WR': 'blade surface',
            'LVJPS': 'blade tip section',
            'HH': 'blade surface',
            'DQ': 'leading edge or trailing edge',
            'SH': 'blade surface',
            'TL': 'blade surface',
            'LJ': 'blade surface',
        }
        location = location_map.get(defect_type, 'blade surface')
        desc = BLADE_DEFECT_INFO.get(defect_type, f'{defect_type} damage')
        return f"{location}: {desc}, professional inspection, high detail, realistic"
    
    def describe_blade_with_defect(self, full_image: Image, defect_type: str, 
                                   bbox, img_size) -> dict:
        """
        描述完整叶片及其缺陷位置
        
        返回:
            dict: {
                'location': '缺陷在叶片上的位置',
                'context': '周围环境描述'
            }
        """
        if not self.model_name:
            return self._default_location_info()
        
        defect_desc = BLADE_DEFECT_INFO.get(defect_type, f'{defect_type} defect')
        x_center, y_center, width, height = bbox
        
        x_pos = "left" if x_center < 0.33 else ("center" if x_center < 0.66 else "right")
        y_pos = "tip" if y_center < 0.33 else ("middle" if y_center < 0.66 else "root")
        
        prompt = f"""Analyze this complete wind turbine blade image.

Defect type: {defect_desc}
Defect position: Approximately {x_pos} side horizontally, {y_pos} position vertically

Please describe:
1. **Location**: Specific location on blade (leading edge, trailing edge, tip, root, surface, etc.)
2. **Context**: Surrounding blade surface condition

Output format (JSON, IMPORTANT: Use ENGLISH for all values):
{{
    "location": "Detailed description of defect location",
    "context": "Surrounding environment description"
}}

Output JSON only, no other content."""
        
        try:
            img_base64 = image_to_base64(full_image)
            
            response = requests.post(
                self.api_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [img_base64],
                    "stream": False,
                    "think": False,
                    "options": {
                        "temperature": 0.3,
                        "num_predict": 150
                    }
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                caption = result.get('response', '').strip()
                
                if self._contains_chinese(caption):
                    print(f"    ⚠️ 叶片分析：检测到中文输出，使用默认信息")
                    return self._default_location_info()
                
                return self._parse_location_response(caption)
        except Exception as e:
            pass
        
        return self._default_location_info()
    
    def _parse_vlm_response(self, response: str, defect_type: str) -> dict:
        """解析VLM对缺陷patch的响应"""
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return self._default_defect_info(defect_type)
    
    def _parse_comprehensive_response(self, response: str, defect_type: str) -> dict:
        """解析VLM综合分析响应"""
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return self._default_comprehensive_info(defect_type)
    
    def _parse_location_response(self, response: str) -> dict:
        """解析VLM对位置描述的响应"""
        try:
            import re
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return self._default_location_info()
    
    def _clean_caption(self, raw_response: str, defect_type: str) -> str:
        """清理 VLM 输出的 caption，确保符合 SD 微调格式"""
        caption = raw_response.strip()
        caption = caption.strip('"').strip("'").strip()

        lines = caption.split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) > 10:
                caption = line
                break

        # 允许多种开头：wind turbine blade 或 defect类型开头
        valid_start = any(caption.lower().startswith(x) for x in [
            'wind turbine', 'paint', 'jagged', 'gel coat', 'crack', 'lightning',
            'scratch', 'stain', 'contamination', 'shell', 'fracture', 'split',
            'broken', 'burn', 'delamination', 'peeling', 'charring', 'charred'
        ])
        if not valid_start:
            return self._default_caption(defect_type)

        if len(caption) > 220:
            caption = caption[:220]

        # 标准化结尾
        if not any(caption.rstrip().endswith(x) for x in ['realistic', 'realistic"']):
            caption = caption.rstrip(',').strip()
            if not caption.endswith('realistic'):
                caption = caption + ", professional blade inspection, high detail, realistic"

        return caption
    
    def _default_caption(self, defect_type: str) -> str:
        """生成默认的 SD caption"""
        desc = BLADE_DEFECT_INFO.get(defect_type, f'{defect_type} damage')
        return f"wind turbine blade surface with {desc}, professional blade inspection, detailed"
    
    def _default_comprehensive_info(self, defect_type: str) -> dict:
        """默认综合缺陷信息"""
        desc = BLADE_DEFECT_INFO.get(defect_type, f'{defect_type} damage')
        return {
            'location': 'blade surface',
            'size': 'moderate (5-15cm)',
            'visual_features': desc,
            'severity': 'moderate',
            'context': 'wind turbine blade surface area'
        }

    def _default_defect_info(self, defect_type: str) -> dict:
        """默认缺陷信息"""
        desc = BLADE_DEFECT_INFO.get(defect_type, f'{defect_type} damage')
        return {
            'visual_features': desc,
            'size': 'moderate (5-15cm)',
            'severity': 'moderate'
        }

    def _default_location_info(self) -> dict:
        """默认位置信息"""
        return {
            'location': 'blade surface',
            'context': 'wind turbine blade surface area'
        }


def process_yolo_dataset_hq(dataset_path, label_map, args, metadata):
    """
    处理YOLO格式数据集，生成高质量数据集（随机均衡采样）

    参数:
        dataset_path: 数据集路径
        label_map: 标签映射
        args: 命令行参数
        metadata: 元数据列表

    返回:
        int: 处理总数
    """
    import random
    dataset_path = Path(dataset_path)

    # 输出目录：完整图片和mask
    output_full_images_dir = Path(args.output_dir) / "full_images"
    output_full_masks_dir = Path(args.output_dir) / "full_masks"
    # 输出目录：缺陷patch
    output_patches_dir = Path(args.output_dir) / "patches"

    output_full_images_dir.mkdir(parents=True, exist_ok=True)
    output_full_masks_dir.mkdir(parents=True, exist_ok=True)
    output_patches_dir.mkdir(parents=True, exist_ok=True)

    vlm = None
    if getattr(args, 'use_vlm', False):
        vlm = OllamaVLMEnhancer(
            model_name=args.vlm_model,
            ollama_host=args.vlm_host
        )

    target_per_class = args.samples_per_class

    # 第一步：建立所有样本的索引（按类别分组）
    print("  建立样本索引...")
    class_samples = {class_id: [] for class_id in label_map.keys()}

    for split in [args.train_split, args.valid_split]:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        if not images_dir.exists():
            continue

        for label_file in labels_dir.glob("*.txt"):
            img_stem = label_file.stem
            img_dir = images_dir

            # 查找对应的图片
            img_path = None
            for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.PNG']:
                potential = img_dir / f"{img_stem}{ext}"
                if potential.exists():
                    img_path = potential
                    break

            if not img_path:
                continue

            with open(label_file, 'r') as f:
                annotations = f.readlines()

            for ann_idx, line in enumerate(annotations):
                parts = line.strip().split()
                if not parts:
                    continue
                try:
                    class_id = int(parts[0])
                    if class_id not in label_map:
                        continue
                    class_samples[class_id].append((img_path, line))
                except:
                    continue

    # 第二步：从每个类别随机采样
    print("  随机采样...")
    sampled_samples = []
    for class_id, samples in class_samples.items():
        if len(samples) >= target_per_class:
            chosen = random.sample(samples, target_per_class)
        else:
            print(f"    警告: 类别 {label_map[class_id]} 只有 {len(samples)} 个样本（需要 {target_per_class}）")
            chosen = samples
        sampled_samples.extend(chosen)
        print(f"    {label_map[class_id]}: 采样 {len(chosen)} 个")

    # 打乱顺序
    random.shuffle(sampled_samples)
    print(f"\n  共采样 {len(sampled_samples)} 个样本")

    # 第三步：处理采样的样本
    count = 0
    for img_path, ann_line in tqdm(sampled_samples, desc=f"  {args.dataset_name}"):
        parts = ann_line.strip().split()
        if not parts:
            continue

        try:
            class_id = int(parts[0])
            defect_type = label_map[class_id]
            coords = list(map(float, parts[1:]))

            # 读取图片
            img = Image.open(img_path).convert('RGB')
            img_w, img_h = img.size

            # 根据坐标数量判断是 bbox 还是 polygon
            if len(coords) == 4:
                # bbox模式
                bbox = tuple(coords)
                polygon_coords = None
            else:
                # polygon模式：保留多边形坐标用于mask
                polygon_coords = coords
                # 计算bbox用于patch提取
                x_coords = coords[0::2]
                y_coords = coords[1::2]
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                bbox = (
                    (x_min + x_max) / 2,
                    (y_min + y_max) / 2,
                    x_max - x_min,
                    y_max - y_min
                )

            # 调整完整图片大小并生成mask
            img_resized = img.resize((args.resolution, args.resolution), Image.Resampling.LANCZOS)
            new_w, new_h = args.resolution, args.resolution

            # 缩放bbox用于patch提取
            bbox_resized = (
                bbox[0] * new_w / img_w,
                bbox[1] * new_h / img_h,
                bbox[2] * new_w / img_w,
                bbox[3] * new_h / img_h
            )

            # 缩放polygon坐标用于mask - 直接传递原始归一化坐标
            # generate_mask内部会将归一化坐标乘以img_size得到像素坐标
            if polygon_coords:
                # polygon_coords已经是归一化坐标，不需要再缩放
                polygon_for_mask = polygon_coords
            else:
                polygon_for_mask = None

            mask = generate_mask((new_w, new_h), polygon_for_mask if polygon_for_mask else bbox_resized)

            # 提取缺陷patch - 使用mask的非零区域
            # 如果mask太小（占比<2%），返回None，跳过此样本
            patch = extract_patch_by_mask(img_resized, mask, args.patch_size, min_mask_ratio=0.02)
            if patch is None:
                print(f"\n  跳过样本 (mask太小)")
                continue

            # 保存文件
            full_image_path = output_full_images_dir / f"{args.dataset_name}_blade_{count:05d}.png"
            full_mask_path = output_full_masks_dir / f"{args.dataset_name}_blade_{count:05d}_mask.png"
            patch_path = output_patches_dir / f"{args.dataset_name}_defect_{count:05d}.png"

            img_resized.save(full_image_path)
            mask.save(full_mask_path)
            patch.save(patch_path)

            # VLM生成caption
            # 1. 完整叶片的caption（用于Stage 1）- 包含位置信息
            full_caption = None
            vlm_called_full = False
            if vlm:
                try:
                    full_caption = vlm.describe_defect_comprehensive(
                        img_resized, patch, bbox_resized, defect_type, mask
                    )
                    vlm_called_full = True
                except Exception as e:
                    print(f"\n  ⚠️ VLM处理失败: {e}")

            if full_caption is None:
                full_caption = f"wind turbine blade surface with {BLADE_DEFECT_INFO.get(defect_type, defect_type)} damage, professional blade inspection, detailed"

            # 2. Patch的caption（用于Stage 2）- 包含位置信息
            patch_caption = None
            vlm_called_patch = False
            if vlm:
                try:
                    patch_caption = vlm.describe_patch_caption(img_resized, patch, bbox_resized, defect_type)
                    vlm_called_patch = True
                except Exception as e:
                    print(f"\n  ⚠️ Patch VLM处理失败: {e}")

            if patch_caption is None:
                patch_caption = f"{BLADE_DEFECT_INFO.get(defect_type, defect_type)}, professional inspection, detailed"

            if count < 3:
                print(f"\n  [样本 {count}] {defect_type}:")
                print(f"    Full caption: {full_caption[:80]}...")
                print(f"    Patch caption: {patch_caption[:80]}...")

            metadata.append({
                'file_name': str(full_image_path.relative_to(args.output_dir)),
                'mask_name': str(full_mask_path.relative_to(args.output_dir)),
                'patch_name': str(patch_path.relative_to(args.output_dir)),
                'text': full_caption,           # Stage 1用（包含位置）
                'patch_text': patch_caption,     # Stage 2用（纯视觉特征）
                'defect_type': defect_type
            })

            count += 1

        except Exception as e:
            print(f"\n  警告: 处理标注失败: {e}")
            continue

    return count


def main():
    parser = argparse.ArgumentParser(
        description="高质量风机叶片缺陷数据集生成",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="数据集路径（YOLO格式）")
    parser.add_argument("--label-map", type=str, required=True,
                        help="缺陷类别映射，格式: 'class_id:label,...'")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="输出目录")
    
    parser.add_argument("--dataset-name", type=str, default=None,
                        help="数据集名称")
    parser.add_argument("--patch-size", type=int, default=512,
                        help="缺陷patch大小，默认512")
    parser.add_argument("--resolution", type=int, default=512,
                        help="完整图片分辨率，默认512")
    parser.add_argument("--margin-ratio", type=float, default=0.15,
                        help="缺陷区域边距比例，默认0.15")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="最大样本数量（已废弃，使用--samples-per-class）")
    parser.add_argument("--samples-per-class", type=int, default=20,
                        help="每个类别采样的样本数量，默认20")
    parser.add_argument("--train-split", type=str, default="train",
                        help="训练集目录名")
    parser.add_argument("--valid-split", type=str, default="val",
                        help="验证集目录名")
    
    parser.add_argument("--use-vlm", action="store_true",
                        help="使用VLM生成详细描述")
    parser.add_argument("--vlm-model", type=str, default="qwen3.5:27b",
                        help="VLM模型名称")
    parser.add_argument("--vlm-host", type=str, default="http://localhost:11434",
                        help="Ollama服务地址")
    
    args = parser.parse_args()
    
    if args.dataset_name is None:
        args.dataset_name = Path(args.dataset).name.replace(" ", "_")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    label_map = parse_label_map(args.label_map)
    
    print("=" * 70)
    print("高质量数据集生成")
    print("=" * 70)
    print(f"数据集: {args.dataset}")
    print(f"输出目录: {args.output_dir}")
    print(f"Patch大小: {args.patch_size}")
    print(f"边距比例: {args.margin_ratio}")
    print(f"VLM增强: {'启用' if args.use_vlm else '禁用'}")
    print("=" * 70)
    
    metadata = []
    
    total = process_yolo_dataset_hq(
        args.dataset, label_map, args, metadata
    )
    
    metadata_path = Path(args.output_dir) / "metadata.jsonl"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for item in metadata:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)
    print(f"生成样本: {total} 个")
    print(f"完整图片: {args.output_dir}/full_images/")
    print(f"完整mask: {args.output_dir}/full_masks/")
    print(f"缺陷patch: {args.output_dir}/patches/")
    print(f"元数据文件: {metadata_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()