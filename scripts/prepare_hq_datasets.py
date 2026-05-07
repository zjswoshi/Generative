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
from PIL import Image, ImageDraw
from tqdm import tqdm
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
    从图片中提取缺陷区域
    
    参数:
        img: PIL.Image对象
        bbox: (x_center, y_center, width, height) 归一化坐标
        target_size: 输出图片大小
        margin_ratio: 边距比例
        
    返回:
        PIL.Image: 裁剪后的缺陷图片
    """
    img_width, img_height = img.size
    x1, y1, x2, y2 = expand_bbox(bbox, img_width, img_height, margin_ratio)
    
    patch = img.crop((x1, y1, x2, y2))
    
    if patch.size[0] > 0 and patch.size[1] > 0:
        patch = patch.resize((target_size, target_size), Image.Resampling.LANCZOS)
    
    return patch


def generate_mask(img_size, bbox, mask_value=255, bg_value=0):
    """
    生成缺陷区域的二值掩码
    
    参数:
        img_size: (width, height)
        bbox: (x_center, y_center, width, height) 归一化坐标
        mask_value: 掩码值（缺陷区域，白色=255）
        bg_value: 背景值（黑色=0）
        
    返回:
        PIL.Image: 二值掩码
    """
    x1, y1, x2, y2 = expand_bbox(bbox, img_size[0], img_size[1], margin_ratio=0)
    
    mask = Image.new('L', img_size, bg_value)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([x1, y1, x2, y2], fill=mask_value)
    
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

Defect type: {defect_desc}
Defect position: Approximately {x_pos} side horizontally, {y_pos} position vertically

Generate ONE clean caption for SD fine-tuning. Follow these rules:
1. Start with "wind turbine blade" (lowercase, no proper nouns)
2. Use simple, clear descriptions a diffusion model can learn
3. Include: defect type, location, visual features, severity
4. End with "professional blade inspection, detailed" 
5. Keep it concise (40-150 characters total)
6. Use lowercase for all words except the first letter
7. Output ONLY the caption, nothing else

Example output format:
"wind turbine blade surface with dark charcoal streaks on leading edge, rough gritty texture, severe damage, professional blade inspection, detailed"

Now analyze the provided images and output your caption:"""
        
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
        
        if not caption.startswith('wind turbine'):
            return self._default_caption(defect_type)
        
        if len(caption) > 200:
            caption = caption[:200]
        
        if not caption.endswith('detailed'):
            if not caption.endswith('inspection'):
                caption = caption.rstrip(',').strip()
        
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
    处理YOLO格式数据集，生成高质量数据集
    
    参数:
        dataset_path: 数据集路径
        label_map: 标签映射
        args: 命令行参数
        metadata: 元数据列表
        
    返回:
        int: 处理总数
    """
    dataset_path = Path(dataset_path)
    count = 0
    
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
    
    for split in [args.train_split, args.valid_split]:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"
        
        if not images_dir.exists():
            continue
        
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
        
        print(f"\n{split}: 找到 {len(image_files)} 张图片")
        
        for img_path in tqdm(image_files, desc=f"  {args.dataset_name}/{split}"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            
            if not label_path.exists():
                continue
            
            try:
                img = Image.open(img_path).convert('RGB')
            except Exception as e:
                print(f"  警告: 无法打开图片 {img_path}: {e}")
                continue
            
            with open(label_path, 'r') as f:
                annotations = f.readlines()
            
            if not annotations:
                continue
            
            for ann_idx, line in enumerate(annotations):
                parts = line.strip().split()
                if not parts:
                    continue
                
                try:
                    class_id = int(parts[0])
                    if class_id not in label_map:
                        continue
                    
                    defect_type = label_map[class_id]
                    coords = list(map(float, parts[1:]))
                    
                    # 根据坐标数量判断是 bbox 还是 polygon
                    if len(coords) == 4:
                        # bbox 格式: x_center, y_center, width, height
                        bbox = tuple(coords)
                    else:
                        # polygon 格式: x1,y1,x2,y2,... 找到 bounding box
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
                    # 调整bbox坐标
                    img_w, img_h = img.size
                    new_w, new_h = args.resolution, args.resolution
                    bbox_resized = (
                        bbox[0] * new_w / img_w,
                        bbox[1] * new_h / img_h,
                        bbox[2] * new_w / img_w,
                        bbox[3] * new_h / img_h
                    )
                    # 生成精确的mask
                    mask = generate_mask((new_w, new_h), bbox_resized)
                    
                    # 提取缺陷patch
                    patch = extract_defect_patch(img_resized, bbox_resized, args.patch_size, args.margin_ratio)
                    
                    # 保存文件
                    full_image_path = output_full_images_dir / f"{args.dataset_name}_blade_{count:05d}.png"
                    full_mask_path = output_full_masks_dir / f"{args.dataset_name}_blade_{count:05d}_mask.png"
                    patch_path = output_patches_dir / f"{args.dataset_name}_defect_{count:05d}.png"
                    
                    img_resized.save(full_image_path)
                    mask.save(full_mask_path)
                    patch.save(patch_path)
                    
                    caption = None
                    vlm_called = False
                    if vlm:
                        try:
                            caption = vlm.describe_defect_comprehensive(
                                img_resized, patch, bbox_resized, defect_type, mask
                            )
                            vlm_called = True
                        except Exception as e:
                            print(f"\n  ⚠️ VLM处理失败: {e}")
                    
                    if caption is None:
                        caption = f"wind turbine blade surface with {BLADE_DEFECT_INFO.get(defect_type, defect_type)} damage, professional blade inspection, detailed"
                    
                    if count < 3:
                        print(f"\n  [样本 {count}] VLM: {'是' if vlm_called else '否'}, caption: {caption[:80]}...")
                    
                    metadata.append({
                        'file_name': str(full_image_path.relative_to(args.output_dir)),
                        'mask_name': str(full_mask_path.relative_to(args.output_dir)),
                        'patch_name': str(patch_path.relative_to(args.output_dir)),
                        'text': caption,
                        'defect_type': defect_type
                    })
                    
                    count += 1
                    
                    if args.max_samples and count >= args.max_samples:
                        return count
                        
                except Exception as e:
                    print(f"\n  警告: 处理标注失败: {e}")
                    continue
            
            if (count > 0 and args.max_samples and count >= args.max_samples):
                break
        
        if count > 0 and args.max_samples and count >= args.max_samples:
            break
    
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
                        help="最大样本数量")
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