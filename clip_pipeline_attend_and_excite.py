"""
CLIP增强的Attend-and-Excite Pipeline - 中文注释版

该文件实现了基于Stable Diffusion的文本到图像生成Pipeline,
集成了Attend-and-Excite技术,用于增强文本描述中特定词汇的视觉表现力。

主要功能:
1. 文本到图像的生成 - 使用Stable Diffusion模型
2. Attend-and-Excite优化 - 增强特定单词在生成图像中的表现
3. CLIP Loss监督 - 使用CLIP模型确保图像与文本的一致性
4. 关系注意力机制 - 支持词与词之间的关系建模

Author: 
"""

# ==================== 导入必要的库 ====================
import inspect
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

import numpy as np
import torch
from torch.nn import functional as F

from packaging import version
from transformers import CLIPTextModel, CLIPTokenizer
try:
    from transformers import CLIPFeatureExtractor
except ImportError:
    from transformers import CLIPImageProcessor as CLIPFeatureExtractor

from diffusers.configuration_utils import FrozenDict
from diffusers.models import AutoencoderKL, UNet2DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
# from diffusers.utils import deprecate, is_accelerate_available, logging, randn_tensor, replace_example_docstring
from diffusers.utils import deprecate, is_accelerate_available, logging, replace_example_docstring

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipeline

# 自定义工具模块
from utils.gaussian_smoothing import GaussianSmoothing  # 高斯平滑处理注意力图
from utils.ptp_utils import AttentionStore, aggregate_attention  # 注意力存储和聚合
from clip_loss import CLIPLoss  # CLIP损失函数
import open_clip as clip_lib  # OpenCLIP模型库
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize  # 图像预处理变换


def _load_clip_model(model_name="ViT-L/14", device="cuda"):
    """
    加载CLIP模型,支持本地权重和预训练权重两种方式
    
    参数:
        model_name (str): CLIP模型名称,默认为"ViT-L/14"(Vision Transformer Large, patch size 14)
        device (str): 设备类型,默认为"cuda",支持"cpu"等
    
    返回:
        tuple: (model, preprocess) - CLIP模型和对应的图像预处理函数
    
    加载策略:
        1. 首先尝试从本地路径加载预下载的safetensors格式权重
        2. 如果本地加载失败,回退到从网络下载openai预训练权重
    """
    local_clip_path = "/home/cn/yolo/AnomalyAny/ViT-L-14-openai.safetensors"
    import os

    if os.path.exists(local_clip_path):
        try:
            print(f"Loading CLIP model from: {local_clip_path}")
            from safetensors.torch import load_file
            state_dict = load_file(local_clip_path)
            print(f"Loaded {len(state_dict)} tensors from safetensors")

            model, _, preprocess = clip_lib.create_model_and_transforms('ViT-L-14', pretrained=None)

            # Convert state dict to float16 if device is cuda
            if 'cuda' in str(device):
                state_dict = {k: v.half() if v.dtype == torch.float32 else v
                             for k, v in state_dict.items()}

            result = model.load_state_dict(state_dict, strict=False)
            # Handle _IncompatibleKeys object
            if hasattr(result, 'missing_keys'):
                missing = len(result.missing_keys)
            else:
                missing = 'unknown'
            print(f"CLIP model loaded: {len(state_dict)} keys, missing: {missing}")
            return model.to(device), preprocess
        except Exception as e:
            print(f"Warning: Could not load local CLIP: {e}")

    # Fallback to pretrained
    print("Using pretrained CLIP model from openai...")
    model, _, preprocess = clip_lib.create_model_and_transforms('ViT-L-14', pretrained='openai')
    return model.to(device), preprocess

# TODO
from PIL import Image
from torch import autocast
import gc

logger = logging.get_logger(__name__)


def image2latent(vae, image, width, height, device, generator):
    """
    将输入图像编码为Stable Diffusion的潜空间表示
    
    参数:
        vae: Variational Autoencoder模型,用于将图像编码到潜空间
        image: 输入的PIL图像或numpy数组格式的图像
        width: 目标宽度,用于调整图像大小
        height: 目标高度,用于调整图像大小
        device: torch设备(如'cuda'或'cpu')
        generator: 随机数生成器,用于控制采样过程的确定性
    
    返回:
        torch.Tensor: 编码后的潜空间表示,形状为[1, 4, height//8, width//8]
    
    处理流程:
        1. 将图像调整为目标尺寸
        2. 将像素值从[0,255]归一化到[-1,1]
        3. 处理Alpha通道(如果存在)
        4. 使用VAE编码器编码到潜空间
        5. 乘以缩放因子0.18215(SD模型的固定缩放因子)
    """
    init_image = image
    # Resize and transpose for numpy b h w c -> torch b c h w
    init_image = init_image.resize((width, height), resample=Image.Resampling.LANCZOS)
    init_image = np.array(init_image).astype(np.float32) / 255.0 * 2.0 - 1.0
    init_image = torch.from_numpy(init_image[np.newaxis, ...].transpose(0, 3, 1, 2))

    # If there is alpha channel, composite alpha for white, as the diffusion model does not support alpha channel
    if init_image.shape[1] > 3:
        init_image = init_image[:, :3] * init_image[:, 3:] + (1 - init_image[:, 3:])

    # Move image to GPU
    init_image = init_image.to(device)

    # Encode image
    with autocast(device):
        init_latent = vae.encode(init_image).latent_dist.sample(generator=generator) * 0.18215

    return init_latent


def read_mask(mask_path: str, dest_size=(64, 64)):
    """
    读取并预处理掩码图像,将其转换为二值化的torch张量
    
    参数:
        mask_path: 掩码图像的路径(str类型)或PIL图像对象
        dest_size: 目标尺寸,默认为(64,64),与Stable Diffusion的潜空间尺寸匹配
    
    返回:
        tuple: (mask, org_mask)
            - mask: 二值化的torch张量,形状为[1, 1, 64, 64],值为0或1
            - org_mask: 原始的PIL灰度图像
    
    处理流程:
        1. 加载掩码图像并转换为灰度模式
        2. 调整大小到目标尺寸
        3. 二值化处理: 非零像素设为255,然后归一化为0/1
        4. 转换为torch的half类型张量并移到GPU
    """
    if isinstance(mask_path, str):
        org_mask = Image.open(mask_path).convert("L")
    else: 
        org_mask = mask_path.convert("L")
    mask = org_mask.resize(dest_size, Image.NEAREST)
    mask = np.array(mask)
    # print(mask)
    mask[mask != 0] = 255
    mask = np.array(mask) / 255
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = mask[np.newaxis, np.newaxis, ...]
    mask = torch.from_numpy(mask).half().to('cuda')

    return mask, org_mask


class RelationalAttendAndExcitePipeline(StableDiffusionPipeline):
    """
    关系型Attend-and-Excite扩散Pipeline
    
    该Pipeline继承自StableDiffusionPipeline,专门用于文本到图像的生成任务,
    并集成了Attend-and-Excite算法来增强特定文本标记在生成图像中的表现力。
    
    核心特性:
        1. Attend-and-Excite优化: 通过迭代优化过程,确保文本描述中的关键词
           能够被准确地体现在生成的图像中
        2. 关系注意力机制: 支持建模文本中不同词之间的关系
        3. CLIP Loss监督: 使用CLIP模型提供的对比学习损失来确保生成图像
           与文本描述的一致性
        4. 灵活的掩码支持: 可以指定关注特定区域的图像生成
    
    继承关系:
        StableDiffusionPipeline -> DiffusionPipeline -> BaseOutput
    
    参数说明:
        vae (AutoencoderKL): 
            变分自编码器模型,用于将图像编码到潜空间以及从潜空间解码图像
        
        text_encoder (CLIPTextModel): 
            冻结的文本编码器,使用CLIP的文本编码部分
            Stable Diffusion使用 clip-vit-large-patch14 变体
        
        tokenizer (CLIPTokenizer): 
            CLIP分词器,用于将文本转换为模型可处理的token序列
        
        unet (UNet2DConditionModel): 
            条件U-Net网络,负责在潜空间中执行去噪过程
            接收文本嵌入作为条件信息
        
        scheduler (SchedulerMixin): 
            扩散调度器,控制噪声添加和去除的策略
            支持DDIMScheduler、LMSDiscreteScheduler、PNDMScheduler等
        
        safety_checker (StableDiffusionSafetyChecker): 
            安全检查器,用于过滤可能包含不当内容的生成图像
        
        feature_extractor (CLIPFeatureExtractor): 
            特征提取器,从生成的图像中提取特征用于安全检查
    
    使用示例:
        >>> pipe = RelationalAttendAndExcitePipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
        >>> image = pipe("a cute cat sitting on a chair", indices_to_alter=[2, 4]).images[0]
    
    注意事项:
        - 默认启用分类器无指导(Classifier-Free Guidance),guidance_scale建议在7-8之间
        - 通过indices_to_alter参数指定需要增强的词索引
        - 支持多种输出格式:PIL图像、numpy数组或torch张量
    """
    _optional_components = ["safety_checker", "feature_extractor"]

    def decode_latents_new(self, latents):
        """
        将潜空间表示解码为RGB图像
        
        参数:
            latents (torch.Tensor): 潜空间张量,形状为[batch_size, channels, height, width]
        
        返回:
            torch.Tensor: 解码后的RGB图像,形状为[batch_size, height, width, channels],
                        值域在[0,1]范围内
        
        处理步骤:
            1. 反缩放: 使用VAE的缩放因子(0.18215)的倒数进行反缩放
            2. VAE解码: 使用VAE解码器将潜变量转换为图像
            3. 图像后处理: 将图像值从[-1,1]范围映射到[0,1]范围
            4. 维度转换: 从CHW格式转换为HWC格式(用于PIL/OpenCV兼容性)
        
        注意:
            - 已在方法开始时调用zero_grad()清空梯度,避免内存泄漏
            - 图像数据保持在GPU上,不转换为numpy格式
        """
        # deprecation_message = "The decode_latents method is deprecated and will be removed in 1.0.0. Please use VaeImageProcessor.postprocess(...) instead"
        # deprecate("decode_latents", "1.0.0", deprecation_message, standard_warn=False)

        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents, return_dict=False)[0]
        self.vae.zero_grad()
        image = torch.clamp(image / 2 + 0.5, min=0, max=1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        # image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        image = image.permute(0, 2, 3, 1)
        return image

    def latent_process(self, img_latent):
        """
        处理潜空间图像以适配CLIP模型的输入要求
        
        参数:
            img_latent (torch.Tensor): 潜空间图像,通常来自SD的中间层或VAE输出
                                       形状为[batch_size, height, width, channels] (HWC格式)
        
        返回:
            torch.Tensor: 处理后的图像张量,形状为[batch_size, 3, 224, 224],
                        已进行CLIP标准的ImageNet归一化
        
        处理流程:
            1. 维度转换: 从HWC格式转换为CHW格式 [B,H,W,C] -> [B,C,H,W]
            2. 尺寸调整: 使用双三次插值将图像resize到224x224
                       (CLIP-ViT-L/14的标准输入尺寸)
            3. 归一化: 使用ImageNet统计量进行标准化
                      - 均值: [0.48145466, 0.4578275, 0.40821073] (RGB通道)
                      - 标准差: [0.26862954, 0.26130258, 0.27577711]
        
        用途:
            该方法主要用于将Stable Diffusion的潜空间图像转换为CLIP图像嵌入,
            以便计算CLIP损失或进行图像-文本相似度评估
        """
        img_latent = img_latent.permute(0, 3, 1, 2)  # HWC -> CHW
        img_latent_resized = torch.nn.functional.interpolate(img_latent, size=(224, 224), mode='bicubic',
                                                             align_corners=False)  # Resize到CLIP输入尺寸
        transform = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))  # ImageNet归一化
        img_latent_resized = transform(img_latent_resized)
        return img_latent_resized

    def _encode_prompt(
            self,
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt=None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            img_prompt=None,
            indice=None,
            normal_prompt=None
    ):
        """
        将文本提示词编码为文本编码器的隐藏状态(文本嵌入)
        
        参数:
            prompt (str或List[str], 可选): 
                要编码的文本提示词,可以是单个字符串或字符串列表
            
            device (torch.device): 
                torch设备类型,如'cuda'或'cpu'
            
            num_images_per_prompt (int): 
                每个提示词要生成的图像数量
            
            do_classifier_free_guidance (bool): 
                是否使用分类器无指导(CFG)技术
                - True: 启用CFG,同时生成正向和负向条件的嵌入
                - False: 只使用正向条件
            
            negative_prompt (str或List[str], 可选): 
                负向提示词,用于CFG中"不想要"的内容指导
                如果为None且启用CFG,则使用空字符串""
            
            prompt_embeds (torch.FloatTensor, 可选): 
                预先生成的文本嵌入向量
                如果提供,将直接使用而不进行编码
                适用于需要微调输入权重的场景
            
            negative_prompt_embeds (torch.FloatTensor, 可选): 
                预先生成的负向文本嵌入向量
                如果启用CFG但未提供,将从negative_prompt生成
            
            img_prompt (可选): 
                图像嵌入向量,可用于替换或增强某些token的嵌入
                与indice参数配合使用
            
            indice (可选): 
                需要替换或增强的token索引列表
                配合img_prompt使用
            
            normal_prompt (可选): 
                规范化后的提示词嵌入,用于计算归一化偏移量
        
        返回:
            tuple: (text_inputs, prompt_embeds)
                - text_inputs: 分词器输出的字典,包含input_ids等
                - prompt_embeds: 编码后的文本嵌入向量
                                 当启用CFG时: [batch_size*2, seq_len, hidden_dim]
                                 否则: [batch_size, seq_len, hidden_dim]
        
        处理流程:
            1. 处理输入: 确定batch_size和提示词文本
            2. 文本编码: 如果未提供prompt_embeds,则使用tokenizer和text_encoder编码
            3. CFG处理: 如果启用CFG,生成负向嵌入并拼接
            4. 重复嵌入: 根据num_images_per_prompt复制嵌入向量
        
        重要细节:
            - CLIP模型最大处理77个token,超出部分会被截断
            - 返回的prompt_embeds会在末尾拼接负向嵌入(当启用CFG时)
            - 拼接顺序为:[negative_prompt_embeds, prompt_embeds],这是CFG计算的标准顺序
        """
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt_text = prompt
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
            prompt_text = prompt[0] if prompt else ""
        else:
            batch_size = prompt_embeds.shape[0]
            prompt_text = ""

        if prompt_embeds is None:
            text_inputs = self.tokenizer(
                prompt_text,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt_text, padding="longest", return_tensors="pt").input_ids

            if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(
                    text_input_ids, untruncated_ids
            ):
                removed_text = self.tokenizer.batch_decode(
                    untruncated_ids[:, self.tokenizer.model_max_length - 1: -1]
                )
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = text_inputs.attention_mask.to(device)
            else:
                attention_mask = None

            prompt_embeds = self.text_encoder(
                text_input_ids.to(device),
                attention_mask=attention_mask,
            )
            prompt_embeds = prompt_embeds[0]
            # print(prompt_embeds.shape) # torch.Size([1, 77, 768])
            # TODO switch prompt with image embedding
            # print(prompt[:, indice, :].shape)
            # if img_prompt is not None:
            #    prompt_embeds[:, indice, :] = prompt_embeds_normal[:, indice, :] + (prompt_embeds[:, indice, :] - prompt_embeds_normal[:, indice, :]).norm(dim=-1, keepdim=True)*img_prompt/img_prompt.norm(dim=-1, keepdim=True)
            # prompt_embeds[:, indice, :] = prompt_embeds[:, indice, :] - prompt_embeds_normal[:, indice, :] + img_prompt
            # prompt_embeds[:, indice, :] = prompt_embeds_normal[:, indice, :] + img_prompt
            # prompt_embeds[:, indice, :] = img_prompt

        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        # print("before", prompt_embeds.size()) # [1, 77, 768]

        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)
        # print("after", prompt_embeds.size())  # [1, 77, 768]

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = prompt_embeds.shape[1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            negative_prompt_embeds = self.text_encoder(
                uncond_input.input_ids.to(device),
                attention_mask=attention_mask,
            )
            negative_prompt_embeds = negative_prompt_embeds[0]

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]

            negative_prompt_embeds = negative_prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)

            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # print("final", prompt_embeds.size())  # [1, 77, 768]

        return text_inputs, prompt_embeds

    def _compute_max_attention_per_index(self,
                                         attention_maps: torch.Tensor,
                                         indices_to_alter: List[int],
                                         smooth_attentions: bool = False,
                                         sigma: float = 0.5,
                                         kernel_size: int = 3,
                                         normalize_eot: bool = False,
                                         return_attention: bool = False) -> List[torch.Tensor]:
        """
        计算每个待修改token的最大注意力值
        
        Attend-and-Excite算法的核心: 评估文本中特定单词在图像生成过程中
        获得的关注程度,并将其作为优化目标的依据。
        
        参数:
            attention_maps (torch.Tensor): 
                聚合后的注意力图,形状为[batch_size, num_heads, seq_len, spatial_dim]
                - seq_len: 文本序列长度(通常为77)
                - spatial_dim: 空间维度,取决于attention_res(通常为16x16=256或更大)
            
            indices_to_alter (List[int]): 
                需要关注/增强的token索引列表
                这些索引对应于prompt中需要重点体现在图像中的词
            
            smooth_attentions (bool): 
                是否对注意力图进行高斯平滑处理
                平滑可以减少噪声,使注意力分布更加连续
            
            sigma (float): 
                高斯平滑的标准差,控制平滑程度
                值越大,平滑效果越明显
            
            kernel_size (int): 
                高斯滤波核的大小,必须是正奇数
                常用值为3或5
            
            normalize_eot (bool): 
                是否规范化到End-of-Text token
                如果为True,将排除EOT token之后的token
            
            return_attention (bool): 
                是否返回处理后的注意力图
                用于调试或可视化
        
        返回:
            List[torch.Tensor] 或 Tuple:
                - 如果return_attention=False: 
                  返回每个目标token的最大注意力值列表
                - 如果return_attention=True: 
                  返回(最大注意力值列表, 最后一个注意力图)
        
        算法原理:
            1. 提取文本token的注意力: 从注意力图中提取对应token的注意力权重
            2. Softmax归一化: 对注意力值应用softmax,使其和为1
            3. 缩放: 乘以100以增大差异(Attention的缩放因子)
            4. 取最大值: 对空间维度取最大值,得到每个token的"最大激活强度"
            
        优化逻辑:
            - 理想情况下,关键词应该获得较高的注意力值
            - 如果某个关键词的注意力值过低,说明该词的特征没有很好地体现在图像中
            - 通过损失函数 L = max(0, 1 - max_attention) 来驱动优化
            - 当max_attention < 1时,损失为正,梯度会反向传播以增强该词的表现
        """
        last_idx = -1
        if normalize_eot:
            prompt = self.prompt
            if isinstance(self.prompt, list):
                prompt = self.prompt[0]
            last_idx = len(self.tokenizer(prompt)['input_ids']) - 1
        attention_for_text = attention_maps[:, :, 1:last_idx]
        attention_for_text *= 100
        attention_for_text = torch.nn.functional.softmax(attention_for_text, dim=-1)

        # Shift indices since we removed the first token
        indices_to_alter = [index - 1 for index in indices_to_alter]

        # Extract the maximum values
        max_indices_list = []
        for i in indices_to_alter:
            image = attention_for_text[:, :, i]
            if smooth_attentions:
                smoothing = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=sigma, dim=2).cuda()
                input = F.pad(image.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1), mode='reflect')
                image = smoothing(input).squeeze(0).squeeze(0)
            max_indices_list.append(image.max())
        if return_attention:
            return max_indices_list, image
        return max_indices_list

    def _aggregate_and_get_max_attention_per_token(self, attention_store: AttentionStore,
                                                   indices_to_alter: List[int],
                                                   attention_res: int = 16,
                                                   smooth_attentions: bool = False,
                                                   sigma: float = 0.5,
                                                   kernel_size: int = 3,
                                                   normalize_eot: bool = False,
                                                   return_maps: bool = False):
        """
        聚合注意力并计算每个目标token的最大激活值
        
        该方法是Attend-and-Excite pipeline的核心方法之一,
        负责从AttentionStore中提取并聚合注意力信息,
        然后计算每个需要增强的token的最大注意力值。
        
        参数:
            attention_store (AttentionStore): 
                注意力存储对象,包含了去噪过程中所有层的注意力权重
                由UNet的hook机制自动收集
            
            indices_to_alter (List[int]): 
                需要增强的token索引列表
                这些索引对应prompt中需要重点体现在图像中的关键词
            
            attention_res (int): 
                注意力图的分辨率,默认为16
                表示将潜空间划分为16x16的网格
                值越大,空间分辨率越高
            
            smooth_attentions (bool): 
                是否对注意力图进行高斯平滑
                有助于减少噪声,获得更连续的注意力分布
            
            sigma (float): 
                高斯平滑的标准差
            
            kernel_size (int): 
                高斯滤波核的大小
            
            normalize_eot (bool): 
                是否规范化到EOT token位置
                如果为True,将排除EOT token之后的无效token
            
            return_maps (bool): 
                是否返回聚合后的注意力图
                用于调试或可视化目的
        
        返回:
            List[torch.Tensor] 或 Tuple:
                - 如果return_maps=False: 
                  返回每个目标token的最大注意力值
                - 如果return_maps=True: 
                  返回(最大注意力值列表, 处理后的注意力图)
        
        处理流程:
            1. 聚合注意力: 从attention_store中提取cross-attention
                          在指定的层(up, down, mid)中聚合
            2. 计算最大注意力: 调用_compute_max_attention_per_index
                             计算每个目标token的最大注意力激活值
            3. 可选返回: 根据return_maps决定是否返回注意力图
        
        重要性:
            这个方法将原始的注意力数据转换为可供优化器使用的标量值,
            是连接去噪过程和优化目标的桥梁。
        """
        attention_maps = aggregate_attention(
            attention_store=attention_store,
            res=attention_res,
            from_where=("up", "down", "mid"),
            is_cross=True,
            select=0)
        max_attention_per_index, attention_per_index = self._compute_max_attention_per_index(
            attention_maps=attention_maps,
            indices_to_alter=indices_to_alter,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot,
            return_attention=True)
        if return_maps:
            return max_attention_per_index, attention_per_index
        return max_attention_per_index

    @staticmethod
    def _compute_loss(max_attention_per_index: List[torch.Tensor], return_losses: bool = False,
                      compute_clip=True) -> torch.Tensor:
        """
        计算Attend-and-Excite损失函数
        
        Attend-and-Excite的核心思想:
        如果某个关键词在图像中获得的注意力值过低,
        则损失为正值,会驱动梯度下降以增强该词在图像中的表现。
        
        参数:
            max_attention_per_index (List[torch.Tensor]): 
                每个待修改token的最大注意力值列表
                理想情况下,每个值应该接近或大于1
            
            return_losses (bool): 
                是否返回所有token的单独损失
                - True: 返回总损失和所有单独损失
                - False: 只返回总损失
        
        返回:
            torch.Tensor 或 Tuple:
                - 如果return_losses=False: 返回总损失(单个标量)
                - 如果return_losses=True: 返回(总损失, 单独损失列表)
        
        损失函数设计:
            对于每个token i:
                loss_i = max(0, 1 - max_attention_i)
            
            总损失为所有token损失的最大值:
                loss = max(loss_1, loss_2, ..., loss_n)
        
        为什么使用max而不是sum?
            - 使用max确保只要所有关键词都得到足够关注,损失就为0
            - 鼓励所有目标词同时被良好表现,而不是只有一个词被过度强调
            - 这种设计更加平衡和公平
        
        优化目标:
            - 当所有max_attention >= 1时, loss = 0 (最优状态)
            - 当任何max_attention < 1时, loss > 0 (需要优化)
            - 梯度会反向传播到latents,驱动其向增强关键词表现的方向更新
        """
        losses = [max(0, 1. - curr_max) for curr_max in max_attention_per_index]
        loss = max(losses)
        if not isinstance(loss, torch.Tensor):
            loss = torch.tensor(float(loss), requires_grad=False)
        if return_losses:
            return loss, losses
        else:
            del losses
            return loss

    @staticmethod
    def _update_latent(latents: torch.Tensor, loss: torch.Tensor, step_size: float, return_grad=False) -> torch.Tensor:
        """
        根据计算得到的损失更新潜空间变量
        
        这是Attend-and-Excite优化循环中的关键步骤,
        通过梯度下降来更新latents,使得关键词在图像中的表现得到增强。
        
        参数:
            latents (torch.Tensor): 
                当前的潜空间变量
                形状为[batch_size, channels, height, width]
                这是Stable Diffusion去噪过程的中间结果
            
            loss (torch.Tensor): 
                计算得到的Attend-and-Excite损失
                标量张量,表示需要优化的程度
            
            step_size (float): 
                梯度下降的学习率/步长
                控制每次更新的幅度
                - 值过大: 可能导致优化不稳定
                - 值过小: 收敛速度慢,需要更多迭代步骤
            
            return_grad (bool): 
                是否返回计算得到的梯度
                - True: 返回更新后的latents和梯度
                - False: 只返回更新后的latents
        
        返回:
            torch.Tensor 或 Tuple:
                - 如果return_grad=False: 返回更新后的latents
                - 如果return_grad=True: 返回(更新后的latents, 梯度)
        
        更新公式:
            latents_new = latents - step_size * gradient
            
            减去梯度是因为我们希望沿着损失函数下降的方向移动
        
        优化策略:
            1. 使用torch.autograd.grad计算损失对latents的梯度
            2. 设置retain_graph=True以支持多次反向传播
            3. 执行梯度下降: latents = latents - step_size * grad
            4. 清理中间变量以释放GPU内存
        
        注意事项:
            - 梯度计算后立即清理grad_cond以释放内存
            - 损失变量在更新后也会被删除
            - 这种手动梯度管理对于大模型的内存优化很重要
        """
        grad_cond = torch.autograd.grad(loss.requires_grad_(True), [latents], retain_graph=True)[0]
        # print()
        latents = latents - step_size * grad_cond
        if return_grad:
            return latents, grad_cond
        del grad_cond
        del loss
        # gc.collect()
        # torch.cuda.empty_cache()
        return latents

    def _perform_iterative_refinement_step(self,
                                           latents: torch.Tensor,
                                           indices_to_alter: List[int],
                                           loss: torch.Tensor,
                                           threshold: float,
                                           text_embeddings: torch.Tensor,
                                           text_input,
                                           attention_store: AttentionStore,
                                           step_size: float,
                                           t: int,
                                           attention_res: int = 16,
                                           smooth_attentions: bool = True,
                                           sigma: float = 0.5,
                                           kernel_size: int = 3,
                                           max_refinement_steps: int = 20,
                                           normalize_eot: bool = False):
        """
        执行论文中提出的迭代式潜空间精炼过程
        
        这是Attend-and-Excite算法的核心优化循环,
        在每个去噪时间步中,持续根据损失目标更新潜空间变量,
        直到所有目标token的注意力值都达到阈值要求。
        
        参数:
            latents (torch.Tensor): 
                当前的潜空间变量,形状为[batch_size, channels, height, width]
            
            indices_to_alter (List[int]): 
                需要增强的token索引列表
            
            loss (torch.Tensor): 
                初始损失值,用于判断是否需要进一步优化
            
            threshold (float): 
                注意力阈值,默认为0.1
                当所有目标token的注意力值都>= (1-threshold)时,认为优化完成
            
            text_embeddings (torch.Tensor): 
                文本嵌入向量,包含正向和负向嵌入
                [negative_embeds, positive_embeds] 的拼接形式
            
            text_input: 
                分词器的输出,包含input_ids等信息
                用于解码token索引对应的实际单词
            
            attention_store (AttentionStore): 
                注意力存储对象,收集UNet各层的注意力权重
            
            step_size (float): 
                梯度下降的步长,控制每次更新的幅度
            
            t (int): 
                当前去噪时间步的timestep
                用于U-Net的条件生成
            
            attention_res (int): 
                注意力图的分辨率,默认为16
            
            smooth_attentions (bool): 
                是否对注意力图进行高斯平滑
            
            sigma (float): 
                高斯平滑的标准差
            
            kernel_size (int): 
                高斯滤波核的大小
            
            max_refinement_steps (int): 
                最大迭代次数,防止无限循环
                默认为20次
            
            normalize_eot (bool): 
                是否规范化到EOT token
        
        返回:
            Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
                - loss: 最终的Attend-and-Excite损失值
                - latents: 更新后的潜空间变量
                - max_attention_per_index: 每个目标token的最终最大注意力值
        
        优化流程:
            1. 初始化迭代计数器
            2. 进入优化循环(最多max_refinement_steps次):
                a. 克隆latents并启用梯度
                b. 执行UNet前向传播,获取噪声预测
                c. 聚合注意力并计算最大注意力值
                d. 计算Attend-and-Excite损失
                e. 如果损失>0,执行梯度下降更新latents
                f. 找出注意力值最低的token(瓶颈token)
                g. 打印调试信息(可选)
                h. 检查是否达到收敛条件
            3. 最后一次前向传播(不更新梯度),收集最终状态
        
        关键设计:
            - 使用克隆.detach()避免修改原始latents
            - 启用梯度以支持反向传播
            - 每轮迭代都重新计算注意力,确保数据最新
            - 使用max_refinement_steps防止无限循环
        """
        iteration = 0
        target_loss = max(0, 1. - threshold)
        # while loss > target_loss:
        while iteration < max_refinement_steps:
            iteration += 1

            latents = latents.clone().detach().requires_grad_(True)
            noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
            self.unet.zero_grad()

            # Get max activation value for each subject token
            max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
                attention_store=attention_store,
                indices_to_alter=indices_to_alter,
                attention_res=attention_res,
                smooth_attentions=smooth_attentions,
                sigma=sigma,
                kernel_size=kernel_size,
                normalize_eot=normalize_eot
            )

            loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)

            if loss != 0:
                latents = self._update_latent(latents, loss, step_size)

            with torch.no_grad():
                noise_pred_uncond = self.unet(latents, t, encoder_hidden_states=text_embeddings[0].unsqueeze(0)).sample
                noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample

            try:
                low_token = np.argmax([l.item() if type(l) != int else l for l in losses])
            except Exception as e:
                print(e)  # catch edge case :)
                low_token = np.argmax(losses)

            low_word = self.tokenizer.decode(text_input.input_ids[0][indices_to_alter[low_token]])
            # print(f'\t Try {iteration}. {low_word} has a max attention of {max_attention_per_index[low_token]}')

            if iteration >= max_refinement_steps:
                # print(f'\t Exceeded max number of iterations ({max_refinement_steps})! '
                #   f'Finished with a max attention of {max_attention_per_index[low_token]}')
                break

        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
        # print(f"\t Finished with loss of: {loss}")
        return loss, latents, max_attention_per_index
    
    def _perform_att(self,
                    latents: torch.Tensor,
                    indices_to_alter: List[int],
                    text_embeddings: torch.Tensor,
                    text_input,
                    attention_store: AttentionStore,
                    step_size: float,
                    t: int,
                    attention_res: int = 16,
                    smooth_attentions: bool = True,
                    sigma: float = 0.5,
                    kernel_size: int = 3,
                    max_refinement_steps: int = 20,
                    normalize_eot: bool = False):
        """
        执行单步Attend-and-Excite注意力计算(不进行迭代优化)
        
        与_perform_iterative_refinement_step不同,这个方法只执行一次前向传播,
        计算当前latents下的注意力损失,而不会迭代更新latents。
        
        参数:
            latents (torch.Tensor): 
                当前的潜空间变量,形状为[batch_size, channels, height, width]
            
            indices_to_alter (List[int]): 
                需要关注/增强的token索引列表
            
            text_embeddings (torch.Tensor): 
                文本嵌入向量,包含正向和负向嵌入的拼接
            
            text_input: 
                分词器输出,包含input_ids
            
            attention_store (AttentionStore): 
                注意力存储对象,用于收集注意力权重
            
            step_size (float): 
                梯度下降步长(在此方法中可能未使用)
            
            t (int): 
                当前去噪时间步
            
            attention_res (int): 
                注意力图的分辨率
            
            smooth_attentions (bool): 
                是否对注意力图进行高斯平滑
            
            sigma (float): 
                高斯平滑标准差
            
            kernel_size (int): 
                高斯滤波核大小
            
            max_refinement_steps (int): 
                最大优化步数(在此方法中未使用)
            
            normalize_eot (bool): 
                是否规范化到EOT token
        
        返回:
            Tuple[torch.Tensor, List[torch.Tensor]]:
                - loss: Attend-and-Excite损失值
                - max_attention_per_index: 每个目标token的最大注意力值列表
        
        方法特点:
            - 单次前向传播: 不执行迭代优化,只计算当前状态的损失
            - 用于: 在去噪循环中快速评估当前latents的表现
            - 用途: 可能用于调试、监控或轻量级优化
        
        与_perform_iterative_refinement_step的区别:
            - _perform_iterative_refinement_step: 执行迭代优化,会更新latents
            - _perform_att: 只计算损失,不更新latents
        """
        # latents = latents.clone().detach().requires_grad_(True)
        # Run one more time but don't compute gradients and update the latents.
        # We just need to compute the new loss - the grad update will occur below
        # latents = latents.clone().detach().requires_grad_(True)
        noise_pred_text = self.unet(latents, t, encoder_hidden_states=text_embeddings[1].unsqueeze(0)).sample
        self.unet.zero_grad()

        # Get max activation value for each subject token
        max_attention_per_index = self._aggregate_and_get_max_attention_per_token(
            attention_store=attention_store,
            indices_to_alter=indices_to_alter,
            attention_res=attention_res,
            smooth_attentions=smooth_attentions,
            sigma=sigma,
            kernel_size=kernel_size,
            normalize_eot=normalize_eot)
        loss, losses = self._compute_loss(max_attention_per_index, return_losses=True)
        # print(f"\t Finished with loss of: {loss}")
        return loss, max_attention_per_index
    
    def _prompt_update(self, prompt_embeds, prompt_embeds_original, normal_embeds, indices_to_alter, curr_step_size=0.1, num_iters=5):
        criterion_cosine = torch.nn.CosineSimilarity()
        prompt_anomaly = prompt_embeds[:, indices_to_alter, :]
        for k in range(num_iters):
            with torch.enable_grad():
                prompt_anomaly = prompt_anomaly.detach().requires_grad_(True)
                delta_1 = self._compute_dist(normal_embeds[:, indices_to_alter, :], prompt_anomaly)
                delta_2 = self._compute_dist(normal_embeds[:, indices_to_alter, :], prompt_embeds_original[:, indices_to_alter, :])
                loss_clip = criterion_cosine(delta_1, delta_2).mean()
                loss_prompt = 1.0 * (1.0 - loss_clip)
                if torch.isnan(loss_prompt) or torch.isinf(loss_prompt):
                    break
                prompt_anomaly = self._update_latent(latents=prompt_anomaly, loss=loss_prompt, step_size=curr_step_size)
                if torch.isnan(prompt_anomaly).any() or torch.isinf(prompt_anomaly).any():
                    break
        prompt_embeds[:, indices_to_alter, :] = prompt_anomaly
        del loss_prompt
        return prompt_embeds
    
    def _compute_dist(self, t1, t2):
        delta = t2.mean(dim=0, keepdim=True) - t1.mean(dim=0, keepdim=True)
        # delta = delta/delta.norm(dim=-1, keepdim=True)
        delta_new = torch.nan_to_num(delta)

        return delta_new


    @torch.no_grad()
    def __call__(
            self,
            prompt: Union[str, List[str]],
            attention_store: AttentionStore,
            indices_to_alter: List[int],
            attention_res: int = 16,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            init_image=None,
            init_image_guidance_scale: float = 0.5,
            mask_image=None,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: Optional[int] = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            max_iter_to_alter: Optional[int] = 25,
            run_standard_sd: bool = False,
            thresholds: Optional[dict] = {0: 0.05, 10: 0.5, 20: 0.8},
            scale_factor: int = 20,
            scale_range: Tuple[float, float] = (1., 0.5),
            smooth_attentions: bool = True,
            sigma: float = 0.5,
            kernel_size: int = 3,
            sd_2_1: bool = False,
            img_prompt: torch.Tensor = None,
            normal_prompt=None,
            abnormal_img=None,
            original_prompt=None,
            detailed_prompt=None,
            clip_loss=None,
            inner_loop_iters: int = 12,
            prompt_update_iters: int = 20,
            clip_step_size_init: float = 0.002
    ):
        """
        调用Pipeline进行Attend-and-Excite增强的图像生成
        
        这是Pipeline的主要入口函数,负责协调整个文本到图像的生成过程,
        包括文本编码、扩散去噪、Attend-and-Excite优化和图像解码。
        
        参数详解:
            prompt (str或List[str]): 
                引导图像生成的文本提示词
                支持单个字符串或字符串列表(批量生成)
            
            attention_store (AttentionStore): 
                注意力存储对象,用于收集去噪过程中的注意力权重
                需要在调用前创建并传入
            
            indices_to_alter (List[int]): 
                需要增强的token索引列表
                这些索引对应的词将在图像中获得更多关注
            
            attention_res (int, 默认16): 
                注意力图的分辨率
                将潜空间划分为attention_res × attention_res的网格
            
            height/width (int, 可选): 
                生成图像的高度和宽度(像素)
                默认为512×512 (unet.config.sample_size × vae_scale_factor)
            
            num_inference_steps (int, 默认50): 
                去噪步数,越多步数通常生成质量越高但速度越慢
                DDIM调度器下,20-30步通常就能获得不错的结果
            
            guidance_scale (float, 默认7.5): 
                分类器无指导(CFG)的引导强度
                - 1: 不使用CFG
                - 7-8: 平衡质量和多样性
                - 12+: 更遵循文本但可能牺牲图像质量
            
            init_image (可选): 
                初始图像,用于图像到图像的转换任务
                如果提供,将从init_image开始生成而非随机噪声
            
            init_image_guidance_scale (float, 默认0.5): 
                初始图像的指导强度,控制保留原图特征的程度
            
            mask_image (可选): 
                掩码图像,用于局部重绘
                只在init_image提供时有效
            
            negative_prompt (str或List[str], 可选): 
                负向提示词,指定不想要的内容
                用于CFG中引导生成"避开"某些特征的图像
            
            num_images_per_prompt (int, 默认1): 
                每个提示词生成的图像数量
            
            eta (float, 默认0.0): 
                DDIM调度器的eta参数
                - 0: 确定性采样
                - >0: 增加随机性(仅对DDIM有效)
            
            generator (torch.Generator, 可选): 
                随机数生成器,用于控制生成的可重复性
            
            latents (torch.FloatTensor, 可选): 
                预生成的噪声latents
                如果提供,将使用这些latents而非随机采样
            
            prompt_embeds (torch.FloatTensor, 可选): 
                预编码的文本嵌入向量
                如果提供,将直接使用而不重新编码
            
            negative_prompt_embeds (torch.FloatTensor, 可选): 
                预编码的负向文本嵌入
            
            output_type (str, 默认"pil"): 
                输出格式
                - "pil": 返回PIL.Image对象列表
                - "numpy": 返回numpy数组列表
                - "tensor": 返回torch.Tensor列表
            
            return_dict (bool, 默认True): 
                是否返回结构化的输出对象
                - True: 返回StableDiffusionPipelineOutput
                - False: 返回tuple
            
            callback (Callable, 可选): 
                每隔callback_steps步调用的回调函数
                签名: callback(step, timestep, latents)
            
            callback_steps (int, 默认1): 
                回调函数的调用频率(每多少步调用一次)
            
            cross_attention_kwargs (dict, 可选): 
                传递给交叉注意力处理器的额外参数
            
            max_iter_to_alter (int, 默认25): 
                每个去噪步骤中Attend-and-Excite的最大迭代次数
            
            run_standard_sd (bool, 默认False): 
                是否运行标准Stable Diffusion(不使用Attend-and-Excite)
            
            thresholds (dict, 默认{0:0.05, 10:0.5, 20:0.8}): 
                动态阈值设置
                格式: {去噪步数: 阈值}
                用于在不同阶段设置不同的收敛阈值
            
            scale_factor (int, 默认20): 
                潜空间缩放因子,用于调整优化步长
            
            scale_range (tuple, 默认(1., 0.5)): 
                缩放范围,在去噪过程中线性衰减
                格式: (起始值, 结束值)
            
            smooth_attentions (bool, 默认True): 
                是否对注意力图应用高斯平滑
            
            sigma (float, 默认0.5): 
                高斯平滑的标准差
            
            kernel_size (int, 默认3): 
                高斯滤波核的大小
            
            sd_2_1 (bool, 默认False): 
                是否使用Stable Diffusion 2.1模型
                影响模型配置和某些处理逻辑
            
            img_prompt (torch.Tensor, 可选): 
                图像嵌入向量,可用于替换特定token的嵌入
            
            normal_prompt (str, 可选): 
                正常/标准提示词,用于对比学习
            
            abnormal_img (可选): 
                异常图像,用于某些特定的图像处理任务
            
            original_prompt (str, 可选): 
                原始提示词,可能用于对比或回退
            
            detailed_prompt (str, 可选): 
                详细提示词,用于更精细的控制
            
            clip_loss (CLIPLoss, 可选): 
                CLIP损失计算器实例
                如果未提供,将在方法内部创建
        
        返回:
            StableDiffusionPipelineOutput 或 tuple:
                - images: 生成的图像列表
                - nsfw_content_detected: NSFW内容检测标志列表
        
        完整处理流程:
            1. 初始化和检查
               - 设置默认尺寸
               - 检查输入参数
               - 确定batch_size和设备
            
            2. 文本编码
               - 将prompt编码为文本嵌入
               - 处理CFG的正向/负向嵌入
               - 可选地准备原始/详细提示词的嵌入
            
            3. 准备潜空间变量
               - 如果有init_image,使用VAE编码
               - 否则从随机噪声采样latents
               - 准备掩码(如果提供)
            
            4. 扩散去噪循环
               - 遍历所有timesteps
               - 在特定步骤应用Attend-and-Excite优化
               - 根据thresholds动态调整优化强度
            
            5. 图像解码和后处理
               - 使用VAE解码latents到像素空间
               - 可选的安全检查
               - 转换为目标输出格式(PIL/numpy/tensor)
        
        使用示例:
            >>> from clip_pipeline_attend_and_excite import RelationalAttendAndExcitePipeline
            >>> from utils.ptp_utils import AttentionStore
            >>> 
            >>> # 初始化Pipeline
            >>> pipe = RelationalAttendAndExcitePipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
            >>> attention_store = AttentionStore()
            >>> 
            >>> # 定义要增强的词(假设prompt是"a red cat")
            >>> # indices_to_alter = [2] 表示增强"cat"这个词
            >>> 
            >>> # 生成图像
            >>> output = pipe(
            ...     prompt="a red cat sitting on a chair",
            ...     attention_store=attention_store,
            ...     indices_to_alter=[3],  # 增强"cat"
            ...     num_inference_steps=50
            ... )
            >>> 
            >>> # 获取结果
            >>> image = output.images[0]
        
        注意事项:
            - indices_to_alter的索引从0开始,对应分词后的token序列
            - 使用AttentionStore时需要注意清理,避免内存泄漏
            - Attend-and-Excite会显著增加计算时间(每个优化步骤都需额外前向传播)
            - 建议在GPU上运行,CPU运行会非常慢
        """
        criterion_mse = torch.nn.MSELoss()
        criterion_cosine = torch.nn.CosineSimilarity()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, preprocess = _load_clip_model(device=device)
        model.train()
        clip_loss = CLIPLoss(device,
                             lambda_direction=1.0,
                             lambda_patch=0.0,
                             lambda_global=0.0,
                             lambda_manifold=0.0,
                             lambda_texture=0.0,
                             clip_model=model, clip_processor=preprocess)

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt, height, width, callback_steps, negative_prompt, prompt_embeds, negative_prompt_embeds
        )

        # 2. Define call parameters
        self.prompt = prompt
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # print("===========", indices_to_alter[0]) [8]

        # 3. Encode input prompt
        text_inputs, prompt_embeds = self._encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            img_prompt=img_prompt,
            indice=indices_to_alter[0],
            normal_prompt=normal_prompt

        )
        prompt_original = detailed_prompt
        original_inputs, prompt_embeds_original = self._encode_prompt(
            [prompt_original],
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
            img_prompt=img_prompt,
            indice=indices_to_alter[0],
            normal_prompt=None
        )

        normal_inputs, normal_embeds = self._encode_prompt(
            [normal_prompt],
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=None,
            negative_prompt_embeds=negative_prompt_embeds,
            img_prompt=img_prompt,
            indice=indices_to_alter[0],
            normal_prompt=None
        )
        #
        # print(normal_prompt, prompt, prompt_original)
        model.zero_grad()

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        if init_image is not None:
            latents = image2latent(self.vae, init_image, width, height, "cuda", generator)

        num_channels_latents = self.unet.in_channels
        latents_source = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        if mask_image is not None:
            latent_mask, org_mask = read_mask(mask_image)
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        # Test-time Normal Sample Conditioning: init_image_guidance_scale
        if init_image is None:
            t_start = 0
        else:
            t_start = num_inference_steps - int(num_inference_steps * init_image_guidance_scale)
        timesteps_initial = timesteps
        timesteps = timesteps[t_start:]
        # print(timesteps)

        loss_img = 1000000
        k_range = 1
        clip_step_size = clip_step_size_init

        prompt_embeds = prompt_embeds.detach().requires_grad_(True)
        prompt_embeds = self._prompt_update(prompt_embeds=prompt_embeds,
                                            prompt_embeds_original=prompt_embeds_original,
                                            indices_to_alter=indices_to_alter,
                                            normal_embeds=normal_embeds,
                                            num_iters=prompt_update_iters)


        for k in range(k_range):
            #Generate random normal noise
            noise = torch.randn(latents.shape, generator=generator, device='cuda', dtype=latents.dtype)
            # latent = noise * scheduler.init_noise_sigma
            latents = self.scheduler.add_noise(latents, noise,
                                               torch.tensor([self.scheduler.timesteps[t_start]], device='cuda')).to(
                'cuda')

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            scale_range = np.linspace(scale_range[0], scale_range[1], len(self.scheduler.timesteps))

            if max_iter_to_alter is None:
                max_iter_to_alter = len(self.scheduler.timesteps) + 1

            # 7. Denoising loop
            localization_update = True
            localization_count = 0
            n_start = 100
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    # update with L_img, L_att and L_prompt
                    with torch.enable_grad():
                        
                        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                        latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                        noise_pred = self.unet(
                            latent_model_input,
                            t,
                            encoder_hidden_states=prompt_embeds,
                            cross_attention_kwargs=cross_attention_kwargs,
                        ).sample
                        self.unet.zero_grad()

                        # perform guidance
                        if do_classifier_free_guidance:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        latents = latents.clone().detach().requires_grad_(True)
                        prompt_embeds = prompt_embeds.clone().detach().requires_grad_(True)
                        curr_step_size = max(0.0001, clip_step_size - 0.0001 * i)

                        # L = L_img + alpha*L_att
                        for q in range(inner_loop_iters):

                            # L_img
                            if i > 0.7*(num_inference_steps-t_start):
                                image_curr = self.decode_latents_new(latents)
                                img_tensor = self.latent_process(image_curr)
                                loss_img = clip_loss.global_clip_loss(img_tensor, prompt_original)
                                loss_img = 1.0 * loss_img
                                del image_curr, img_tensor
                            else:
                                loss_img = 0

                            # L_att
                            # Get max activation value for each subject token
                            max_attention_per_index, maps_curr = self._aggregate_and_get_max_attention_per_token(
                                attention_store=attention_store,
                                indices_to_alter=indices_to_alter,
                                attention_res=attention_res,
                                smooth_attentions=smooth_attentions,
                                sigma=sigma,
                                kernel_size=kernel_size,
                                normalize_eot=sd_2_1,
                                return_maps=True)
                            maps_curr = maps_curr / torch.max(maps_curr)
                            maps_ = maps_curr.clone()
                            map_m = torch.mean(maps_curr)
                            maps_[maps_curr < map_m] = 0
                            maps_[maps_curr >= map_m] = 1
                            maps_curr = maps_
                            n_curr = torch.sum(maps_curr)
                            if i == 0:
                                n_start = n_curr
                            if 10 < n_curr < 50:
                                localization_update = False
                            loss_att, max_attention_per_index = self._perform_att(
                                        latents=latents,
                                        indices_to_alter=indices_to_alter,
                                        text_embeddings=prompt_embeds,
                                        text_input=text_inputs,
                                        attention_store=attention_store,
                                        step_size=n_curr/n_start*scale_factor * np.sqrt(scale_range[i]),
                                        t=t,
                                        attention_res=attention_res,
                                        smooth_attentions=smooth_attentions,
                                        sigma=sigma,
                                        kernel_size=kernel_size,
                                        normalize_eot=sd_2_1)
                            if i >= 0:
                                loss_img = 0.1*loss_img + 0.5*(i/(num_inference_steps-t_start))*loss_att
                                # clamp loss to prevent divergence
                                if isinstance(loss_img, torch.Tensor):
                                    loss_img = torch.clamp(loss_img, max=5.0)
                                # update latent
                                if loss_img != 0 or loss_att != 0:
                                    latents = self._update_latent(latents=latents, loss=loss_img, step_size=curr_step_size*2)
                                    if torch.isnan(latents).any() or torch.isinf(latents).any():
                                        break
                                # L_prompt
                                loss_prompt = loss_img + (1.0-criterion_cosine(prompt_embeds, prompt_embeds_original).mean())
                                if isinstance(loss_prompt, torch.Tensor):
                                    loss_prompt = torch.clamp(loss_prompt, max=5.0)
                                # update embedding
                                prompt_embeds = self._update_latent(latents=prompt_embeds, loss=loss_prompt, step_size=curr_step_size)
                                if torch.isnan(prompt_embeds).any() or torch.isinf(prompt_embeds).any():
                                    break

                                del loss_img, loss_prompt

                            del max_attention_per_index
                            del maps_curr, maps_

                    del noise_pred_uncond, noise_pred_text, noise_pred, latent_model_input
                    gc.collect()
                    torch.cuda.empty_cache()

                    
                    
                    
                    # additional attention optimization with L_att at early stage
                    with torch.enable_grad():

                        if i < 10 and localization_update:
                            latents = latents.clone().detach().requires_grad_(True)
                            # Forward pass of denoising with text conditioning
                            noise_pred_text = self.unet(latents, t,
                                                        encoder_hidden_states=prompt_embeds[1].unsqueeze(0),
                                                        cross_attention_kwargs=cross_attention_kwargs).sample
                            self.unet.zero_grad()
                            e = 0
                            if i in thresholds.keys() and loss_att.item() > 1.0 - thresholds[i]:
                                e += 1
                                del noise_pred_text
                                torch.cuda.empty_cache()
                                
                                loss_att, latents, max_attention_per_index = self._perform_iterative_refinement_step(
                                    # TODO
                                    latents=latents,
                                    indices_to_alter=indices_to_alter,
                                    loss=loss_att,
                                    threshold=thresholds[i],
                                    text_embeddings=prompt_embeds,
                                    text_input=text_inputs,
                                    attention_store=attention_store,
                                    step_size=scale_factor * np.sqrt(scale_range[i]/10),
                                    t=t,
                                    attention_res=attention_res,
                                    smooth_attentions=smooth_attentions,
                                    sigma=sigma,
                                    kernel_size=kernel_size,
                                    normalize_eot=sd_2_1,
                                    max_refinement_steps=5)

                                # Perform gradient update
                                if i < max_iter_to_alter:
                                    loss_att = self._compute_loss(max_attention_per_index=max_attention_per_index)
                                    if loss_att != 0:
                                        latents = self._update_latent(latents=latents, loss=loss_att,
                                                                    step_size=scale_factor * np.sqrt(scale_range[i]))
                                
                                del loss_att, max_attention_per_index
                                        
                        
                    # TODO perform again
                    latents.detach()
                    latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                    latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                    noise_pred = self.unet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        cross_attention_kwargs=cross_attention_kwargs,
                    ).sample

                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    # TODO masked noise blending
                    noise_source_latents = self.scheduler.add_noise(latents_source, torch.randn_like(latents), t)
                    if mask_image is not None:
                        latents = latents * latent_mask + noise_source_latents * (1 - latent_mask)

                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                            (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)

                    del noise_pred_uncond, noise_pred_text, noise_pred, latent_model_input, noise_source_latents
                    gc.collect()
                    torch.cuda.empty_cache()

            # 8. Post-processing
            # print(latents, latents.grad_fn)
            # print("3", latents.grad_fn)
            # latents = latents.clone().detach().requires_grad_(True)
            # print("4", latents.grad_fn)

        # 9. Run safety checker
        image = self.decode_latents_new(latents)
        image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
        del latents
        gc.collect()
        torch.cuda.empty_cache()

        # 10. Convert to PIL
        if output_type == "pil":
            new_image = self.numpy_to_pil(image.detach().cpu().numpy())
            return new_image, image

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
