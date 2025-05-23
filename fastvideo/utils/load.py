import os
from pathlib import Path

import torch
import torch.nn.functional as F
from diffusers import AutoencoderKLHunyuanVideo, AutoencoderKLMochi
from torch import nn
from transformers import AutoTokenizer, T5EncoderModel

from fastvideo.models.hunyuan.modules.models import (
    HYVideoDiffusionTransformer, MMDoubleStreamBlock, MMSingleStreamBlock)

from fastvideo.models.hunyuan.modules.model_audio import (
    HYVideoDiffusionTransformerAudio, MMDoubleStreamBlockAudio, MMSingleStreamBlockAudio)
from fastvideo.models.hunyuan.modules.model_audio_i2v import (
    HYVideoDiffusionTransformerAudioI2V, MMDoubleStreamBlockAudioI2V, MMSingleStreamBlockAudioI2V)
from fastvideo.models.hunyuan.text_encoder import TextEncoder, TextEncoder_i2v
from fastvideo.models.hunyuan.vae.autoencoder_kl_causal_3d import \
    AutoencoderKLCausal3D
from fastvideo.models.hunyuan_hf.modeling_hunyuan import (
    HunyuanVideoSingleTransformerBlock, HunyuanVideoTransformer3DModel,
    HunyuanVideoTransformerBlock)
from fastvideo.models.mochi_hf.modeling_mochi import (MochiTransformer3DModel,
                                                      MochiTransformerBlock)
from fastvideo.utils.logging_ import main_print
from fastvideo.models.hunyuan.constants import (
    PROMPT_TEMPLATE_ENCODE, PROMPT_TEMPLATE_ENCODE_VIDEO,
    PROMPT_TEMPLATE_ENCODE_I2V, PROMPT_TEMPLATE_ENCODE_VIDEO_I2V,
    NEGATIVE_PROMPT, NEGATIVE_PROMPT_I2V,
    PROMPT_TEMPLATE
)

hunyuan_config = {
    "mm_double_blocks_depth": 20,
    "mm_single_blocks_depth": 40,
    "rope_dim_list": [16, 56, 56],
    "hidden_size": 3072,
    "heads_num": 24,
    "mlp_width_ratio": 4,
    "guidance_embed": True,
}



class HunyuanI2VTextEncoderWrapper(nn.Module):

    def __init__(self, pretrained_model_name_or_path, device, args):
        super().__init__()

        text_len = 256
        crop_start = PROMPT_TEMPLATE["dit-llm-encode-video-i2v"].get(
            "crop_start", 0)

        max_length = text_len + crop_start

        # prompt_template
        prompt_template = PROMPT_TEMPLATE["dit-llm-encode-i2v"]

        # prompt_template_video
        prompt_template_video = PROMPT_TEMPLATE["dit-llm-encode-video-i2v"]
        
        if args.i2v_mode:
            image_embed_interleave = 4
        else:
            image_embed_interleave = 1

        self.text_encoder = TextEncoder_i2v(
            text_encoder_type=args.text_encoder,
            max_length=args.text_len
            + (
                PROMPT_TEMPLATE[args.prompt_template_video].get("crop_start", 0)
                if args.prompt_template_video is not None
                else PROMPT_TEMPLATE[args.prompt_template].get("crop_start", 0)
                if args.prompt_template is not None
                else 0
            ),
            text_encoder_precision=args.text_encoder_precision,
            tokenizer_type=args.tokenizer,
            i2v_mode=args.i2v_mode,
            prompt_template=(
                PROMPT_TEMPLATE[args.prompt_template]
                if args.prompt_template is not None
                else None
            ),
            prompt_template_video=(
                PROMPT_TEMPLATE[args.prompt_template_video]
                if args.prompt_template_video is not None
                else None
            ),
            hidden_state_skip_layer=args.hidden_state_skip_layer,
            apply_final_norm=args.apply_final_norm,
            reproduce=args.reproduce,
            logger=None,
            device=device,
            image_embed_interleave=image_embed_interleave
        )

        text_encoder_path_2 = os.path.join(os.environ["MODEL_BASE"],
                                           "text_encoder_2")
        
        self.text_encoder_2 = TextEncoder_i2v(
            text_encoder_type=args.text_encoder_2,
            text_encoder_path=text_encoder_path_2,
            max_length=args.text_len_2,
            tokenizer_type=args.tokenizer_2,
            reproduce=args.reproduce,
            logger=None,
            device=device,
        )

    def encode_i2v(self, prompt, text_encoder, clip_skip=None, semantic_images=None):
        # TODO
        device = self.text_encoder.device
        data_type = "video"
        num_videos_per_prompt = 1

        text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

        if clip_skip is None:
            prompt_outputs = text_encoder.encode(text_inputs,
                                                 data_type="video",
                                                 device=device,
                                                 semantic_images=semantic_images)
            prompt_embeds = prompt_outputs.hidden_state
        else:
            prompt_outputs = text_encoder.encode(
                text_inputs,
                output_hidden_states=True,
                data_type=data_type,
                device=device,
            )
            prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]

            prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                prompt_embeds)

        attention_mask = prompt_outputs.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            bs_embed, seq_len = attention_mask.shape
            attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
            attention_mask = attention_mask.view(
                bs_embed * num_videos_per_prompt, seq_len)

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype,
                                         device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1)
        return (prompt_embeds, attention_mask)


    def encode_prompt(self, prompt, semantic_images=None):
        prompt_embeds, attention_mask = self.encode_i2v(prompt, self.text_encoder, semantic_images=semantic_images)
        prompt_embeds_2, attention_mask_2 = self.encode_i2v(
            prompt, self.text_encoder_2)
        prompt_embeds_2 = F.pad(
            prompt_embeds_2,
            (0, prompt_embeds.shape[2] - prompt_embeds_2.shape[1]),
            value=0,
        ).unsqueeze(1)
        prompt_embeds = torch.cat([prompt_embeds_2, prompt_embeds], dim=1)
        return prompt_embeds, attention_mask



class HunyuanTextEncoderWrapper(nn.Module):

    def __init__(self, pretrained_model_name_or_path, device):
        super().__init__()

        text_len = 256
        crop_start = PROMPT_TEMPLATE["dit-llm-encode-video"].get(
            "crop_start", 0)

        max_length = text_len + crop_start

        # prompt_template
        prompt_template = PROMPT_TEMPLATE["dit-llm-encode"]

        # prompt_template_video
        prompt_template_video = PROMPT_TEMPLATE["dit-llm-encode-video"]
        text_encoder_path = os.path.join(pretrained_model_name_or_path,
                                         "text_encoder")
        self.text_encoder = TextEncoder(
            text_encoder_type="llm",
            text_encoder_path=text_encoder_path,
            max_length=max_length,
            text_encoder_precision="fp16",
            tokenizer_type="llm",
            prompt_template=prompt_template,
            prompt_template_video=prompt_template_video,
            hidden_state_skip_layer=2,
            apply_final_norm=False,
            reproduce=False,
            logger=None,
            device=device,
        )
        text_encoder_path_2 = os.path.join(pretrained_model_name_or_path,
                                           "text_encoder_2")
        self.text_encoder_2 = TextEncoder(
            text_encoder_type="clipL",
            text_encoder_path=text_encoder_path_2,
            max_length=77,
            text_encoder_precision="fp16",
            tokenizer_type="clipL",
            reproduce=False,
            logger=None,
            device=device,
        )

    def encode_(self, prompt, text_encoder, clip_skip=None):
        # TODO
        device = self.text_encoder.device
        data_type = "video"
        num_videos_per_prompt = 1

        text_inputs = text_encoder.text2tokens(prompt, data_type=data_type)

        if clip_skip is None:
            prompt_outputs = text_encoder.encode(text_inputs,
                                                 data_type="video",
                                                 device=device)
            prompt_embeds = prompt_outputs.hidden_state
        else:
            prompt_outputs = text_encoder.encode(
                text_inputs,
                output_hidden_states=True,
                data_type=data_type,
                device=device,
            )
            prompt_embeds = prompt_outputs.hidden_states_list[-(clip_skip + 1)]

            prompt_embeds = text_encoder.model.text_model.final_layer_norm(
                prompt_embeds)

        attention_mask = prompt_outputs.attention_mask
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            bs_embed, seq_len = attention_mask.shape
            attention_mask = attention_mask.repeat(1, num_videos_per_prompt)
            attention_mask = attention_mask.view(
                bs_embed * num_videos_per_prompt, seq_len)

        if text_encoder is not None:
            prompt_embeds_dtype = text_encoder.dtype
        elif self.transformer is not None:
            prompt_embeds_dtype = self.transformer.dtype
        else:
            prompt_embeds_dtype = prompt_embeds.dtype

        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype,
                                         device=device)

        if prompt_embeds.ndim == 2:
            bs_embed, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, -1)
        else:
            bs_embed, seq_len, _ = prompt_embeds.shape
            # duplicate text embeddings for each generation per prompt, using mps friendly method
            prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(
                bs_embed * num_videos_per_prompt, seq_len, -1)
        return (prompt_embeds, attention_mask)

    def encode_prompt(self, prompt):
        prompt_embeds, attention_mask = self.encode_(prompt, self.text_encoder)
        prompt_embeds_2, attention_mask_2 = self.encode_(
            prompt, self.text_encoder_2)
        prompt_embeds_2 = F.pad(
            prompt_embeds_2,
            (0, prompt_embeds.shape[2] - prompt_embeds_2.shape[1]),
            value=0,
        ).unsqueeze(1)
        prompt_embeds = torch.cat([prompt_embeds_2, prompt_embeds], dim=1)
        return prompt_embeds, attention_mask


class MochiTextEncoderWrapper(nn.Module):

    def __init__(self, pretrained_model_name_or_path, device):
        super().__init__()
        self.text_encoder = T5EncoderModel.from_pretrained(
            os.path.join(pretrained_model_name_or_path,
                         "text_encoder")).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(pretrained_model_name_or_path, "tokenizer"))
        self.max_sequence_length = 256

    def encode_prompt(self, prompt):
        device = self.text_encoder.device
        dtype = self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        prompt_attention_mask = text_inputs.attention_mask
        prompt_attention_mask = prompt_attention_mask.bool().to(device)

        untruncated_ids = self.tokenizer(prompt,
                                         padding="longest",
                                         return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[
                -1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(
                untruncated_ids[:, self.max_sequence_length - 1:-1])
            main_print(
                f"Truncated text input: {prompt} to: {removed_text} for model input."
            )
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=prompt_attention_mask)[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)
        prompt_attention_mask = prompt_attention_mask.view(batch_size, -1)

        return prompt_embeds, prompt_attention_mask


def load_hunyuan_state_dict(model, dit_model_name_or_path):
    load_key = "module"
    model_path = dit_model_name_or_path
    bare_model = "unknown"

    state_dict = torch.load(model_path,
                            map_location=lambda storage, loc: storage,
                            weights_only=True)

    if bare_model == "unknown" and ("ema" in state_dict
                                    or "module" in state_dict):
        bare_model = False
    if bare_model is False:
        if load_key in state_dict:
            state_dict = state_dict[load_key]
        else:
            raise KeyError(
                f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                f"are: {list(state_dict.keys())}.")
    model.load_state_dict(state_dict, strict=True)
    return model

def load_hunyuan_audio_state_dict(model, dit_model_name_or_path):
    """
    加载HunyuanAudio模型权重
    
    当dit_model_name_or_path为None时，将返回原始模型，不加载权重
    当dit_model_name_or_path为路径时，从指定路径加载权重
    支持.pt和.safetensors格式的模型权重文件
    """
    # 如果权重路径为None，直接返回初始化模型
    if dit_model_name_or_path is None:
        main_print("警告：未提供权重路径 (dit_model_name_or_path)，模型将使用随机权重初始化。")
        return model
    
    load_key = "module"
    model_path = dit_model_name_or_path
    bare_model = "unknown"

    main_print(f"正在从 {model_path} 加载模型...")
    
    # 根据文件扩展名选择正确的加载方法
    if str(model_path).endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            main_print(f"使用safetensors加载模型: {model_path}")
            state_dict = load_file(model_path)
        except ImportError:
            main_print("警告：未安装safetensors库，尝试使用torch.load加载")
            state_dict = torch.load(model_path,
                                   map_location=lambda storage, loc: storage,
                                   weights_only=True)
    else:
        # 对于.pt文件使用torch.load
        main_print(f"使用torch.load加载模型: {model_path}")
        state_dict = torch.load(model_path,
                               map_location=lambda storage, loc: storage,
                               weights_only=True)

    if bare_model == "unknown" and ("ema" in state_dict
                                    or "module" in state_dict):
        bare_model = False
    if bare_model is False:
        if load_key in state_dict:
            state_dict = state_dict[load_key]
        else:
            # 检查是否是直接的模型状态字典（没有module键）
            main_print(f"模型文件中没有'{load_key}'键，尝试直接加载状态字典。")
            # 打印状态字典中的前几个键，帮助调试
            main_print(f"状态字典包含的键: {list(state_dict.keys())[:5]}...")
    
    # 加载模型权重
    try:
        model.load_state_dict(state_dict, strict=False)
        main_print("模型加载成功！")
    except Exception as e:
        main_print(f"模型加载时出错: {e}")
        # 尝试进行键名称修复
        main_print("尝试修复键名称不匹配问题...")
        fixed_state_dict = {}
        for k, v in state_dict.items():
            # 移除可能的前缀（如果有的话）
            if k.startswith('transformer.'):
                fixed_state_dict[k[12:]] = v
            else:
                fixed_state_dict[k] = v
        model.load_state_dict(fixed_state_dict, strict=False)
        main_print("使用修复的状态字典加载模型成功！")
    
    return model

def load_transformer(
    model_type,
    dit_model_name_or_path,
    pretrained_model_name_or_path,
    master_weight_type,
):
    if model_type == "mochi":
        if dit_model_name_or_path:
            transformer = MochiTransformer3DModel.from_pretrained(
                dit_model_name_or_path,
                torch_dtype=master_weight_type,
                # torch_dtype=torch.bfloat16 if args.use_lora else torch.float32,
            )
        else:
            transformer = MochiTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=master_weight_type,
                # torch_dtype=torch.bfloat16 if args.use_lora else torch.float32,
            )
    elif model_type == "hunyuan_hf":
        if dit_model_name_or_path:
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                dit_model_name_or_path,
                torch_dtype=master_weight_type,
                # torch_dtype=torch.bfloat16 if args.use_lora else torch.float32,
            )
        else:
            transformer = HunyuanVideoTransformer3DModel.from_pretrained(
                pretrained_model_name_or_path,
                subfolder="transformer",
                torch_dtype=master_weight_type,
                # torch_dtype=torch.bfloat16 if args.use_lora else torch.float32,
            )
    elif model_type == "hunyuan":
        transformer = HYVideoDiffusionTransformer(
            in_channels=16,
            out_channels=16,
            **hunyuan_config,
            dtype=master_weight_type,
        )
        transformer = load_hunyuan_state_dict(transformer,
                                              dit_model_name_or_path)
        if master_weight_type == torch.bfloat16:
            transformer = transformer.bfloat16()
    elif model_type == "hunyuan_audio":
        transformer = HYVideoDiffusionTransformerAudio(
            in_channels=16,
            out_channels=16,
            **hunyuan_config,
            dtype=master_weight_type,
        )
        transformer = load_hunyuan_audio_state_dict(transformer,
                                              dit_model_name_or_path)
        if master_weight_type == torch.bfloat16:
            transformer = transformer.bfloat16()
    elif model_type == "hunyuan_audio_i2v":
        transformer = HYVideoDiffusionTransformerAudioI2V(
            in_channels=16,
            out_channels=16,
            **hunyuan_config,
            dtype=master_weight_type,
        )
        transformer = load_hunyuan_audio_state_dict(transformer,
                                              dit_model_name_or_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return transformer


def load_vae(model_type, pretrained_model_name_or_path):
    weight_dtype = torch.float32
    if model_type == "mochi":
        vae = AutoencoderKLMochi.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=weight_dtype)
        autocast_type = torch.bfloat16
        fps = 30
    elif model_type == "hunyuan_hf":
        vae = AutoencoderKLHunyuanVideo.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
            torch_dtype=weight_dtype)
        autocast_type = torch.bfloat16
        fps = 24
    elif model_type == "hunyuan" or model_type == "hunyuan_audio" or model_type == "hunyuan_audio_i2v":
        vae_precision = torch.float32
        vae_path = os.path.join(pretrained_model_name_or_path,
                                "hunyuan-video-t2v-720p/vae")

        config = AutoencoderKLCausal3D.load_config(vae_path)
        vae = AutoencoderKLCausal3D.from_config(config)

        vae_ckpt = Path(vae_path) / "pytorch_model.pt"
        assert vae_ckpt.exists(), f"VAE checkpoint not found: {vae_ckpt}"

        ckpt = torch.load(vae_ckpt, map_location="cpu", weights_only=True)
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        if any(k.startswith("vae.") for k in ckpt.keys()):
            ckpt = {
                k.replace("vae.", ""): v
                for k, v in ckpt.items() if k.startswith("vae.")
            }
        vae.load_state_dict(ckpt)
        vae = vae.to(dtype=vae_precision)
        vae.requires_grad_(False)
        vae.eval()
        autocast_type = torch.float32
        fps = 24
    return vae, autocast_type, fps


def load_text_encoder(model_type, pretrained_model_name_or_path, device, args):
    if model_type == "mochi":
        text_encoder = MochiTextEncoderWrapper(pretrained_model_name_or_path,
                                               device)
    elif model_type == "hunyuan" or model_type == "hunyuan_hf":
        text_encoder = HunyuanTextEncoderWrapper(pretrained_model_name_or_path,
                                                 device)
    elif model_type == "hunyuan_audio_i2v":
        text_encoder = HunyuanI2VTextEncoderWrapper(pretrained_model_name_or_path,
                                                 device, args)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return text_encoder


def get_no_split_modules(transformer):
    # if of type MochiTransformer3DModel
    if isinstance(transformer, MochiTransformer3DModel):
        return (MochiTransformerBlock, )
    elif isinstance(transformer, HunyuanVideoTransformer3DModel):
        return (HunyuanVideoSingleTransformerBlock,
                HunyuanVideoTransformerBlock)
    elif isinstance(transformer, HYVideoDiffusionTransformer):
        return (MMDoubleStreamBlock, MMSingleStreamBlock)
    elif isinstance(transformer, HYVideoDiffusionTransformerAudio):
        return (MMDoubleStreamBlockAudio, MMSingleStreamBlockAudio)
    elif isinstance(transformer, HYVideoDiffusionTransformerAudioI2V):
        return (MMDoubleStreamBlockAudioI2V, MMSingleStreamBlockAudioI2V)
    else:
        raise ValueError(f"Unsupported transformer type: {type(transformer)}")


if __name__ == "__main__":
    # test encode prompt
    device = torch.cuda.current_device()
    pretrained_model_name_or_path = "data/hunyuan"
    text_encoder = load_text_encoder("hunyuan", pretrained_model_name_or_path,
                                     device)
    prompt = "A man on stage claps his hands together while facing the audience. The audience, visible in the foreground, holds up mobile devices to record the event, capturing the moment from various angles. The background features a large banner with text identifying the man on stage. Throughout the sequence, the man's expression remains engaged and directed towards the audience. The camera angle remains constant, focusing on capturing the interaction between the man on stage and the audience."
    prompt_embeds, attention_mask = text_encoder.encode_prompt(prompt)
