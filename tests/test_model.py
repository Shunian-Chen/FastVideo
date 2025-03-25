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
from fastvideo.models.hunyuan.text_encoder import TextEncoder
from fastvideo.models.hunyuan.vae.autoencoder_kl_causal_3d import \
    AutoencoderKLCausal3D
from fastvideo.models.hunyuan_hf.modeling_hunyuan import (
    HunyuanVideoSingleTransformerBlock, HunyuanVideoTransformer3DModel,
    HunyuanVideoTransformerBlock)
from fastvideo.models.mochi_hf.modeling_mochi import (MochiTransformer3DModel,
                                                      MochiTransformerBlock)
from fastvideo.utils.logging_ import main_print

def load_hunyuan_audio_state_dict(model, dit_model_name_or_path):
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
    model.load_state_dict(state_dict, strict=False)
    return model
    
hunyuan_config = {
    "mm_double_blocks_depth": 20,
    "mm_single_blocks_depth": 40,
    "rope_dim_list": [16, 56, 56],
    "hidden_size": 3072,
    "heads_num": 24,
    "mlp_width_ratio": 4,
    "guidance_embed": True,
}

transformer = HYVideoDiffusionTransformerAudio(
            in_channels=16,
            out_channels=16,
            **hunyuan_config,
            dtype=torch.bfloat16,
        )
transformer = load_hunyuan_audio_state_dict(transformer,
                                              "/sds_wangby/models/HunyuanVideo/HunyuanVideo/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt")
print(transformer)