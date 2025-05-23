from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import math
import torch.nn.init as init
from torch.nn.functional import scaled_dot_product_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models import ModelMixin
from einops import rearrange

from fastvideo.models.hunyuan.modules.posemb_layers import \
    get_nd_rotary_pos_embed
from fastvideo.utils.parallel_states import nccl_info

from .activation_layers import get_activation_layer
from .attenion import parallel_attention, cross_attention
from .embed_layers import PatchEmbed, TextProjection, TimestepEmbedder, FaceProjModel, AudioProjModel
from .mlp_layers import MLP, FinalLayer, MLPEmbedder
from .modulate_layers import ModulateDiT, apply_gate, modulate
from .norm_layers import get_norm_layer
from .posemb_layers import apply_rotary_emb
from .token_refiner import SingleTokenRefiner
from loguru import logger

class MMDoubleStreamBlockAudio(nn.Module):
    """
    A multimodal dit block with separate modulation for
    text and image/video, see more details (SD3): https://arxiv.org/abs/2403.03206
                                     (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qkv_bias: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)

        self.img_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.img_norm1 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6,
                                      **factory_kwargs)

        self.img_attn_qkv = nn.Linear(hidden_size,
                                      hidden_size * 3,
                                      bias=qkv_bias,
                                      **factory_kwargs)
        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.img_attn_q_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())
        self.img_attn_k_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())
        self.img_attn_proj = nn.Linear(hidden_size,
                                       hidden_size,
                                       bias=qkv_bias,
                                       **factory_kwargs)

        self.img_norm2 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6,
                                      **factory_kwargs)
        self.img_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )

        self.txt_mod = ModulateDiT(
            hidden_size,
            factor=6,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.txt_norm1 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6,
                                      **factory_kwargs)

        self.txt_attn_qkv = nn.Linear(hidden_size,
                                      hidden_size * 3,
                                      bias=qkv_bias,
                                      **factory_kwargs)
        self.txt_attn_q_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())
        self.txt_attn_k_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())
        self.txt_attn_proj = nn.Linear(hidden_size,
                                       hidden_size,
                                       bias=qkv_bias,
                                       **factory_kwargs)

        self.txt_norm2 = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6,
                                      **factory_kwargs)
        self.txt_mlp = MLP(
            hidden_size,
            mlp_hidden_dim,
            act_layer=get_activation_layer(mlp_act_type),
            bias=True,
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None

        # # 添加face的cross attention模块
        # self.face_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        # self.face_attn_q_norm = (qk_norm_layer(
        #     head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        #                         if qk_norm else nn.Identity())
        # self.face_attn_k_norm = (qk_norm_layer(
        #     head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        #                         if qk_norm else nn.Identity())
        

        # 添加audio的cross attention模块
        self.audio_scale = nn.Parameter(torch.ones(1))
        self.audio_norm = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6,
                                      **factory_kwargs)
        self.audio_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=qkv_bias, **factory_kwargs)
        init.xavier_uniform_(self.audio_attn_qkv.weight)
        init.zeros_(self.audio_attn_qkv.bias)
        self.audio_attn_q_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())
        self.audio_attn_k_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        img: torch.Tensor,
        txt: torch.Tensor,
        vec: torch.Tensor,
        freqs_cis: tuple = None,
        text_mask: torch.Tensor = None,
        audio_emb: torch.Tensor = None,
        face_emb: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        (
            img_mod1_shift,
            img_mod1_scale,
            img_mod1_gate,
            img_mod2_shift,
            img_mod2_scale,
            img_mod2_gate,
        ) = self.img_mod(vec).chunk(6, dim=-1)
        (
            txt_mod1_shift,
            txt_mod1_scale,
            txt_mod1_gate,
            txt_mod2_shift,
            txt_mod2_scale,
            txt_mod2_gate,
        ) = self.txt_mod(vec).chunk(6, dim=-1)

        # Prepare image for attention.
        img_modulated = self.img_norm1(img)
        img_modulated = modulate(img_modulated,
                                 shift=img_mod1_shift,
                                 scale=img_mod1_scale)
        img_qkv = self.img_attn_qkv(img_modulated)
        img_q, img_k, img_v = rearrange(img_qkv,
                                        "B L (K H D) -> K B L H D",
                                        K=3,
                                        H=self.heads_num)
        # Apply QK-Norm if needed
        img_q = self.img_attn_q_norm(img_q).to(img_v)
        img_k = self.img_attn_k_norm(img_k).to(img_v)

        # Apply RoPE if needed.
        if freqs_cis is not None:

            def shrink_head(encoder_state, dim):
                local_heads = encoder_state.shape[dim] // nccl_info.sp_size
                return encoder_state.narrow(
                    dim, nccl_info.rank_within_group * local_heads,
                    local_heads)

            freqs_cis = (
                shrink_head(freqs_cis[0], dim=0),
                shrink_head(freqs_cis[1], dim=0),
            )

            img_qq, img_kk = apply_rotary_emb(img_q,
                                              img_k,
                                              freqs_cis,
                                              head_first=False)
            assert (
                img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
            ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
            img_q, img_k = img_qq, img_kk

        # Prepare txt for attention.
        txt_modulated = self.txt_norm1(txt)
        txt_modulated = modulate(txt_modulated,
                                 shift=txt_mod1_shift,
                                 scale=txt_mod1_scale)
        txt_qkv = self.txt_attn_qkv(txt_modulated)
        txt_q, txt_k, txt_v = rearrange(txt_qkv,
                                        "B L (K H D) -> K B L H D",
                                        K=3,
                                        H=self.heads_num)
        # Apply QK-Norm if needed.
        txt_q = self.txt_attn_q_norm(txt_q).to(txt_v)
        txt_k = self.txt_attn_k_norm(txt_k).to(txt_v)

        # logger.info(f"="*80)
        # logger.info(f"parallel attention")
        # logger.info(f"="*80)
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
        )
        # logger.info(f"parallel attention done")
        # logger.info(f"="*80)

        # attention computation end

        img_attn, txt_attn = attn[:, :img.shape[1]], attn[:, img.shape[1]:]

        # Calculate the img blocks.
        img = img + apply_gate(self.img_attn_proj(img_attn),
                               gate=img_mod1_gate)

        audio_emb = self.audio_norm(audio_emb)
        aud_qkv = self.audio_attn_qkv(audio_emb)
        aud_q, aud_k, aud_v = rearrange(aud_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        aud_q = self.audio_attn_q_norm(aud_q).to(aud_v)
        aud_k = self.audio_attn_k_norm(aud_k).to(aud_v)
        # audio mask
        audio_mask = torch.ones_like(aud_q[:, :, 1, 1])
        aud_out = cross_attention(
            (img_q, aud_q),
            (img_k, aud_k),
            (img_v, aud_v),
            text_mask=audio_mask,
        )

        max_value = 2
        scale = torch.clamp(self.audio_scale, max=max_value)
        img = img + aud_out * scale

        img = img + apply_gate(
            self.img_mlp(
                modulate(self.img_norm2(img),
                         shift=img_mod2_shift,
                         scale=img_mod2_scale)),
            gate=img_mod2_gate,
        )

        # Calculate the txt blocks.
        txt = txt + apply_gate(self.txt_attn_proj(txt_attn),
                               gate=txt_mod1_gate)
        txt = txt + apply_gate(
            self.txt_mlp(
                modulate(self.txt_norm2(txt),
                         shift=txt_mod2_shift,
                         scale=txt_mod2_scale)),
            gate=txt_mod2_gate,
        )

        return img, txt


class MMSingleStreamBlockAudio(nn.Module):
    """
    A DiT block with parallel linear layers as described in
    https://arxiv.org/abs/2302.05442 and adapted modulation interface.
    Also refer to (SD3): https://arxiv.org/abs/2403.03206
                  (Flux.1): https://github.com/black-forest-labs/flux
    """

    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        qk_scale: float = None,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.deterministic = False
        self.hidden_size = hidden_size
        self.heads_num = heads_num
        head_dim = hidden_size // heads_num
        mlp_hidden_dim = int(hidden_size * mlp_width_ratio)
        self.mlp_hidden_dim = mlp_hidden_dim
        self.scale = qk_scale or head_dim**-0.5

        # qkv and mlp_in
        self.linear1 = nn.Linear(hidden_size, hidden_size * 3 + mlp_hidden_dim,
                                 **factory_kwargs)
        # proj and mlp_out
        self.linear2 = nn.Linear(hidden_size + mlp_hidden_dim, hidden_size,
                                 **factory_kwargs)

        qk_norm_layer = get_norm_layer(qk_norm_type)
        self.q_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                       if qk_norm else nn.Identity())
        self.k_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                       if qk_norm else nn.Identity())

        self.pre_norm = nn.LayerNorm(hidden_size,
                                     elementwise_affine=False,
                                     eps=1e-6,
                                     **factory_kwargs)

        self.mlp_act = get_activation_layer(mlp_act_type)()
        self.modulation = ModulateDiT(
            hidden_size,
            factor=3,
            act_layer=get_activation_layer("silu"),
            **factory_kwargs,
        )
        self.hybrid_seq_parallel_attn = None


        # # 添加face的cross attention模块
        # self.face_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, **factory_kwargs)
        # self.face_attn_q_norm = (qk_norm_layer(
        #     head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        #                         if qk_norm else nn.Identity())
        # self.face_attn_k_norm = (qk_norm_layer(
        #     head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
        #                         if qk_norm else nn.Identity())
        

        # 添加audio的cross attention模块
        self.audio_scale = nn.Parameter(torch.ones(1))
        self.audio_norm = nn.LayerNorm(hidden_size,
                                      elementwise_affine=False,
                                      eps=1e-6,
                                      **factory_kwargs)
        self.audio_attn_qkv = nn.Linear(hidden_size, hidden_size * 3, bias=True, **factory_kwargs)
        # print(f"qkv.shape: {self.audio_attn_qkv.weight.shape}")
        init.xavier_uniform_(self.audio_attn_qkv.weight)
        init.zeros_(self.audio_attn_qkv.bias)
        self.audio_attn_q_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())
        self.audio_attn_k_norm = (qk_norm_layer(
            head_dim, elementwise_affine=True, eps=1e-6, **factory_kwargs)
                                if qk_norm else nn.Identity())

    def enable_deterministic(self):
        self.deterministic = True

    def disable_deterministic(self):
        self.deterministic = False

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        txt_len: int,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor] = None,
        text_mask: torch.Tensor = None,
        audio_emb: torch.Tensor = None,
        face_emb: torch.Tensor = None,
    ) -> torch.Tensor:
        mod_shift, mod_scale, mod_gate = self.modulation(vec).chunk(3, dim=-1)
        x_mod = modulate(self.pre_norm(x), shift=mod_shift, scale=mod_scale)
        qkv, mlp = torch.split(self.linear1(x_mod),
                               [3 * self.hidden_size, self.mlp_hidden_dim],
                               dim=-1)

        q, k, v = rearrange(qkv,
                            "B L (K H D) -> K B L H D",
                            K=3,
                            H=self.heads_num)

        # Apply QK-Norm if needed.
        q = self.q_norm(q).to(v)
        k = self.k_norm(k).to(v)

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // nccl_info.sp_size
            return encoder_state.narrow(
                dim, nccl_info.rank_within_group * local_heads, local_heads)

        freqs_cis = (shrink_head(freqs_cis[0],
                                 dim=0), shrink_head(freqs_cis[1], dim=0))

        img_q, txt_q = q[:, :-txt_len, :, :], q[:, -txt_len:, :, :]
        img_k, txt_k = k[:, :-txt_len, :, :], k[:, -txt_len:, :, :]
        img_v, txt_v = v[:, :-txt_len, :, :], v[:, -txt_len:, :, :]
        img_qq, img_kk = apply_rotary_emb(img_q,
                                          img_k,
                                          freqs_cis,
                                          head_first=False)
        assert (
            img_qq.shape == img_q.shape and img_kk.shape == img_k.shape
        ), f"img_kk: {img_qq.shape}, img_q: {img_q.shape}, img_kk: {img_kk.shape}, img_k: {img_k.shape}"
        img_q, img_k = img_qq, img_kk

        # print(f"img_q shape: {img_q.shape}, txt_q shape: {txt_q.shape}, img_k shape: {img_k.shape}, txt_k shape: {txt_k.shape}")
        attn = parallel_attention(
            (img_q, txt_q),
            (img_k, txt_k),
            (img_v, txt_v),
            img_q_len=img_q.shape[1],
            img_kv_len=img_k.shape[1],
            text_mask=text_mask,
        )

        img_attn, txt_attn = attn[:, :img_q.shape[1]], attn[:, img_q.shape[1]:]

        # print(f"img_attn shape: {img_attn.shape}, txt_attn shape: {txt_attn.shape}")

        # audio cross attention
        # print(f"audio_emb shape: {audio_emb.shape}")
        audio_emb = self.audio_norm(audio_emb)
        # print(f"audio_emb shape after norm: {audio_emb.shape}")
        aud_qkv = self.audio_attn_qkv(audio_emb)
        # print(f"aud_qkv shape: {aud_qkv.shape}")
        aud_q, aud_k, aud_v = rearrange(aud_qkv, "B L (K H D) -> K B L H D", K=3, H=self.heads_num)
        # print(f"aud_q shape: {aud_q.shape}, aud_k shape: {aud_k.shape}, aud_v shape: {aud_v.shape}")
        aud_q = self.audio_attn_q_norm(aud_q).to(aud_v)
        aud_k = self.audio_attn_k_norm(aud_k).to(aud_v)
        audio_mask = torch.ones_like(aud_q[:, :, 1, 1])
        
        aud_out = cross_attention(
            (img_q, aud_q),
            (img_k, aud_k),
            (img_v, aud_v),
            text_mask=audio_mask,
        )

        # print(f"aud_out shape: {aud_out.shape}")
        max_value = 2
        scale = torch.clamp(self.audio_scale, max=max_value)
        img_attn = img_attn + aud_out * scale

        # print(f"img_attn shape: {img_attn.shape}, txt_attn shape: {txt_attn.shape}")

        attn = torch.cat((img_attn, txt_attn), dim=1)

        # print(f"attn shape: {attn.shape}")

        # Compute activation in mlp stream, cat again and run second linear layer.
        output = self.linear2(torch.cat((attn, self.mlp_act(mlp)), 2))
        output = x + apply_gate(output, gate=mod_gate)

        # attention computation end

        

        return output



class HYVideoDiffusionTransformerAudio(ModelMixin, ConfigMixin):
    """
    HunyuanVideo Transformer backbone

    Inherited from ModelMixin and ConfigMixin for compatibility with diffusers' sampler StableDiffusionPipeline.

    Reference:
    [1] Flux.1: https://github.com/black-forest-labs/flux
    [2] MMDiT: http://arxiv.org/abs/2403.03206

    Parameters
    ----------
    args: argparse.Namespace
        The arguments parsed by argparse.
    patch_size: list
        The size of the patch.
    in_channels: int
        The number of input channels.
    out_channels: int
        The number of output channels.
    hidden_size: int
        The hidden size of the transformer backbone.
    heads_num: int
        The number of attention heads.
    mlp_width_ratio: float
        The ratio of the hidden size of the MLP in the transformer block.
    mlp_act_type: str
        The activation function of the MLP in the transformer block.
    depth_double_blocks: int
        The number of transformer blocks in the double blocks.
    depth_single_blocks: int
        The number of transformer blocks in the single blocks.
    rope_dim_list: list
        The dimension of the rotary embedding for t, h, w.
    qkv_bias: bool
        Whether to use bias in the qkv linear layer.
    qk_norm: bool
        Whether to use qk norm.
    qk_norm_type: str
        The type of qk norm.
    guidance_embed: bool
        Whether to use guidance embedding for distillation.
    text_projection: str
        The type of the text projection, default is single_refiner.
    use_attention_mask: bool
        Whether to use attention mask for text encoder.
    dtype: torch.dtype
        The dtype of the model.
    device: torch.device
        The device of the model.
    """

    @register_to_config
    def __init__(
        self,
        patch_size: list = [1, 2, 2],
        in_channels: int = 4,  # Should be VAE.config.latent_channels.
        out_channels: int = None,
        hidden_size: int = 3072,
        heads_num: int = 24,
        mlp_width_ratio: float = 4.0,
        mlp_act_type: str = "gelu_tanh",
        mm_double_blocks_depth: int = 20,
        mm_single_blocks_depth: int = 40,
        rope_dim_list: List[int] = [16, 56, 56],
        qkv_bias: bool = True,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        guidance_embed: bool = False,  # For modulation.
        text_projection: str = "single_refiner",
        use_attention_mask: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        text_states_dim: int = 4096,
        text_states_dim_2: int = 768,
        rope_theta: int = 256,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.unpatchify_channels = self.out_channels
        self.guidance_embed = guidance_embed
        self.rope_dim_list = rope_dim_list
        self.rope_theta = rope_theta
        # Text projection. Default to linear projection.
        # Alternative: TokenRefiner. See more details (LI-DiT): http://arxiv.org/abs/2406.11831
        self.use_attention_mask = use_attention_mask
        self.text_projection = text_projection

        if hidden_size % heads_num != 0:
            raise ValueError(
                f"Hidden size {hidden_size} must be divisible by heads_num {heads_num}"
            )
        pe_dim = hidden_size // heads_num
        if sum(rope_dim_list) != pe_dim:
            raise ValueError(
                f"Got {rope_dim_list} but expected positional dim {pe_dim}")
        self.hidden_size = hidden_size
        self.heads_num = heads_num

        # image projection
        self.img_in = PatchEmbed(self.patch_size, self.in_channels,
                                 self.hidden_size, **factory_kwargs)

        # text projection
        if self.text_projection == "linear":
            self.txt_in = TextProjection(
                self.config.text_states_dim,
                self.hidden_size,
                get_activation_layer("silu"),
                **factory_kwargs,
            )
        elif self.text_projection == "single_refiner":
            self.txt_in = SingleTokenRefiner(
                self.config.text_states_dim,
                hidden_size,
                heads_num,
                depth=2,
                **factory_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}")

        # time modulation
        self.time_in = TimestepEmbedder(self.hidden_size,
                                        get_activation_layer("silu"),
                                        **factory_kwargs)

        # text modulation
        self.vector_in = MLPEmbedder(self.config.text_states_dim_2,
                                     self.hidden_size, **factory_kwargs)

        # guidance modulation
        self.guidance_in = (TimestepEmbedder(
            self.hidden_size, get_activation_layer("silu"), **factory_kwargs)
                            if guidance_embed else None)

        # double blocks
        self.double_blocks = nn.ModuleList([
            MMDoubleStreamBlockAudio(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                qkv_bias=qkv_bias,
                **factory_kwargs,
            ) for _ in range(mm_double_blocks_depth)
        ])

        # single blocks
        self.single_blocks = nn.ModuleList([
            MMSingleStreamBlockAudio(
                self.hidden_size,
                self.heads_num,
                mlp_width_ratio=mlp_width_ratio,
                mlp_act_type=mlp_act_type,
                qk_norm=qk_norm,
                qk_norm_type=qk_norm_type,
                **factory_kwargs,
            ) for _ in range(mm_single_blocks_depth)
        ])

        self.final_layer = FinalLayer(
            self.hidden_size,
            self.patch_size,
            self.out_channels,
            get_activation_layer("silu"),
            **factory_kwargs,
        )

        self.audio_proj = AudioProjModel(output_dim=768, context_tokens=4)
        self.face_proj = FaceProjModel(hidden_size=hidden_size)

    def enable_deterministic(self):
        for block in self.double_blocks:
            block.enable_deterministic()
        for block in self.single_blocks:
            block.enable_deterministic()

    def disable_deterministic(self):
        for block in self.double_blocks:
            block.disable_deterministic()
        for block in self.single_blocks:
            block.disable_deterministic()

    def get_rotary_pos_embed(self, rope_sizes):
        target_ndim = 3

        head_dim = self.hidden_size // self.heads_num
        rope_dim_list = self.rope_dim_list
        if rope_dim_list is None:
            rope_dim_list = [
                head_dim // target_ndim for _ in range(target_ndim)
            ]
        assert (
            sum(rope_dim_list) == head_dim
        ), "sum(rope_dim_list) should equal to head_dim of attention layer"
        freqs_cos, freqs_sin = get_nd_rotary_pos_embed(
            rope_dim_list,
            rope_sizes,
            theta=self.rope_theta,
            use_real=True,
            theta_rescale_factor=1,
        )
        return freqs_cos, freqs_sin
        # x: torch.Tensor,
        # t: torch.Tensor,  # Should be in range(0, 1000).
        # text_states: torch.Tensor = None,
        # text_mask: torch.Tensor = None,  # Now we don't use it.
        # text_states_2: Optional[torch.Tensor] = None,  # Text embedding for modulation.
        # guidance: torch.Tensor = None,  # Guidance for modulation, should be cfg_scale x 1000.
        # return_dict: bool = True,

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: torch.Tensor,
        audio_emb: torch.Tensor,  # 新增
        face_emb: torch.Tensor = None,   # 新增
        output_features=False,
        output_features_stride=8,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
        guidance=None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if guidance is None:
            guidance = torch.tensor([6016.0],
                                    device=hidden_states.device,
                                    dtype=torch.bfloat16)
        img = x = hidden_states
        text_mask = encoder_attention_mask
        t = timestep
        
        frame_num = hidden_states.shape[2]
        # print(f"frame_num: {frame_num}")
        # print(f"hidden_states shape: {hidden_states.shape}")

        # 处理audio_emb和face_emb,参考BaseTransformer的处理方式
        if audio_emb is not None:

            # print(f"audio_emb shape before proj: {audio_emb.shape}")
            audio_emb = self.audio_proj(audio_emb)
            # print(f"audio_emb shape after proj: {audio_emb.shape}")
            _, f, _ = audio_emb.shape
            assert f==frame_num, print(f"audio_emb frame_num: {f}, hidden_states frame_num: {frame_num}")
            # audio_emb = rearrange(audio_emb, "b f m c -> (b f) m c")
            # logger.info(f"audio_emb shape after rearrange: {audio_emb.shape}")
            # logger.info(f"="*80)

        # if face_emb is not None:
        #     # logger.info(f"face_emb shape before proj: {face_emb.shape}")
        #     print("="*80)
        #     print(f"face_emb shape before proj: {face_emb.shape}")
        #     print("="*80)
        #     face_emb = face_emb.unsqueeze(1)
        #     face_emb = self.face_proj(face_emb)
        #     print("="*80)
        #     print(f"face_emb shape after proj: {face_emb.shape}")
        #     print("="*80)
        #     face_emb = face_emb.unsqueeze(1).repeat(1, frame_num, 1, 1)
        #     face_emb = rearrange(face_emb, "b f d c -> (b f) d c")
        #     print("="*80)
        #     print(f"face_emb shape after rearrange: {face_emb.shape}")
        #     print("="*80)

            # logger.info(f"face_emb shape after proj: {face_emb.shape}")
            # logger.info(f"="*80)
        
        # 原有的文本和图像处理代码
        txt = encoder_hidden_states[:, 1:]
        text_states_2 = encoder_hidden_states[:, 0, :self.config.
                                              text_states_dim_2]
        _, _, ot, oh, ow = x.shape  # codespell:ignore
        tt, th, tw = (
            ot // self.patch_size[0],  # codespell:ignore
            oh // self.patch_size[1],  # codespell:ignore
            ow // self.patch_size[2],  # codespell:ignore
        )
        original_tt = nccl_info.sp_size * tt
        freqs_cos, freqs_sin = self.get_rotary_pos_embed((original_tt, th, tw))
        # Prepare modulation vectors.
        vec = self.time_in(t)

        # text modulation
        vec = vec + self.vector_in(text_states_2)

        # guidance modulation
        if self.guidance_embed:
            if guidance is None:
                raise ValueError(
                    "Didn't get guidance strength for guidance distilled model."
                )

            # our timestep_embedding is merged into guidance_in(TimestepEmbedder)
            vec = vec + self.guidance_in(guidance)

        # Embed image and text.
        img = self.img_in(img) # [b, patch_dim, hid]
        if self.text_projection == "linear":
            txt = self.txt_in(txt)
        elif self.text_projection == "single_refiner":
            txt = self.txt_in(txt, t,
                              text_mask if self.use_attention_mask else None)
        else:
            raise NotImplementedError(
                f"Unsupported text_projection: {self.text_projection}")

        txt_seq_len = txt.shape[1]
        img_seq_len = img.shape[1]

        freqs_cis = (freqs_cos, freqs_sin) if freqs_cos is not None else None
        # --------------------- Pass through DiT blocks ------------------------
        # 修改double blocks的参数传递,添加audio_emb和face_emb
        for _, block in enumerate(self.double_blocks):
            double_block_args = [img, txt, vec, freqs_cis, text_mask, audio_emb, face_emb]
            img, txt = block(*double_block_args)


        # Merge txt and img to pass through single stream blocks.
        # 修改single blocks的参数传递
        x = torch.cat((img, txt), 1)
        if output_features:
            features_list = []
        logger.info(f"="*80)
        logger.info(f"single blocks inference")
        logger.info(f"="*80)
        if len(self.single_blocks) > 0:
            for _, block in enumerate(self.single_blocks):
                single_block_args = [
                    x,
                    vec,
                    txt_seq_len,
                    (freqs_cos, freqs_sin),
                    text_mask,
                    audio_emb,
                    face_emb
                ]
                x = block(*single_block_args)
                if output_features and _ % output_features_stride == 0:
                    features_list.append(x[:, :img_seq_len, ...])
        logger.info(f"="*80)
        logger.info(f"single blocks inference done")
        logger.info(f"="*80)

        # 后续处理保持不变
        img = x[:, :img_seq_len, ...]

        # ---------------------------- Final layer ------------------------------
        img = self.final_layer(img,
                               vec)  # (N, T, patch_size ** 2 * out_channels)

        img = self.unpatchify(img, tt, th, tw)
        
        assert not return_dict, "return_dict is not supported."
        if output_features:
            features_list = torch.stack(features_list, dim=0)
        else:
            features_list = None
        return (img, features_list)

    def unpatchify(self, x, t, h, w):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.unpatchify_channels
        pt, ph, pw = self.patch_size
        assert t * h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], t, h, w, c, pt, ph, pw))
        x = torch.einsum("nthwcopq->nctohpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))

        return imgs

    def params_count(self):
        counts = {
            "double":
            sum([
                sum(p.numel() for p in block.img_attn_qkv.parameters()) +
                sum(p.numel() for p in block.img_attn_proj.parameters()) +
                sum(p.numel() for p in block.img_mlp.parameters()) +
                sum(p.numel() for p in block.txt_attn_qkv.parameters()) +
                sum(p.numel() for p in block.txt_attn_proj.parameters()) +
                sum(p.numel() for p in block.txt_mlp.parameters())
                for block in self.double_blocks
            ]),
            "single":
            sum([
                sum(p.numel() for p in block.linear1.parameters()) +
                sum(p.numel() for p in block.linear2.parameters())
                for block in self.single_blocks
            ]),
            "total":
            sum(p.numel() for p in self.parameters()),
        }
        counts["attn+mlp"] = counts["double"] + counts["single"]
        return counts

#################################################################################
#                             HunyuanVideo Configs                              #
#################################################################################

HUNYUAN_VIDEO_CONFIG = {
    "HYVideo-T/2": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
    },
    "HYVideo-T/2-cfgdistill": {
        "mm_double_blocks_depth": 20,
        "mm_single_blocks_depth": 40,
        "rope_dim_list": [16, 56, 56],
        "hidden_size": 3072,
        "heads_num": 24,
        "mlp_width_ratio": 4,
        "guidance_embed": True,
    },
}
