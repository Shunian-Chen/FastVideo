import math

import torch
import torch.nn as nn
from einops import rearrange

from ..utils.helpers import to_2tuple


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding

    Image to Patch Embedding using Conv2d

    A convolution based approach to patchifying a 2D image w/ embedding projection.

    Based on the impl in https://github.com/google-research/vision_transformer

    Hacked together by / Copyright 2020 Ross Wightman

    Remove the _assert function in forward function to be compatible with multi-resolution images.
    """

    def __init__(
        self,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        bias=True,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv3d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
            **factory_kwargs,
        )
        nn.init.xavier_uniform_(
            self.proj.weight.view(self.proj.weight.size(0), -1))
        if bias:
            nn.init.zeros_(self.proj.bias)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TextProjection(nn.Module):
    """
    Projects text embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self,
                 in_channels,
                 hidden_size,
                 act_layer,
                 dtype=None,
                 device=None):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.linear_1 = nn.Linear(
            in_features=in_channels,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs,
        )
        self.act_1 = act_layer()
        self.linear_2 = nn.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            bias=True,
            **factory_kwargs,
        )

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


def timestep_embedding(t, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        t (torch.Tensor): a 1-D Tensor of N indices, one per batch element. These may be fractional.
        dim (int): the dimension of the output.
        max_period (int): controls the minimum frequency of the embeddings.

    Returns:
        embedding (torch.Tensor): An (N, D) Tensor of positional embeddings.

    .. ref_link: https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) *
                      torch.arange(start=0, end=half, dtype=torch.float32) /
                      half).to(device=t.device)
    args = t[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(
        self,
        hidden_size,
        act_layer,
        frequency_embedding_size=256,
        max_period=10000,
        out_size=None,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        if out_size is None:
            out_size = hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size,
                      hidden_size,
                      bias=True,
                      **factory_kwargs),
            act_layer(),
            nn.Linear(hidden_size, out_size, bias=True, **factory_kwargs),
        )
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    def forward(self, t):
        t_freq = timestep_embedding(t, self.frequency_embedding_size,
                                    self.max_period).type(
                                        self.mlp[0].weight.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

class AudioProjModel(torch.nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=12,  # add a new parameter blocks
        channels=768,  # add a new parameter channels
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32
    ):
        super().__init__()

        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = (
            seq_len * blocks * channels
        )  # update input_dim to be the product of blocks and channels.
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        # define multiple linear layers
        self.proj1 = torch.nn.Linear(self.input_dim, intermediate_dim)
        self.proj2 = torch.nn.Linear(intermediate_dim, intermediate_dim)
        self.proj3 = torch.nn.Linear(intermediate_dim, context_tokens * output_dim)

        self.norm = torch.nn.LayerNorm(output_dim)
        
        self.conv1 = torch.nn.Conv1d(in_channels=context_tokens * output_dim,
                                     out_channels=context_tokens * output_dim,
                                     kernel_size=2,
                                     stride=2,
                                     padding=0)

    def forward(self, audio_embeds):
        # merge
        video_length = audio_embeds.shape[1]
        audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
        batch_size, window_size, blocks, channels = audio_embeds.shape
        audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

        audio_embeds = torch.relu(self.proj1(audio_embeds))
        audio_embeds = torch.relu(self.proj2(audio_embeds))

        context_tokens = self.proj3(audio_embeds).reshape(
            batch_size, self.context_tokens, self.output_dim
        )

        # context_tokens = self.norm(context_tokens)
        context_tokens = rearrange(
            context_tokens, "(bz f) m c -> bz f (m c)", f=video_length
        )
        
        b, f, c = context_tokens.shape
        for _ in range(2):
            context_tokens = context_tokens.permute(0, 2, 1)
            if context_tokens.shape[-1] % 2 == 1:
                x_first, x_rest = context_tokens[..., 0], context_tokens[..., 1:]
                if x_rest.shape[-1] > 0:
                    x_rest = self.conv1(x_rest)

                context_tokens = torch.cat([x_first[..., None], x_rest], dim=-1)
                context_tokens = context_tokens.reshape(b, c, context_tokens.shape[-1]).permute(0, 2, 1)
            else:
                context_tokens = self.conv1(context_tokens)
                context_tokens = context_tokens.reshape(b, c, context_tokens.shape[-1]).permute(0, 2, 1)
        
        context_tokens = rearrange(context_tokens, "b f (m c) -> b f m c", m=self.context_tokens) 
        context_tokens = self.norm(context_tokens)       

        return context_tokens
    
class FaceProjModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size=768,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
    ):
        super().__init__()

        self.generator = None
        self.hidden_size = hidden_size
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * hidden_size
        )
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, embeds):
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.hidden_size
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens  
