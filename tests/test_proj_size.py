# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import torch
# from einops import rearrange
# import argparse


# class AudioProjModel(torch.nn.Module):
#     def __init__(
#         self,
#         seq_len=5,
#         blocks=12,  # add a new parameter blocks
#         channels=768,  # add a new parameter channels
#         intermediate_dim=512,
#         output_dim=768,
#         context_tokens=32,
#         dtype=None,
#         device=None,
#     ):
#         factory_kwargs = {"dtype": dtype, "device": device}
#         super().__init__()

#         self.seq_len = seq_len
#         self.blocks = blocks
#         self.channels = channels
#         self.input_dim = seq_len * blocks * channels  # updated input_dim
#         self.intermediate_dim = intermediate_dim
#         self.context_tokens = context_tokens
#         self.output_dim = output_dim

#         # define multiple linear layers
#         self.proj1 = torch.nn.Linear(self.input_dim, intermediate_dim, **factory_kwargs)
#         self.proj2 = torch.nn.Linear(intermediate_dim, intermediate_dim, **factory_kwargs)
#         self.proj3 = torch.nn.Linear(intermediate_dim, context_tokens * output_dim, **factory_kwargs)

#         self.norm = torch.nn.LayerNorm(output_dim)
        
#         self.conv1 = torch.nn.Conv1d(
#             in_channels=context_tokens * output_dim,
#             out_channels=context_tokens * output_dim,
#             kernel_size=2,
#             stride=2,
#             padding=0,
#             **factory_kwargs,
#         )

#     def forward(self, audio_embeds):
#         # merge
#         video_length = audio_embeds.shape[1]
#         audio_embeds = rearrange(audio_embeds, "bz f w b c -> (bz f) w b c")
#         batch_size, window_size, blocks, channels = audio_embeds.shape
#         audio_embeds = audio_embeds.view(batch_size, window_size * blocks * channels)

#         audio_embeds = torch.relu(self.proj1(audio_embeds))
#         audio_embeds = torch.relu(self.proj2(audio_embeds))

#         context_tokens = self.proj3(audio_embeds).reshape(
#             batch_size, self.context_tokens, self.output_dim
#         )

#         # reshape back after projecting
#         context_tokens = rearrange(
#             context_tokens, "(bz f) m c -> bz f (m c)", f=video_length
#         )
        
#         b, f, c = context_tokens.shape
#         for _ in range(2):
#             context_tokens = context_tokens.permute(0, 2, 1)  # (b, c, f)
#             if context_tokens.shape[-1] % 2 == 1:
#                 x_first, x_rest = context_tokens[..., 0], context_tokens[..., 1:]
#                 if x_rest.shape[-1] > 0:
#                     x_rest = self.conv1(x_rest)
#                 context_tokens = torch.cat([x_first[..., None], x_rest], dim=-1)
#                 context_tokens = context_tokens.reshape(b, c, context_tokens.shape[-1]).permute(0, 2, 1)
#             else:
#                 context_tokens = self.conv1(context_tokens)
#                 context_tokens = context_tokens.reshape(b, c, context_tokens.shape[-1]).permute(0, 2, 1)
        
#         context_tokens = rearrange(context_tokens, "b f (m c) -> b f m c", m=self.context_tokens) 
#         context_tokens = self.norm(context_tokens)

#         return context_tokens


# def calculate_params(
#     seq_len=5,
#     blocks=12,
#     channels=768,
#     intermediate_dim=512,
#     output_dim=768,
#     context_tokens=32,
#     dtype=None,
#     device=None
# ):
#     """
#     实例化模型并计算可训练参数量
#     """
#     model = AudioProjModel(
#         seq_len=seq_len,
#         blocks=blocks,
#         channels=channels,
#         intermediate_dim=intermediate_dim,
#         output_dim=output_dim,
#         context_tokens=context_tokens,
#         dtype=dtype,
#         device=device,
#     )
#     total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return total_params


# def main():
#     # parser = argparse.ArgumentParser(description="Calculate the number of parameters for AudioProjModel")
#     # parser.add_argument("--seq_len", type=int, default=5, help="Sequence length")
#     # parser.add_argument("--blocks", type=int, default=12, help="Number of blocks")
#     # parser.add_argument("--channels", type=int, default=768, help="Number of channels")
#     # parser.add_argument("--intermediate_dim", type=int, default=512, help="Intermediate dimension")
#     # parser.add_argument("--output_dim", type=int, default=768, help="Output dimension")
#     # parser.add_argument("--context_tokens", type=int, default=32, help="Number of context tokens")
#     # args = parser.parse_args()

#     total_params = calculate_params(
#         seq_len=5,
#         blocks=12,
#         channels=768,
#         intermediate_dim=512,
#         output_dim=768,
#         context_tokens=32,
#     )
#     print(f"Model parameter count with the given configuration: {total_params}")


# if __name__ == "__main__":
#     main()

import torch
from torch import nn
from einops import rearrange

class AudioProjModel(nn.Module):
    def __init__(
        self,
        seq_len=5,
        blocks=12,
        channels=768,
        intermediate_dim=512,
        output_dim=768,
        context_tokens=32,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()
        
        self.seq_len = seq_len
        self.blocks = blocks
        self.channels = channels
        self.input_dim = seq_len * blocks * channels
        self.intermediate_dim = intermediate_dim
        self.context_tokens = context_tokens
        self.output_dim = output_dim

        self.proj1 = nn.Linear(self.input_dim, intermediate_dim, **factory_kwargs)
        self.proj2 = nn.Linear(intermediate_dim, intermediate_dim, **factory_kwargs)
        self.proj3 = nn.Linear(intermediate_dim, context_tokens * output_dim, **factory_kwargs)
        self.norm = nn.LayerNorm(output_dim)
        self.conv1 = nn.Conv1d(
            in_channels=context_tokens * output_dim,
            out_channels=context_tokens * output_dim,
            kernel_size=2,
            stride=2,
            padding=0,
            **factory_kwargs
        )

    def forward(self, audio_embeds):
        # Original forward implementation here...
        pass

class FaceProjModel(torch.nn.Module):
    def __init__(
        self,
        hidden_size=768,
        clip_embeddings_dim=512,
        clip_extra_context_tokens=4,
        dtype=None,
        device=None,
    ):
        factory_kwargs = {"dtype": dtype, "device": device}
        super().__init__()

        self.generator = None
        self.hidden_size = hidden_size
        self.clip_extra_context_tokens = clip_extra_context_tokens
        self.proj = torch.nn.Linear(
            clip_embeddings_dim, self.clip_extra_context_tokens * hidden_size, **factory_kwargs
        )
        self.norm = torch.nn.LayerNorm(hidden_size, **factory_kwargs)

    def forward(self, embeds):
        clip_extra_context_tokens = self.proj(embeds).reshape(
            -1, self.clip_extra_context_tokens, self.hidden_size
        )
        clip_extra_context_tokens = self.norm(clip_extra_context_tokens)
        return clip_extra_context_tokens  
        
def calculate_parameters(model, **kwargs):
    """计算指定配置下的模型参数量"""
    model = model(**kwargs)

    # 计算 fp16 精度下仅存储参数时的显存占用，以 MB 为单位：
    # 一个参数 16 bit = 2 字节
    total_params = sum(p.numel() for p in model.parameters())
    fp16_memory_bytes = total_params * 2
    fp16_memory_mb = fp16_memory_bytes / (1024 ** 2)
    return total_params, fp16_memory_mb

if __name__ == "__main__":
    # 示例用法
    base_config = {
        'seq_len': 5,
        'blocks': 12,
        'channels': 768,
        'intermediate_dim': 512,
        'output_dim': 3072,
        'context_tokens': 2
    }

    # 测试不同配置
    configs = [
        ("默认配置", base_config),
        # ("减少blocks", {**base_config, 'blocks': 6}),
        # ("减少channels", {**base_config, 'channels': 384}),
        # ("增大中间层", {**base_config, 'intermediate_dim': 1024}),
    ]

    for name, config in configs:
        params, fp16_memory_mb = calculate_parameters(AudioProjModel, **config)
        print(f"{name}: {params/1e6:.2f}M, fp16_memory_mb: {fp16_memory_mb:.2f}MB")

    face_config = {
        'hidden_size': 768,
        'clip_embeddings_dim': 512,
        'clip_extra_context_tokens': 4
    }
    
    params, fp16_memory_mb = calculate_parameters(FaceProjModel, **face_config)
    print(f"FaceProjModel: {params/1e6:.2f}M, fp16_memory_mb: {fp16_memory_mb:.2f}MB")
