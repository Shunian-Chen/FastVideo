# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

from typing import Any, Tuple

import torch
import torch.distributed as dist
from torch import Tensor

from fastvideo.utils.parallel_states import nccl_info
from loguru import logger
import sys
## 不显示info级别的日志
logger.remove()
logger.add(sys.stdout, level="WARNING")

def broadcast(input_: torch.Tensor):
    src = nccl_info.group_id * nccl_info.sp_size
    dist.broadcast(input_, src=src, group=nccl_info.group)


def _all_to_all_4D(input: torch.tensor,
                   scatter_idx: int = 2,
                   gather_idx: int = 1,
                   group=None) -> torch.tensor:
    """
    all-to-all for QKV

    Args:
        input (torch.tensor): a tensor sharded along dim scatter dim
        scatter_idx (int): default 1
        gather_idx (int): default 2
        group : torch process group

    Returns:
        torch.tensor: resharded tensor (bs, seqlen/P, hc, hs)
    """
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    seq_world_size = dist.get_world_size(group)

    if scatter_idx == 2 and gather_idx == 1:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen/P, hc, hs) output: (bs, seqlen, hc/P, hs)
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen/P, hc, hs) -reshape-> (bs, seq_len/P, P, hc/P, hs) -transpose(0,2)-> (P, seq_len/P, bs, hc/P, hs)
        input_t = (input.reshape(bs, shard_seqlen, seq_world_size, shard_hc,
                                 hs).transpose(0, 2).contiguous())

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, seq_len/P, bs, hc/P, hs) scatter seqlen -all2all-> (P, seq_len/P, bs, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t
        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(seqlen, bs, shard_hc, hs)

        # (seq_len, bs, hc/P, hs) -reshape-> (bs, seq_len, hc/P, hs)
        output = output.transpose(0, 1).contiguous().reshape(
            bs, seqlen, shard_hc, hs)

        return output

    elif scatter_idx == 1 and gather_idx == 2:
        # input (torch.tensor): a tensor sharded along dim 1 (bs, seqlen, hc/P, hs) output: (bs, seqlen/P, hc, hs)
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size
        seq_world_size = dist.get_world_size(group)

        # transpose groups of heads with the seq-len parallel dimension, so that we can scatter them!
        # (bs, seqlen, hc/P, hs) -reshape-> (bs, P, seq_len/P, hc/P, hs) -transpose(0, 3)-> (hc/P, P, seqlen/P, bs, hs) -transpose(0, 1) -> (P, hc/P, seqlen/P, bs, hs)
        input_t = (input.reshape(
            bs, seq_world_size, shard_seqlen, shard_hc,
            hs).transpose(0, 3).transpose(0, 1).contiguous().reshape(
                seq_world_size, shard_hc, shard_seqlen, bs, hs))

        output = torch.empty_like(input_t)
        # https://pytorch.org/docs/stable/distributed.html#torch.distributed.all_to_all_single
        # (P, bs x hc/P, seqlen/P, hs) scatter seqlen -all2all-> (P, bs x seq_len/P, hc/P, hs) scatter head
        if seq_world_size > 1:
            dist.all_to_all_single(output, input_t, group=group)
            torch.cuda.synchronize()
        else:
            output = input_t

        # if scattering the seq-dim, transpose the heads back to the original dimension
        output = output.reshape(hc, shard_seqlen, bs, hs)

        # (hc, seqlen/N, bs, hs) -tranpose(0,2)-> (bs, seqlen/N, hc, hs)
        output = output.transpose(0, 2).contiguous().reshape(
            bs, shard_seqlen, hc, hs)

        return output
    else:
        raise RuntimeError(
            "scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")


class SeqAllToAll4D(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        input: Tensor,
        scatter_idx: int,
        gather_idx: int,
    ) -> Tensor:
        ctx.group = group
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx

        return _all_to_all_4D(input, scatter_idx, gather_idx, group=group)

    @staticmethod
    def backward(ctx: Any,
                 *grad_output: Tensor) -> Tuple[None, Tensor, None, None]:
        return (
            None,
            SeqAllToAll4D.apply(ctx.group, *grad_output, ctx.gather_idx,
                                ctx.scatter_idx),
            None,
            None,
        )


def all_to_all_4D(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return SeqAllToAll4D.apply(nccl_info.group, input_, scatter_dim,
                               gather_dim)


def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [
        t.contiguous()
        for t in torch.tensor_split(input_, world_size, scatter_dim)
    ]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group,
                             scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )


def all_to_all(
    input_: torch.Tensor,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, nccl_info.group, scatter_dim, gather_dim)


class _AllGather(torch.autograd.Function):
    """All-gather communication with autograd support.

    Args:
        input_: input tensor
        dim: dimension along which to concatenate
    """

    @staticmethod
    def forward(ctx, input_, dim):
        ctx.dim = dim
        world_size = nccl_info.sp_size
        group = nccl_info.group
        input_size = list(input_.size())

        ctx.input_size = input_size[dim]

        tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
        input_ = input_.contiguous()
        dist.all_gather(tensor_list, input_, group=group)

        output = torch.cat(tensor_list, dim=dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        world_size = nccl_info.sp_size
        rank = nccl_info.rank_within_group
        dim = ctx.dim
        input_size = ctx.input_size

        sizes = [input_size] * world_size

        grad_input_list = torch.split(grad_output, sizes, dim=dim)
        grad_input = grad_input_list[rank]

        return grad_input, None


def all_gather(input_: torch.Tensor, dim: int = 1):
    """Performs an all-gather operation on the input tensor along the specified dimension.

    Args:
        input_ (torch.Tensor): Input tensor of shape [B, H, S, D].
        dim (int, optional): Dimension along which to concatenate. Defaults to 1.

    Returns:
        torch.Tensor: Output tensor after all-gather operation, concatenated along 'dim'.
    """
    return _AllGather.apply(input_, dim)


def prepare_sequence_parallel_data(hidden_states, encoder_hidden_states,
                                   attention_mask, encoder_attention_mask):
    if nccl_info.sp_size == 1:
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        )

    def prepare(hidden_states, encoder_hidden_states, attention_mask,
                encoder_attention_mask):
        hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
        encoder_hidden_states = all_to_all(encoder_hidden_states,
                                           scatter_dim=1,
                                           gather_dim=0)
        attention_mask = all_to_all(attention_mask,
                                    scatter_dim=1,
                                    gather_dim=0)
        encoder_attention_mask = all_to_all(encoder_attention_mask,
                                            scatter_dim=1,
                                            gather_dim=0)
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
        )

    sp_size = nccl_info.sp_size
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    (
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
    ) = prepare(
        hidden_states,
        encoder_hidden_states.repeat(1, sp_size, 1),
        attention_mask.repeat(1, sp_size, 1, 1),
        encoder_attention_mask.repeat(1, sp_size),
    )

    logger.info(f"Inside prepare_sequence_parallel_data, after prepare, latent shape: {hidden_states.shape}")
    logger.info(f"Inside prepare_sequence_parallel_data, after prepare, cond shape: {encoder_hidden_states.shape}")
    logger.info(f"Inside prepare_sequence_parallel_data, after prepare, attn_mask shape: {attention_mask.shape}")
    logger.info(f"Inside prepare_sequence_parallel_data, after prepare, cond_mask shape: {encoder_attention_mask.shape}")

    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask


def sp_parallel_dataloader_wrapper(dataloader, device, train_batch_size,
                                   sp_size, train_sp_batch_size):
    while True:
        for data_item in dataloader:
            latents, cond, attn_mask, cond_mask = data_item
            logger.info(f"Inside sp_parallel_dataloader_wrapper, latents shape: {latents.shape}")
            logger.info(f"Inside sp_parallel_dataloader_wrapper, cond shape: {cond.shape}")
            logger.info(f"Inside sp_parallel_dataloader_wrapper, attn_mask shape: {attn_mask.shape}")
            logger.info(f"Inside sp_parallel_dataloader_wrapper, cond_mask shape: {cond_mask.shape}")

            latents = latents.to(device)
            cond = cond.to(device)
            attn_mask = attn_mask.to(device)
            cond_mask = cond_mask.to(device)
            frame = latents.shape[2]
            if frame == 1:
                yield latents, cond, attn_mask, cond_mask
            else:
                latents, cond, attn_mask, cond_mask = prepare_sequence_parallel_data(
                    latents, cond, attn_mask, cond_mask)
                assert (
                    train_batch_size * sp_size >= train_sp_batch_size
                ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size //
                                  train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    encoder_hidden_states = cond[st_idx:ed_idx]
                    attention_mask = attn_mask[st_idx:ed_idx]
                    encoder_attention_mask = cond_mask[st_idx:ed_idx]
                    yield (
                        latents[st_idx:ed_idx],
                        encoder_hidden_states,
                        attention_mask,
                        encoder_attention_mask,
                    )


def prepare_sequence_parallel_data_audio(hidden_states, encoder_hidden_states,
                                   attention_mask, encoder_attention_mask, audio_emb, face_emb):
    if nccl_info.sp_size == 1:
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
            audio_emb,
            face_emb,
        )

    def prepare(hidden_states, encoder_hidden_states, attention_mask,
                encoder_attention_mask, audio_emb, face_emb):
        hidden_states = all_to_all(hidden_states, scatter_dim=2, gather_dim=0)
        encoder_hidden_states = all_to_all(encoder_hidden_states,
                                           scatter_dim=1,
                                           gather_dim=0)
        attention_mask = all_to_all(attention_mask,
                                    scatter_dim=1,
                                    gather_dim=0)
        encoder_attention_mask = all_to_all(encoder_attention_mask,
                                            scatter_dim=1,
                                            gather_dim=0)
        audio_emb = all_to_all(audio_emb,
                              scatter_dim=1,
                              gather_dim=0)
        face_emb = all_to_all(face_emb,
                              scatter_dim=1,
                              gather_dim=0)
        
        
        return (
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
            audio_emb,
            face_emb,
        )


    sp_size = nccl_info.sp_size
    frame = hidden_states.shape[2]
    assert frame % sp_size == 0, "frame should be a multiple of sp_size"

    # 如果audio_frame不是sp_size的倍数，则需要进行padding，audio_emb的维度是(bs, frame, w, b, d)
    audio_frame = audio_emb.shape[1]
    if audio_frame % nccl_info.sp_size != 0:
        remainder = audio_frame % nccl_info.sp_size
        padding = nccl_info.sp_size - remainder
        # 构造需要补齐的零张量
        pad_shape = list(audio_emb.shape)
        pad_shape[1] = padding
        zeros_pad = torch.zeros(pad_shape, dtype=audio_emb.dtype, device=audio_emb.device)
        # 拼接到原张量上完成padding
        audio_emb = torch.cat([audio_emb, zeros_pad], dim=1)


    (
        hidden_states,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
        audio_emb,
        face_emb,
    ) = prepare(
        hidden_states,
        encoder_hidden_states.repeat(1, sp_size, 1),
        attention_mask.repeat(1, sp_size, 1, 1),
        encoder_attention_mask.repeat(1, sp_size),
        audio_emb,
        face_emb.repeat(1, sp_size),
    )

    logger.info(f"Inside prepare_sequence_parallel_data_audio, after prepare, latent shape: {hidden_states.shape}")
    logger.info(f"Inside prepare_sequence_parallel_data_audio, after prepare, cond shape: {encoder_hidden_states.shape}")
    logger.info(f"Inside prepare_sequence_parallel_data_audio, after prepare, attn_mask shape: {attention_mask.shape}")
    logger.info(f"Inside prepare_sequence_parallel_data_audio, after prepare, cond_mask shape: {encoder_attention_mask.shape}")
    logger.info(f"Inside prepare_sequence_parallel_data_audio, after prepare, audio_emb shape: {audio_emb.shape}")
    logger.info(f"Inside prepare_sequence_parallel_data_audio, after prepare, face_emb shape: {face_emb.shape}")

    return hidden_states, encoder_hidden_states, attention_mask, encoder_attention_mask, audio_emb, face_emb

def sp_parallel_dataloader_wrapper_audio(dataloader, device, train_batch_size,
                                   sp_size, train_sp_batch_size):
    while True:
        for data_item in dataloader:
            latents, cond, attn_mask, cond_mask, audio_emb, face_emb, audio_embed_file = data_item
            logger.info(f"Inside sp_parallel_dataloader_wrapper, latents shape: {latents.shape}")
            logger.info(f"Inside sp_parallel_dataloader_wrapper, cond shape: {cond.shape}")
            logger.info(f"Inside sp_parallel_dataloader_wrapper, attn_mask shape: {attn_mask.shape}")
            logger.info(f"Inside sp_parallel_dataloader_wrapper, cond_mask shape: {cond_mask.shape}")
            logger.info(f"Inside sp_parallel_dataloader_wrapper, audio_emb shape: {audio_emb.shape}")
            logger.info(f"Inside sp_parallel_dataloader_wrapper, face_emb shape: {face_emb.shape}")
            latents = latents.to(device)
            cond = cond.to(device)
            attn_mask = attn_mask.to(device)
            cond_mask = cond_mask.to(device)
            # 确保audio_emb和face_emb是tensor类型并移到正确设备
            if audio_emb is not None:
                if not isinstance(audio_emb, torch.Tensor):
                    audio_emb = torch.as_tensor(audio_emb)
                audio_emb = audio_emb.to(device)
            if face_emb is not None:
                if not isinstance(face_emb, torch.Tensor):
                    face_emb = torch.as_tensor(face_emb)
                face_emb = face_emb.to(device)
            frame = latents.shape[2]
            if frame == 1:
                yield latents, cond, attn_mask, cond_mask, audio_emb, face_emb
            else:
                latents, cond, attn_mask, cond_mask, audio_emb, face_emb = prepare_sequence_parallel_data_audio(
                    latents, cond, attn_mask, cond_mask, audio_emb, face_emb)
                assert (
                    train_batch_size * sp_size >= train_sp_batch_size
                ), "train_batch_size * sp_size should be greater than train_sp_batch_size"
                for iter in range(train_batch_size * sp_size //
                                  train_sp_batch_size):
                    st_idx = iter * train_sp_batch_size
                    ed_idx = (iter + 1) * train_sp_batch_size
                    encoder_hidden_states = cond[st_idx:ed_idx]
                    attention_mask = attn_mask[st_idx:ed_idx]
                    encoder_attention_mask = cond_mask[st_idx:ed_idx]
                    audio_emb = audio_emb[st_idx:ed_idx]
                    face_emb = face_emb[st_idx:ed_idx]
                    yield (
                        latents[st_idx:ed_idx],
                        encoder_hidden_states,
                        attention_mask,
                        encoder_attention_mask,
                        audio_emb,
                        face_emb,
                        audio_embed_file,
                    )