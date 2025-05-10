import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional
from torch.nn.functional import scaled_dot_product_attention
from .norm_layers import get_norm_layer


from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import (get_sequence_parallel_state,nccl_info)
import flash_attn
from flash_attn.flash_attn_interface import _flash_attn_forward
from flash_attn.flash_attn_interface import flash_attn_varlen_func

MEMORY_LAYOUT = {
    "flash": (
        lambda x: x.view(x.shape[0] * x.shape[1], *x.shape[2:]),
        lambda x: x,
    ),
    "torch": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
    "vanilla": (
        lambda x: x.transpose(1, 2),
        lambda x: x.transpose(1, 2),
    ),
}

def attention_flash(
    q,
    k,
    v,
    mode="flash",
    drop_rate=0,
    attn_mask=None,
    causal=False,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    batch_size=1,
):
    """
    Perform QKV self attention.

    Args:
        q (torch.Tensor): Query tensor with shape [b, s, a, d], where a is the number of heads.
        k (torch.Tensor): Key tensor with shape [b, s1, a, d]
        v (torch.Tensor): Value tensor with shape [b, s1, a, d]
        mode (str): Attention mode. Choose from 'self_flash', 'cross_flash', 'torch', and 'vanilla'.
        drop_rate (float): Dropout rate in attention map. (default: 0)
        attn_mask (torch.Tensor): Attention mask with shape [b, s1] (cross_attn), or [b, a, s, s1] (torch or vanilla).
            (default: None)
        causal (bool): Whether to use causal attention. (default: False)
        cu_seqlens_q (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into q.
        cu_seqlens_kv (torch.Tensor): dtype torch.int32. The cumulative sequence lengths of the sequences in the batch,
            used to index into kv.
        max_seqlen_q (int): The maximum sequence length in the batch of q.
        max_seqlen_kv (int): The maximum sequence length in the batch of k and v.

    Returns:
        torch.Tensor: Output tensor after self attention with shape [b, s, ad]
    """
    pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
    q = pre_attn_layout(q)
    k = pre_attn_layout(k)
    v = pre_attn_layout(v)

    if mode == "torch":
        if attn_mask is not None and attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q.dtype)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop_rate, is_causal=causal
        )
    elif mode == "flash":
        x = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_kv,
            max_seqlen_q,
            max_seqlen_kv,
        )
        # x with shape [(bxs), a, d]
        x = x.view(
            batch_size, max_seqlen_q, x.shape[-2], x.shape[-1]
        )  # reshape x to [b, s, a, d]
    elif mode == "vanilla":
        scale_factor = 1 / math.sqrt(q.size(-1))

        b, a, s, _ = q.shape
        s1 = k.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q.dtype, device=q.device)
        if causal:
            # Only applied to self attention
            assert (
                attn_mask is None
            ), "Causal mask and attn_mask cannot be used together"
            temp_mask = torch.ones(b, a, s, s, dtype=torch.bool, device=q.device).tril(
                diagonal=0
            )
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(q.dtype)
        
        ##########################################  attention mask ##########################################
        if attn_mask is not None:
            mask_bool = (attn_mask > 0.2).bool()
            if s != s1:
                raise ValueError(f"vanilla require equal length, got s_q={s}, s_k={s1}")
            row_mask = mask_bool.unsqueeze(-1)
            col_mask = mask_bool.unsqueeze(-2)
            keep = row_mask | col_mask
            attn_bias.masked_fill_(~keep, float("-inf"))
        ##########################################  attention mask ##########################################

        # TODO: Maybe force q and k to be float32 to avoid numerical overflow
        attn = (q @ k.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        x = attn @ v
    else:
        raise NotImplementedError(f"Unsupported attention mode: {mode}")

    x = post_attn_layout(x)
    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out                                          

def get_cu_seqlens(text_mask, img_len):
    """Calculate cu_seqlens_q, cu_seqlens_kv using text_mask and img_len

    Args:
        text_mask (torch.Tensor): the mask of text
        img_len (int): the length of image

    Returns:
        torch.Tensor: the calculated cu_seqlens for flash attention
    """
    batch_size = text_mask.shape[0]
    text_len = text_mask.sum(dim=1)
    max_len = text_mask.shape[1] + img_len

    cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device="cuda")

    for i in range(batch_size):
        s = text_len[i] + img_len
        s1 = i * max_len + s
        s2 = (i + 1) * max_len
        cu_seqlens[2 * i + 1] = s1
        cu_seqlens[2 * i + 2] = s2

    return cu_seqlens

def attention(
    q,
    k,
    v,
    drop_rate=0,
    attn_mask=None,
    causal=False,
):

    qkv = torch.stack([q, k, v], dim=2)

    if attn_mask is not None and attn_mask.dtype != torch.bool:
        attn_mask = attn_mask.bool()

    x = flash_attn_no_pad(qkv,
                          attn_mask,
                          causal=causal,
                          dropout_p=drop_rate,
                          softmax_scale=None)

    b, s, a, d = x.shape
    out = x.reshape(b, s, -1)
    return out

def shrink_head(encoder_state, dim):
    local_heads = encoder_state.shape[dim] // nccl_info.sp_size
    return encoder_state.narrow(
        dim, nccl_info.rank_within_group * local_heads, local_heads)

def parallel_attention(q, k, v, img_q_len, img_kv_len, text_mask = None):
    # 1GPU torch.Size([1, 11264, 24, 128]) tensor([    0, 11275, 11520], device='cuda:0', dtype=torch.int32)
    # 2GPU torch.Size([1, 5632, 24, 128]) tensor([   0, 5643, 5888], device='cuda:0', dtype=torch.int32)
    query, encoder_query = q
    key, encoder_key = k
    value, encoder_value = v
    if get_sequence_parallel_state():
        # batch_size, seq_len, attn_heads, head_dim
        query = all_to_all_4D(query, scatter_dim=2, gather_dim=1)
        key = all_to_all_4D(key, scatter_dim=2, gather_dim=1)
        value = all_to_all_4D(value, scatter_dim=2, gather_dim=1)



        encoder_query = shrink_head(encoder_query, dim=2)
        encoder_key = shrink_head(encoder_key, dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)
        # [b, s, h, d]

    sequence_length = query.size(1)
    encoder_sequence_length = encoder_query.size(1)

    # Hint: please check encoder_query.shape
    query = torch.cat([query, encoder_query], dim=1)
    key = torch.cat([key, encoder_key], dim=1)
    value = torch.cat([value, encoder_value], dim=1)
    # B, S, 3, H, D
    qkv = torch.stack([query, key, value], dim=2)

    if text_mask is not None:
        attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
    else:
        attn_mask = None
    hidden_states = flash_attn_no_pad(qkv,
                                      attn_mask,
                                      causal=False,
                                      dropout_p=0.0,
                                      softmax_scale=None)

    hidden_states, encoder_hidden_states = hidden_states.split_with_sizes(
        (sequence_length, encoder_sequence_length), dim=1)
    if get_sequence_parallel_state():
        hidden_states = all_to_all_4D(hidden_states,
                                      scatter_dim=1,
                                      gather_dim=2)
        encoder_hidden_states = all_gather(encoder_hidden_states,
                                           dim=2).contiguous()
    hidden_states = hidden_states.to(query.dtype)
    encoder_hidden_states = encoder_hidden_states.to(query.dtype)

    attn = torch.cat([hidden_states, encoder_hidden_states], dim=1)

    b, s, a, d = attn.shape
    attn = attn.reshape(b, s, -1)

    return attn

def cross_attention(q, k, v, text_mask=None):
    """
    实现一个与 parallel_attention 输入形式相似的 cross attention。
    
    参数：
      q, k, v: 各自是 (decoder_proj, encoder_proj) 形式的元组，即：
               q = (decoder_query, encoder_query)
               k = (decoder_key,   encoder_key)
               v = (decoder_value, encoder_value)
      text_mask: [batch_size, enc_seq_len] 的可选mask，用来屏蔽encoder端某些token的注意力。

    返回：
      cross_out: [batch_size, dec_seq_len, num_heads * head_dim]
                 表示每个decoder token在encoder上做注意力后的输出。
    """
    # 解包
    # 对于 cross attention，一般只需要 decoder_query 做 Q，encoder_key/value 做 K/V
    decoder_query, _ = q
    _, encoder_key = k
    _, encoder_value = v

    bsz, dec_seq_len, nheads, head_dim = decoder_query.shape
    _, enc_seq_len, _, _ = encoder_key.shape

    # 如果开启了 sequence parallel，需要先对 Q/K/V 做相应的 all_to_all / shrink_head
    if get_sequence_parallel_state():
        decoder_query = all_to_all_4D(decoder_query, scatter_dim=2, gather_dim=1)
        # encoder_key   = all_to_all_4D(encoder_key,   scatter_dim=2, gather_dim=1)
        # encoder_value = all_to_all_4D(encoder_value, scatter_dim=2, gather_dim=1)

        # 视具体实现决定对哪些进行 shrink_head，这里假设都需要拆分
        # decoder_query = shrink_head(decoder_query, dim=2)
        encoder_key   = shrink_head(encoder_key,   dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    dec_seqlen = decoder_query.shape[1]
    enc_seqlen = encoder_key.shape[1]

    # 现在，将 decoder_query 放在 sequence 的前 dec_seq_len，
    # 将 encoder_key/value 放在后 enc_seq_len。这样可以用一次flash_attn来完成。
    # 整体的总序列长度 = dec_seq_len + enc_seq_len
    q_cat = torch.cat([decoder_query, torch.zeros_like(encoder_key)], dim=1)  # Q只占前 dec_seq_len
    k_cat = torch.cat([torch.zeros_like(decoder_query), encoder_key], dim=1)  # K只占后 enc_seq_len
    v_cat = torch.cat([torch.zeros_like(decoder_query), encoder_value], dim=1) # V只占后 enc_seq_len

    
    # qkv形状: [batch_size, dec_seq_len + enc_seq_len, 3, num_heads, head_dim]
    qkv = torch.stack([q_cat, k_cat, v_cat], dim=2)

    # 构造 attention mask，确保decoder部分的token只能看见encoder的部分
    # text_mask 预期: [batch_size, enc_seq_len], True表示要被mask的地方
    # 最终 attn_mask 形状也应是 [batch_size, dec_seq_len + enc_seq_len]
    # 在其中，decoder部分不能看到 decoder 部分 => mask出前 dec_seq_len
    # 同时，如果 text_mask 不为空，需要对 encoder 部分应用。
    with torch.no_grad():
        if text_mask is not None:
            # text_mask是针对 encoder 长度的，这里要pad到 dec_seq_len + enc_seq_len
            # decoder的前 dec_seq_len 全部 masked(因为不允许自注意)，encoder部分的mask由 text_mask决定
            pad_decoder = torch.ones((bsz, dec_seq_len), dtype=torch.bool, device=decoder_query.device)
            attn_mask = torch.cat([pad_decoder, text_mask], dim=1)
        else:
            # 如果没有传 text_mask，仍需要对decoder的前 dec_seq_len部分做mask，用True表示不允许关注
            pad_decoder = torch.ones((bsz, dec_seq_len), dtype=torch.bool, device=decoder_query.device)
            pad_encoder = torch.zeros((bsz, enc_seq_len), dtype=torch.bool, device=decoder_query.device)
            attn_mask = torch.cat([pad_decoder, pad_encoder], dim=1)

    # 经过flash attention后,
    # 输出形状: [batch_size, dec_seq_len + enc_seq_len, num_heads, head_dim]
    out = flash_attn_no_pad(qkv,
                            attn_mask,
                            causal=False,
                            dropout_p=0.0,
                            softmax_scale=None)

    # cross attention只关心前 dec_seq_len 对应 Q 的部分输出
    cross_out, _ = out.split_with_sizes([dec_seqlen, enc_seqlen], dim=1)

    # 如果使用了 sequence parallel，需要还原
    if get_sequence_parallel_state():
        # 这里其实只需要处理 cross_out
        cross_out = all_to_all_4D(cross_out,
                                  scatter_dim=1,
                                  gather_dim=2).to(decoder_query.dtype)

    # 拼回 [batch_size, dec_seq_len, num_heads * head_dim]
    cross_out = cross_out.reshape(bsz, dec_seq_len, nheads * head_dim)
    return cross_out


##########################################  attention mask ########################################## 
def cross_attention_face_mask(q, k, v, mode="flash", drop_rate =0, face_mask=None):
    """
    实现一个与 parallel_attention 输入形式相似的 cross attention。
    
    参数：
      q, k, v: 各自是 (decoder_proj, encoder_proj) 形式的元组，即：
               q = (decoder_query, encoder_query)
               k = (decoder_key,   encoder_key)
               v = (decoder_value, encoder_value)
      attn_mask: [batch_size, enc_seq_len] 的可选mask，用来屏蔽encoder端某些token的注意力。

    返回：
      cross_out: [batch_size, dec_seq_len, num_heads * head_dim]
                 表示每个decoder token在encoder上做注意力后的输出。
    """
    # 解包
    # 对于 cross attention，一般只需要 decoder_query 做 Q，encoder_key/value 做 K/V
    decoder_query, _ = q
    _, encoder_key = k
    _, encoder_value = v

    bsz, dec_seq_len, nheads, head_dim = decoder_query.shape
    # print("decoder_query.shape", decoder_query.shape)
    # print("encoder_key.shape", encoder_key.shape)

    _, enc_seq_len, _, _ = encoder_key.shape

    # 如果开启了 sequence parallel，需要先对 Q/K/V 做相应的 all_to_all / shrink_head
    if get_sequence_parallel_state():
        decoder_query = all_to_all_4D(decoder_query, scatter_dim=2, gather_dim=1)
        # encoder_key   = all_to_all_4D(encoder_key,   scatter_dim=2, gather_dim=1)
        # encoder_value = all_to_all_4D(encoder_value, scatter_dim=2, gather_dim=1)

        # 视具体实现决定对哪些进行 shrink_head，这里假设都需要拆分
        # decoder_query = shrink_head(decoder_query, dim=2)
        encoder_key   = shrink_head(encoder_key,   dim=2)
        encoder_value = shrink_head(encoder_value, dim=2)

    dec_seqlen = decoder_query.shape[1]
    enc_seqlen = encoder_key.shape[1]

    # 现在，将 decoder_query 放在 sequence 的前 dec_seq_len，
    # 将 encoder_key/value 放在后 enc_seq_len。这样可以用一次flash_attn来完成。
    # 整体的总序列长度 = dec_seq_len + enc_seq_len
    q_cat = torch.cat([decoder_query, torch.zeros_like(encoder_key)], dim=1)  # Q只占前 dec_seq_len
    k_cat = torch.cat([torch.zeros_like(decoder_query), encoder_key], dim=1)  # K只占后 enc_seq_len
    v_cat = torch.cat([torch.zeros_like(decoder_query), encoder_value], dim=1) # V只占后 enc_seq_len
    # print("torch.cat([decoder_query, torch.zeros_like(encoder_key)], dim=1)",q_cat.shape)
    
    # qkv形状: [batch_size, dec_seq_len + enc_seq_len, 3, num_heads, head_dim]
    qkv = torch.stack([q_cat, k_cat, v_cat], dim=2)

    # 构造 attention mask，确保decoder部分的token只能看见encoder的部分
    # text_mask 预期: [batch_size, enc_seq_len], True表示要被mask的地方
    # 最终 attn_mask 形状也应是 [batch_size, dec_seq_len + enc_seq_len]
    # 在其中，decoder部分不能看到 decoder 部分 => mask出前 dec_seq_len
    # 同时，如果 text_mask 不为空，需要对 encoder 部分应用。
    with torch.no_grad():
        if face_mask is not None:
            pad_encoder = torch.ones((bsz, enc_seq_len), dtype=torch.bool, device=decoder_query.device)
            face_mask = (face_mask > 0.3).to(torch.bool)
            attn_mask = torch.cat([face_mask, pad_encoder], dim=1)
            # pad_decoder = torch.ones((bsz, dec_seq_len), dtype=torch.bool, device=decoder_query.device)
            # attn_mask = torch.cat([pad_decoder, pad_encoder], dim=1)
        else:
            # 如果没有传 text_mask，仍需要对decoder的前 dec_seq_len部分做mask，用True表示不允许关注
            pad_decoder = torch.ones((bsz, dec_seq_len), dtype=torch.bool, device=decoder_query.device)
            pad_encoder = torch.zeros((bsz, enc_seq_len), dtype=torch.bool, device=decoder_query.device)
            attn_mask = torch.cat([pad_decoder, pad_encoder], dim=1)

    num_true  = attn_mask.sum().item()                   # True 的个数
    num_false = (~attn_mask).sum().item()                # False 的个数

    if mode == "flash":
        out = flash_attn_no_pad(qkv,
                                attn_mask,
                                causal=False,
                                dropout_p=0.0,
                                softmax_scale=None)

        # out_cpu = out.detach().cpu()
        # print("out stats → max, min, mean", torch.max(out_cpu).item(), torch.min(out_cpu).item(), torch.mean(out_cpu).item())

    elif mode == "vanilla":
        pre_attn_layout, post_attn_layout = MEMORY_LAYOUT[mode]
        # print("q_cat before pre_attn_layout",q_cat.shape)
        q_cat = pre_attn_layout(q_cat)
        k_cat = pre_attn_layout(k_cat)
        v_cat = pre_attn_layout(v_cat)

        scale_factor = 1 / math.sqrt(q_cat.size(-1))

        # print("q_cat after pre_attn_layout",q_cat.shape)
        # print("attn_mask",attn_mask.shape)
        b, a, s, _ = q_cat.shape
        s1 = k_cat.size(2)
        attn_bias = torch.zeros(b, a, s, s1, dtype=q_cat.dtype, device=q_cat.device)

        if face_mask is not None:
            mask_bool = (attn_mask > 0.3).bool()
            if s != s1:
                raise ValueError(f"vanilla require equal length, got s_q={s}, s_k={s1}")

            row_mask = mask_bool.unsqueeze(-1)
            col_mask = mask_bool.unsqueeze(-2)
            keep = row_mask | col_mask
            keep = keep.unsqueeze(1) 
            attn_bias.masked_fill_(~keep, float("-inf"))

        attn = (q_cat @ k_cat.transpose(-2, -1)) * scale_factor
        attn += attn_bias
        attn = attn.softmax(dim=-1)
        attn = torch.dropout(attn, p=drop_rate, train=True)
        out = attn @ v_cat
        out = post_attn_layout(out)
        b, s, a, d = out.shape
        out = out.reshape(b, s, -1)

    # cross attention只关心前 dec_seq_len 对应 Q 的部分输出
    cross_out, _ = out.split_with_sizes([dec_seqlen, enc_seqlen], dim=1)

    # 如果使用了 sequence parallel，需要还原
    if get_sequence_parallel_state():
        # 这里其实只需要处理 cross_out
        cross_out = all_to_all_4D(cross_out,
                                  scatter_dim=1,
                                  gather_dim=2).to(decoder_query.dtype)

    # 拼回 [batch_size, dec_seq_len, num_heads * head_dim]
    cross_out = cross_out.reshape(bsz, dec_seq_len, nheads * head_dim)
    return cross_out
##########################################  attention mask ##########################################