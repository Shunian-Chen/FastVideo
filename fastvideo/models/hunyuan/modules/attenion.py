import torch
import torch.nn.functional as F

from fastvideo.models.flash_attn_no_pad import flash_attn_no_pad
from fastvideo.utils.communications import all_gather, all_to_all_4D
from fastvideo.utils.parallel_states import (get_sequence_parallel_state,
                                             nccl_info)


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


def parallel_attention(q, k, v, img_q_len, img_kv_len, text_mask):
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

        def shrink_head(encoder_state, dim):
            local_heads = encoder_state.shape[dim] // nccl_info.sp_size
            return encoder_state.narrow(
                dim, nccl_info.rank_within_group * local_heads, local_heads)

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

    attn_mask = F.pad(text_mask, (sequence_length, 0), value=True)
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

class CrossAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        heads_num: int,
        qk_norm: bool = True,
        qk_norm_type: str = "rms",
        attention_dropout: float = 0.1,
        output_dropout: float = 0.1,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        
        if hidden_size % heads_num != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by heads_num ({heads_num})"
            )
        
        self.hidden_size = hidden_size
        self.num_attention_heads = heads_num
        self.hidden_size_per_attention_head = hidden_size // heads_num
        
        # Projections
        self.query = nn.Linear(hidden_size, hidden_size, bias=True, **factory_kwargs)
        self.key_value = nn.Linear(hidden_size, 2 * hidden_size, bias=True, **factory_kwargs)
        self.dense = nn.Linear(hidden_size, hidden_size, bias=True, **factory_kwargs)
        
        # Normalization
        qk_norm_layer = get_norm_layer(qk_norm_type)
        norm_args = {
            "eps": 1e-6,
            "elementwise_affine": True,
            **factory_kwargs
        }
        self.q_norm = qk_norm_layer(self.hidden_size_per_attention_head, **norm_args) if qk_norm else nn.Identity()
        self.k_norm = qk_norm_layer(self.hidden_size_per_attention_head, **norm_args) if qk_norm else nn.Identity()
        
        # Dropout
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output_dropout = nn.Dropout(output_dropout)
        
        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.query.weight)
        nn.init.xavier_uniform_(self.key_value.weight)
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.zeros_(self.query.bias)
        nn.init.zeros_(self.key_value.bias)
        nn.init.zeros_(self.dense.bias)

    def _transpose_for_scores(self, tensor):
        new_shape = tensor.size()[:-1] + (self.num_attention_heads, self.hidden_size_per_attention_head)
        return tensor.view(new_shape).permute(0, 2, 1, 3)

    def forward(self, hidden_states, encoder_outputs):
        # Query projection
        query = self._transpose_for_scores(self.query(hidden_states))
        query = self.q_norm(query)
        
        # Key-Value projection
        kv = self.key_value(encoder_outputs)
        key, value = torch.chunk(kv, 2, dim=-1)
        key = self._transpose_for_scores(key)
        value = self._transpose_for_scores(value)
        key = self.k_norm(key)
        
        # Scaled dot-product attention
        scale = math.sqrt(self.hidden_size_per_attention_head)
        context = scaled_dot_product_attention(
            query, key, value,
            dropout_p=self.attention_dropout.p if self.training else 0.0,
            scale=1.0/scale
        )
        
        # Output projection
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(context.size()[:-2] + (self.hidden_size,))
        output = self.output_dropout(self.dense(context))
        
        return output
