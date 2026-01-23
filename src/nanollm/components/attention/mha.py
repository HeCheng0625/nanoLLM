import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from ...configuration_nanollm import NanoLLMConfig
from ..rotary import apply_rotary_pos_emb
from ..norms import NanoRMSNorm

from transformers.cache_utils import Cache, DynamicCache
from transformers.utils.import_utils import is_torch_greater_or_equal

_is_torch_greater_or_equal_than_2_5 = is_torch_greater_or_equal("2.5", accept_dev=True)

# def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
#     return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def use_gqa_in_sdpa(attention_mask: torch.Tensor | None) -> bool:
    # GQA can only be used under the following conditions
    # cuda
    #   - torch version >= 2.5
    #   - attention_mask is None (otherwise it will fall back to the math kernel)
    return _is_torch_greater_or_equal_than_2_5 and attention_mask is None


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    if n_rep == 1:
        return hidden_states
    bsz, n_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = (
        hidden_states[:, :, None, :, :]
        .expand(bsz, n_kv_heads, n_rep, seq_len, head_dim)
        .contiguous()
        .reshape(bsz, n_kv_heads * n_rep, seq_len, head_dim)
    )
    return hidden_states


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    key_states = repeat_kv(
        key, module.num_key_value_groups
    )  # [bsz, n_heads, seq_len, head_dim]
    value_states = repeat_kv(
        value, module.num_key_value_groups
    )  # [bsz, n_heads, seq_len, head_dim]

    attn_weights = (
        torch.matmul(query, key_states.transpose(2, 3)) * scaling
    )  # [bsz, n_heads, seq_len, seq_len]

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    # else:
    #     # ONLY FOR DEBUG
    #     bsz, n_heads, q_len, _ = query.shape
    #     k_len = key_states.shape[2]
    #     causal_mask = torch.tril(
    #         torch.ones((q_len, k_len), device=query.device, dtype=torch.bool)
    #     ).view(1, 1, q_len, k_len)
    #     attn_weights = attn_weights.masked_fill(~causal_mask, float("-inf"))

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)

    attn_output = torch.matmul(
        attn_weights, value_states
    )  # [bsz, n_heads, seq_len, head_dim]
    attn_output = attn_output.transpose(
        1, 2
    ).contiguous()  # [bsz, seq_len, n_heads, head_dim]

    return attn_output, attn_weights


def flash_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    is_causal: bool = True,
) -> torch.Tensor:
    """
    Wrapper for flash_attn_func to handle shape transformations and checks.
    Input shapes: [batch, heads, seq, dim] (Standard PyTorch/HF format)
    Output shape: [batch, seq, heads, dim] (Flash Attention native format)
    """
    try:
        from flash_attn import flash_attn_func
    except ImportError:
        raise ImportError(
            "Flash Attention 2 is not installed. Please install it to use 'flash_attention_2'."
        )

    batch_size, num_heads, q_len, head_dim = query.shape

    # Flash Attention expects inputs in [Batch, Seq, Heads, Dim] (Channel Last)
    q = query.transpose(1, 2)
    k = key.transpose(1, 2)
    v = value.transpose(1, 2)

    # Logic for Causal Masking in Flash Attn
    # Standard FA2 assumes if 'causal=True', it applies the lower-triangular mask.
    # It does NOT accept an explicit `attention_mask` tensor for causal LM training (usually).
    # If `attention_mask` is provided (e.g. for padding), it requires the varlen API which is complex.
    # For nanoLLM pretraining (usually dense batches without padding), we simplify:

    # 1. If masking is causal AND no padding mask provided -> Use optimized causal kernel
    is_causal_kernel = is_causal and (attention_mask is None) and (q_len > 1)

    # 2. If an explicit mask is provided, FA2 basic kernel doesn't support it directly.
    # We warn the user or fallback. For now, we assume standard pretraining flow.
    if attention_mask is not None:
        # TODO: Implement varlen/padding support if needed later.
        raise NotImplementedError(
            "Flash Attention 2 does not support explicit attention masks."
        )

    attn_output = flash_attn_func(
        q,
        k,
        v,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=is_causal_kernel,
    )
    # FA2 returns [Batch, Seq, Heads, Dim], ready for reshape
    return attn_output


def sdpa_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    dropout: float = 0.0,
    is_causal: bool = True,
) -> torch.Tensor:
    sdpa_kwargs = {}
    if use_gqa_in_sdpa(attention_mask):
        sdpa_kwargs["enable_gqa"] = True
    else:
        key = repeat_kv(key, module.num_key_value_groups)
        value = repeat_kv(value, module.num_key_value_groups)

    # 1. attention_mask is None 2. len(query) > 1, 3. is_causal is True
    # query shape is [bsz, n_heads, q_len, head_dim]
    causal_mask = query.shape[2] > 1 and attention_mask is None and is_causal

    attn_output = F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attention_mask,
        is_causal=causal_mask,
        dropout_p=dropout if module.training else 0.0,
        **sdpa_kwargs,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


class NanoAttention(nn.Module):
    def __init__(self, config: NanoLLMConfig, layer_idx: int = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Dimensions
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = (
            config.head_dim
            if config.head_dim is not None
            else self.hidden_size // self.num_heads
        )
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        # TODO: Define Linear layers (q_proj, k_proj, v_proj, o_proj)
        self.q_proj = nn.Linear(
            self.hidden_size, self.head_dim * self.num_heads, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.head_dim * self.num_key_value_heads,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            self.head_dim * self.num_heads, self.hidden_size, bias=config.attention_bias
        )

        self.q_norm = NanoRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = NanoRMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: Cache | None = None,
        cache_position: torch.LongTensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Standard Multi-Head / Grouped-Query Attention forward pass.
        """
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (
            *input_shape,
            -1,
            self.head_dim,
        )  # [bsz, seq_len, hidden_size] -> [bsz, seq_len, num_heads, head_dim]

        # 1. Projections (Q, K, V)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)

        # 2. Reshape for heads: [bsz, seq_len, num_heads, head_dim] -> [bsz, num_heads, seq_len, head_dim]
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        # 3. Apply RoPE (using apply_rotary_pos_emb)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin
        )

        # 4. KV Cache handling (if past_key_value is not None)
        if past_key_values is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        # 5. Repeat KV heads for GQA (if num_kv_groups > 1)
        # move to attention forward

        # 6. Scaled Dot Product Attention (use F.scaled_dot_product_attention for efficiency)
        attn_weights = None
        if self.config._attn_implementation == "eager":
            attn_output, attn_weights = eager_attention_forward(
                self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        elif self.config._attn_implementation == "flash_attention_2":
            attn_output = flash_attention_forward(
                self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            # sdpa
            attn_output = sdpa_attention_forward(
                self,
                query=query_states,
                key=key_states,
                value=value_states,
                attention_mask=attention_mask,
                scaling=self.scaling,
                dropout=self.attention_dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )

        # 7. Reshape back and Output Projection
        attn_output = attn_output.reshape(
            *input_shape, -1
        ).contiguous()  # [bsz, seq_len, n_heads, head_dim] -> [bsz, seq_len, hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
