from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..configuration_nanollm import NanoLLMConfig


@dataclass(frozen=True)
class FlopsBreakdown:
    """FLOPs breakdown for one forward pass."""

    per_token: float  # FLOPs / token (avg over positions in seq)
    per_sequence: float  # FLOPs / sequence (batch_size=1)
    per_batch: float  # FLOPs / batch (batch_size as provided)
    attn_per_token_per_layer: float  # Attention FLOPs / token / layer
    mlp_per_token_per_layer: float  # MLP (dense or active-MoE) FLOPs / token / layer
    logits_per_token: float  # Logits projection FLOPs / token (if included)


def estimate_flops(
    config: NanoLLMConfig,
    seq_len: int = 2048,
    batch_size: int = 1,
    include_logits: bool = True,
) -> FlopsBreakdown:
    """
    Estimate FLOPs for one forward pass.

    Notes on accounting:
    - We count FLOPs as 2 * MACs for matmul/linear layers.
    - Attention mechanism cost is approximated for causal attention by assuming
      average context length = seq_len / 2. This yields:
        QK^T: 2 * (seq_len/2) * total_head_dim
        AV:   2 * (seq_len/2) * total_head_dim
      => total ≈ 2 * seq_len * total_head_dim FLOPs per token (per layer)
    - This function returns a breakdown and includes seq_len & batch_size scaling.
    """

    if seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    hidden = config.hidden_size
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    n_kv_heads = (
        config.num_key_value_heads
        if config.num_key_value_heads is not None
        else n_heads
    )

    if hidden is None or n_layers is None or n_heads is None:
        raise ValueError(
            "config.hidden_size, config.num_hidden_layers, config.num_attention_heads must not be None"
        )

    head_dim = (hidden // n_heads) if config.head_dim is None else config.head_dim
    if head_dim is None:
        raise ValueError(
            "head_dim could not be inferred (config.head_dim is None and hidden/heads invalid)"
        )

    total_q_dim = n_heads * head_dim
    total_kv_dim = n_kv_heads * head_dim

    # -------------------------
    # 1) Attention FLOPs / token / layer
    # -------------------------
    if config.attention_type == "mha":
        # Projections:
        # Q: [hidden -> total_q_dim]
        # K: [hidden -> total_kv_dim]
        # V: [hidden -> total_kv_dim]
        proj_flops = 2 * hidden * (total_q_dim + 2 * total_kv_dim)

        # Output projection: [total_q_dim -> hidden]
        out_flops = 2 * total_q_dim * hidden

        # Mechanism (QK + AV), causal avg context = seq_len/2
        mech_flops = 2 * seq_len * total_q_dim

        attn_per_token_per_layer = float(proj_flops + out_flops + mech_flops)

    elif config.attention_type == "mla":
        # Placeholder: MLA implementation-specific. Do NOT silently return 0.
        # Keep it explicit so experiments don't accidentally compare wrong FLOPs.
        raise NotImplementedError(
            "FLOPs for attention_type='mla' is not implemented yet. "
            "Add an MLA-specific estimator to avoid incorrect fixed-FLOPs comparisons."
        )
    elif config.attention_type in ("linear", "sparse", "sliding_window", "hybrid"):
        raise NotImplementedError(
            f"FLOPs for attention_type='{config.attention_type}' is not implemented yet. "
            "Implement it instead of defaulting to 0."
        )
    else:
        raise ValueError(f"Unknown attention_type: {config.attention_type}")

    # -------------------------
    # 2) MLP / MoE FLOPs / token / layer
    # -------------------------
    if config.mlp_type == "dense":
        if config.intermediate_size is None:
            raise ValueError("config.intermediate_size must not be None for dense MLP")
        # SwiGLU:
        # gate: hidden -> inter
        # up:   hidden -> inter
        # down: inter  -> hidden
        # FLOPs = 2*hidden*inter + 2*hidden*inter + 2*inter*hidden = 6*hidden*inter
        mlp_per_token_per_layer = float(6 * hidden * config.intermediate_size)

    elif config.mlp_type == "moe":
        if config.num_experts is None or config.num_experts_per_tok is None:
            raise ValueError(
                "config.num_experts and config.num_experts_per_tok must not be None for MoE"
            )

        # Router logits: hidden -> num_experts
        router_flops = float(2 * hidden * config.num_experts)

        # Expert MLP size (None-safe fallback)
        expert_size = config.moe_intermediate_size
        if expert_size is None:
            if config.intermediate_size is None:
                raise ValueError(
                    "config.moe_intermediate_size is None and config.intermediate_size is None; cannot infer expert size"
                )
            expert_size = config.intermediate_size // config.num_experts_per_tok

        # Active experts only (top-k)
        active_k = config.num_experts_per_tok
        experts_flops = float(active_k * (6 * hidden * expert_size))

        # Optional shared expert
        shared_size = config.shared_expert_intermediate_size or 0
        shared_flops = float(6 * hidden * shared_size) if shared_size > 0 else 0.0

        mlp_per_token_per_layer = router_flops + experts_flops + shared_flops

    else:
        raise ValueError(f"Unknown mlp_type: {config.mlp_type}")

    # -------------------------
    # 3) Logits FLOPs / token
    # -------------------------
    logits_per_token = 0.0
    if include_logits:
        if config.vocab_size is None:
            raise ValueError(
                "config.vocab_size must not be None when include_logits=True"
            )
        logits_per_token = float(2 * hidden * config.vocab_size)

    # -------------------------
    # Total FLOPs
    # -------------------------
    per_token = float(
        n_layers * (attn_per_token_per_layer + mlp_per_token_per_layer)
        + logits_per_token
    )
    per_sequence = per_token * float(seq_len)
    per_batch = per_sequence * float(batch_size)

    return FlopsBreakdown(
        per_token=per_token,
        per_sequence=per_sequence,
        per_batch=per_batch,
        attn_per_token_per_layer=attn_per_token_per_layer,
        mlp_per_token_per_layer=mlp_per_token_per_layer,
        logits_per_token=logits_per_token,
    )


def print_model_stats(
    config: NanoLLMConfig,
    seq_len: int = 2048,
    batch_size: int = 1,
    include_logits: bool = True,
) -> None:
    stats = estimate_flops(
        config, seq_len=seq_len, batch_size=batch_size, include_logits=include_logits
    )

    print(
        f"Architecture: {getattr(config, 'model_type', 'nanollm')} | "
        f"Attn: {config.attention_type} | MLP: {config.mlp_type} | "
        f"seq_len={seq_len} | batch={batch_size}"
    )
    print(f"FLOPs/token: {stats.per_token / 1e9:.3f} GFLOPs")
    print(f"FLOPs/sequence: {stats.per_sequence / 1e12:.3f} TFLOPs")
    print(f"FLOPs/batch: {stats.per_batch / 1e12:.3f} TFLOPs")

    print(
        f"  Attn/token/layer: {stats.attn_per_token_per_layer / 1e9:.3f} GFLOPs | "
        f"MLP/token/layer: {stats.mlp_per_token_per_layer / 1e9:.3f} GFLOPs"
        + (
            f" | Logits/token: {stats.logits_per_token / 1e9:.3f} GFLOPs"
            if include_logits
            else ""
        )
    )

    if config.mlp_type == "moe":
        print(f"  Active Experts: {config.num_experts_per_tok}/{config.num_experts}")
