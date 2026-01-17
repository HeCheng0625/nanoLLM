import torch
import torch.nn as nn
from typing import Tuple, Optional
from ..configuration_nanollm import NanoLLMConfig


class NanoRotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: NanoLLMConfig,
        device=None,
    ):
        super().__init__()
        # TODO:
        # 1. Precompute 'inv_freq' based on dim and base
        # 2. Register 'inv_freq' as buffers (non-learnable)
        base = config.rope_theta
        dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Input position_ids: [batch, seq_len]
        """
        inv_freq_expanded = (
            self.inv_freq[None, :, None]
            .expand(position_ids.shape[0], -1, 1)
            .to(x.device)
        )  # [batch, head_dim//2, 1]
        position_ids_expanded = position_ids[:, None, :].float()  # [batch, 1, seq_len]
        device_type = x.device.type
        with torch.autocast(device_type=device_type, dtype=torch.float32):
            freqs = (
                inv_freq_expanded.float() @ position_ids_expanded.float()
            ).transpose(
                1, 2
            )  # [batch, seq_len, head_dim//2]
            emb = torch.cat((freqs, freqs), dim=-1)  # [batch, seq_len, head_dim]
            cos = torch.cos(emb)
            sin = torch.sin(emb)
        return cos.to(x.dtype), sin.to(x.dtype)


class NanoRotaryEmbeddingCached(nn.Module):
    def __init__(self, config: NanoLLMConfig, device=None):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.max_position_embeddings = config.max_position_embeddings

        # 1. Compute inv_freq (kept fully consistent with the dynamic version)
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # 2. Precompute cache (build cache)
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, device=device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        # t: [0, 1, ..., seq_len-1]
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        # Outer product: [seq_len, dim/2]
        # Equivalent to inv_freq @ position_ids in the dynamic version
        freqs = torch.outer(t, self.inv_freq)

        # Concat: [seq_len, dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        # Store as buffers
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    @torch.no_grad()
    def forward(
        self, x: torch.Tensor, position_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use table lookup (indexing) to get cos/sin, avoiding repeated matrix multiplications.
        """
        # 1. Auto-expand cache (when inference length exceeds the preset)
        seq_len = position_ids.max().item() + 1
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=int(seq_len * 1.5), device=x.device, dtype=x.dtype
            )

        # 2. Lookup table (advanced indexing)
        # cos_cached: [max_seq, dim]
        # position_ids: [batch, seq]
        # result: [batch, seq, dim] -> matches the dynamic output shape
        cos = self.cos_cached[position_ids].to(x.dtype)
        sin = self.sin_cached[position_ids].to(x.dtype)

        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to Query and Key states.
    Input q, k: [batch, num_heads, seq_len, head_dim]
    cos, sin: [batch, seq_len, head_dim]
    """
    # TODO: Implement standard RoPE rotation
    # x_rotated = (x * cos) + (rotate_half(x) * sin)
    cos = cos.unsqueeze(dim=1)
    sin = sin.unsqueeze(dim=1)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed
