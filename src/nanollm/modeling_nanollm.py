import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Tuple
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin

from .configuration_nanollm import NanoLLMConfig
from .components.norms import NanoRMSNorm
from .components.mlp import NanoMLP
from .components.attention.mha import NanoAttention
from .components.rotary import NanoRotaryEmbedding


class NanoLLMDecoderLayer(nn.Module):
    def __init__(self, config: NanoLLMConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        # 1. Attention Block
        self.input_layernorm = NanoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Switch Logic: Future proofing for MLA / Linear etc.
        if config.attention_type == "mha":
            self.self_attn = NanoAttention(config, layer_idx=layer_idx)
        else:
            raise NotImplementedError(
                f"Attention {config.attention_type} not implemented yet"
            )

        # 2. MLP Block
        self.post_attention_layernorm = NanoRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        if config.mlp_type == "dense":
            self.mlp = NanoMLP(config)
        else:
            raise NotImplementedError(f"MLP {config.mlp_type} not implemented yet")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:

        # TODO: Implement Residual Connection Structure
        # 1. Pre-Norm: norm_hidden = input_layernorm(hidden_states)
        # 2. Attention: attn_out, present_kv = self_attn(norm_hidden, ...)
        # 3. Residual 1: hidden_states = hidden_states + attn_out

        # 4. Pre-Norm: norm_hidden = post_attention_layernorm(hidden_states)
        # 5. MLP: mlp_out = mlp(norm_hidden)
        # 6. Residual 2: hidden_states = hidden_states + mlp_out

        # Self Attention
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            cache_position=cache_position,
        )
        hidden_states = hidden_states + residual

        # FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


class NanoLLMPreTrainedModel(PreTrainedModel):
    config: NanoLLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["NanoLLMDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_flash_attn_2 = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": NanoLLMDecoderLayer,
        "attentions": NanoAttention,
    }


class NanoLLMModel(NanoLLMPreTrainedModel):
    """
    Transformer Body (Encoder). No LM Head.
    """

    def __init__(self, config: NanoLLMConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                NanoLLMDecoderLayer(config, layer_idx=i)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = NanoRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.rotary_emb = NanoRotaryEmbedding(config)

        self.gradient_checkpointing = False

        self.post_init()  # Initialize weights

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs,
    ) -> BaseModelOutputWithPast | Tuple[torch.Tensor, ...]:

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if use_cache and past_key_values is None:
            try:
                past_key_values = DynamicCache(config=self.config)
            except:
                past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeddings = self.rotary_emb(
            hidden_states, position_ids
        )  # (cos, sin): ([batch, seq_len, head_dim], [batch, seq_len, head_dim])

        # create causal mask
        causal_mask = self._update_causal_mask(
            attention_mask=attention_mask,
            input_tensor=hidden_states,
            cache_position=cache_position,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )

        for decoder_layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states=hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

        # final norm
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor | None,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ) -> torch.Tensor | None:
        """
        Convert 2D attention_mask (batch, seq) to 4D Causal Mask (batch, 1, seq, seq+past)
        Used for Eager Attention and SDPA (when using math/mem_efficient kernels)
        """

        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            else:
                return None

        if self.config._attn_implementation == "sdpa":
            if attention_mask is None or 0.0 not in attention_mask:
                return None

        batch_size, seq_len = input_tensor.shape[0], input_tensor.shape[1]
        dtype, device = input_tensor.dtype, input_tensor.device

        if past_key_values is not None:
            past_len = past_key_values.get_seq_length()
        else:
            past_len = 0

        target_length = past_len + seq_len

        min_dtype = torch.finfo(dtype).min

        causal_mask = torch.full(
            (seq_len, target_length), fill_value=min_dtype, dtype=dtype, device=device
        )
        if seq_len != 1:
            causal_mask = torch.triu(causal_mask, diagonal=past_len + 1)
        else:
            causal_mask = torch.zeros(
                (seq_len, target_length), dtype=dtype, device=device
            )

        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

        if attention_mask is not None:
            if attention_mask.shape[1] < target_length:
                padding = torch.ones(
                    (batch_size, target_length - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=device,
                )
                attention_mask = torch.cat([padding, attention_mask], dim=1)

            expanded_mask = attention_mask[:, None, None, :].expand(
                batch_size, 1, seq_len, -1
            )
            padding_mask = (1.0 - expanded_mask.to(dtype)) * min_dtype

            causal_mask = causal_mask + padding_mask

            # # bool padding positions: True means "masked"
            # pad_mask = attention_mask[:, None, None, :].expand(batch_size, 1, seq_len, -1) == 0

            # # Instead of adding a huge negative number, just set masked positions to min_dtype
            # causal_mask = causal_mask.masked_fill(pad_mask, min_dtype)

        return causal_mask.contiguous()


class NanoLLMForCausalLM(NanoLLMPreTrainedModel, GenerationMixin):
    """
    End-to-End Model with LM Head.
    """

    _tied_weights_keys = {
        "lm_head.weight": "model.embed_tokens.weight"
    }  # Ensure embedding tying works

    def __init__(self, config: NanoLLMConfig):
        super().__init__(config)
        self.model = NanoLLMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        # 1. Forward body
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
        )
        hidden_states = outputs.last_hidden_state

        # 2. Forward head
        logits = self.lm_head(hidden_states)  # [batch, seq_len, vocab_size]

        # 3. Calculate Loss (if labels provided)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
