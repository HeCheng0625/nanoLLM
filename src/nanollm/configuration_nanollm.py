from typing import Optional
from transformers import PretrainedConfig


class NanoLLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NanoLLMModel`].
    It is used to instantiate a NanoLLM model according to the specified arguments,
    defining the model architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control
    the model outputs. Read the documentation from [`PretrainedConfig`] for more information.

    Args:
        vocab_size (`int`, *optional*, defaults to 151936):
            Vocabulary size of the NanoLLM model. Defines the number of different tokens that can be represented
            by the `input_ids` passed to the model.
        hidden_size (`int`, *optional*, defaults to 1024):
            Dimension of the hidden representations.
            for Qwen2.5 0.5B, hidden_size = 896
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimension of the dense MLP representations.
            for Qwen2.5 0.5B, intermediate_size = 4864
        num_hidden_layers (`int`, *optional*, defaults to 28):
            Number of hidden layers in the Transformer decoder.
            for Qwen2.5 0.5B, num_hidden_layers = 24
        num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer.
            for Qwen2.5 0.5B, num_attention_heads = 14
        num_key_value_heads (`int`, *optional*, defaults to 8):
            Number of key/value heads used for Grouped Query Attention (GQA). If equal to
            `num_attention_heads`, standard Multi-Head Attention (MHA) is used. If set to 1,
            Multi-Query Attention (MQA) is used. Otherwise, GQA is used.
            for Qwen2.5 0.5B, num_key_value_heads = 2
        head_dim (`int`, *optional*, defaults to 128):
            Dimension of each attention head.
            for Qwen2.5 0.5B, head_dim = 64
        hidden_act (`str`, *optional*, defaults to `"silu"`):
            The non-linear activation function in the decoder.
        max_position_embeddings (`int`, *optional*, defaults to 40960):
            The maximum sequence length that this model might ever be used with.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated normal initializer for initializing all weight matrices.
        rms_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the RMS normalization layers.
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions.
        tie_word_embeddings (`bool`, *optional*, defaults to `True`):
            Whether the model's input and output word embeddings should be tied.

        attention_type (`str`, *optional*, defaults to `"mha"`):
            Attention variant to use. Supported values: `"mha"`, `"mla"`, `"linear"`,
            `"sparse"`, `"hybrid"`.
        mlp_type (`str`, *optional*, defaults to `"dense"`):
            MLP variant to use. Supported values: `"dense"` or `"moe"`.

        attention_bias (`bool`, *optional*, defaults to `False`):
            Whether to use a bias in the query, key, value and output projection layers
            during self-attention.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            Dropout ratio for the attention probabilities.

        rope_theta (`float`, *optional*, defaults to 1_000_000.0):
            Base period of the RoPE embeddings.
        rope_scaling (`dict`, *optional*):
            Dictionary containing the configuration parameters for RoPE scaling, used to
            extend the context length beyond `max_position_embeddings`.

        use_sliding_window (`bool`, *optional*, defaults to `False`):
            Whether to enable sliding-window attention.
        sliding_window (`int`, *optional*):
            Sliding window attention (SWA) window size.
        max_window_layers (`int`, *optional*, defaults to 28):
            Number of layers using full attention. Layers beyond this will use SWA.

        kv_lora_rank (`int`, *optional*, defaults to 512):
            Low-rank dimension for key/value projection in MLA.
        q_lora_rank (`int`, *optional*, defaults to 1536):
            Low-rank dimension for query projection in MLA.
        qk_rope_head_dim (`int`, *optional*, defaults to 64):
            Head dimension for Q/K to which RoPE is applied in MLA.
        v_head_dim (`int`, *optional*, defaults to 128):
            Head dimension for values in MLA.

        num_experts (`int`, *optional*, defaults to 64):
            Number of experts in MoE layers.
        num_experts_per_tok (`int`, *optional*, defaults to 8):
            Number of experts selected per token.
        moe_intermediate_size (`int`, *optional*):
            Hidden dimension of each expert MLP. If not specified, it will be inferred
            from `intermediate_size`.
        shared_expert_intermediate_size (`int`, *optional*, defaults to 0):
            Hidden dimension of the shared expert MLP.

        use_mhc (`bool`, *optional*, defaults to `False`):
            Whether to enable Multi-Head Hyper-Connections.
        use_engram (`bool`, *optional*, defaults to `False`):
            Whether to enable Engram-style memory.

        # bos_token_id (`int`, *optional*, defaults to 151643):
        #     Beginning-of-sequence token id.
        # eos_token_id (`int`, *optional*, defaults to 151645):
        #     End-of-sequence token id.
        torch_dtype (`str`, *optional*, defaults to `"bfloat16"`):
            Default torch dtype for the model.

    ```python
    >>> from nanollm import NanoLLMConfig
    >>> config = NanoLLMConfig()
    >>> print(config)
    ```
    """

    model_type = "nanollm"

    def __init__(
        self,
        vocab_size: int | None = 151936,
        hidden_size: int | None = 1024,
        intermediate_size: int | None = 3072,
        num_hidden_layers: int | None = 28,
        num_attention_heads: int | None = 16,
        num_key_value_heads: int | None = 8,
        head_dim: int | None = 128,
        hidden_act: str | None = "silu",
        max_position_embeddings: int | None = 40960,
        initializer_range: float | None = 0.02,
        rms_norm_eps: float | None = 1e-6,
        use_cache: bool | None = True,
        tie_word_embeddings: bool | None = True,
        attention_type: str | None = "mha",
        mlp_type: str | None = "dense",
        attention_bias: bool | None = False,
        attention_dropout: float | None = 0.0,
        rope_theta: float | None = 1_000_000.0,
        rope_scaling: Optional[dict] | None = None,
        use_sliding_window: bool | None = False,
        sliding_window: int | None = None,
        max_window_layers: int | None = 28,
        kv_lora_rank: int | None = 512,
        q_lora_rank: int | None = 1536,
        qk_rope_head_dim: int | None = 64,
        v_head_dim: int | None = 128,
        num_experts: int | None = 64,
        num_experts_per_tok: int | None = 8,
        moe_intermediate_size: int | None = None,
        shared_expert_intermediate_size: int | None = 0,
        use_mhc: bool | None = False,
        use_engram: bool | None = False,
        # bos_token_id: int | None = 151643,
        # eos_token_id: int | None = 151645,
        torch_dtype: str | None = "bfloat16",
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.max_position_embeddings = max_position_embeddings

        # GQA
        self.num_key_value_heads = (
            num_key_value_heads
            if num_key_value_heads is not None
            else num_attention_heads
        )

        # Attention
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # RoPE
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        # MLP
        self.hidden_act = hidden_act

        # Init / norm
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.tie_word_embeddings = tie_word_embeddings
        self.torch_dtype = torch_dtype

        # Architecture switches
        self.attention_type = attention_type
        self.mlp_type = mlp_type

        # Sliding window
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers

        # MLA
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim

        # MoE
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.shared_expert_intermediate_size = shared_expert_intermediate_size

        if moe_intermediate_size is None and mlp_type == "moe":
            self.moe_intermediate_size = (
                intermediate_size // num_experts_per_tok
                if intermediate_size is not None and num_experts_per_tok is not None
                else None
            )
        else:
            self.moe_intermediate_size = moe_intermediate_size

        # Experimental
        self.use_mhc = use_mhc
        self.use_engram = use_engram

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
