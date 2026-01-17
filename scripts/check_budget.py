import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from nanollm.configuration_nanollm import NanoLLMConfig
from nanollm.utils.flops import print_model_stats

print("=== Qwen3 0.6B (Dense) ===")
conf_base = NanoLLMConfig(
    hidden_size=1024,
    intermediate_size=3072,
    num_hidden_layers=28,
    num_attention_heads=16,
    num_key_value_heads=8,
    head_dim=128,
    mlp_type="dense",
)
print_model_stats(conf_base)

print("=== Qwen2.5 0.5B (Dense) ===")
conf_base = NanoLLMConfig(
    hidden_size=896,
    intermediate_size=4864,
    num_hidden_layers=24,
    num_attention_heads=14,
    num_key_value_heads=2,
    head_dim=64,
    mlp_type="dense",
)
print_model_stats(conf_base)

# print("\n=== MoE Variant (Targeting same FLOPs) ===")
# # Adjust moe_intermediate_size until GFLOPs match baseline
# conf_moe = NanoLLMConfig(
#     hidden_size=896,
#     mlp_type="moe",
#     num_experts=64,
#     num_experts_per_tok=6,
#     moe_intermediate_size=1400,  # Tweak this number!
# )
# print_model_stats(conf_moe)
