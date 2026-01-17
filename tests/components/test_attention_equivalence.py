import unittest
import torch
import sys
import os

# Ensure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from nanollm.configuration_nanollm import NanoLLMConfig
from nanollm.components.attention.mha import NanoAttention
from nanollm.components.rotary import NanoRotaryEmbedding

# Check for Flash Attention availability
import flash_attn

try:
    import flash_attn

    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False
print(f"HAS_FLASH_ATTN: {HAS_FLASH_ATTN}")


class TestAttentionEquivalence(unittest.TestCase):
    def setUp(self):
        # 1. Environment Setup
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available, skipping attention equivalence tests.")

        self.device = "cuda"
        # FA2 requires fp16 or bf16. bf16 is more stable if available.
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"\n[Setup] Device: {self.device}, Dtype: {self.dtype}")

        # 2. Config (Qwen2-like Small)
        self.config = NanoLLMConfig(
            hidden_size=256,
            num_attention_heads=8,  # Head Dim = 32
            num_key_value_heads=8,  # GQA 2:1
            attention_dropout=0.0,  # Disable dropout for deterministic check
            max_position_embeddings=512,
            rms_norm_eps=1e-6,
            attention_bias=False,
        )

        # 3. Prepare Common Inputs
        self.batch_size = 2
        self.seq_len = 64
        self.hidden_states = torch.randn(
            self.batch_size,
            self.seq_len,
            self.config.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )

        # 4. Prepare RoPE (Shared)
        self.rotary = NanoRotaryEmbedding(self.config, device=self.device)
        self.position_ids = (
            torch.arange(self.seq_len, device=self.device)
            .unsqueeze(0)
            .expand(self.batch_size, -1)
        )
        # Get cos, sin
        self.cos, self.sin = self.rotary(self.hidden_states, self.position_ids)
        self.pos_emb = (self.cos, self.sin)

        # 5. Create a "Master" State Dict (Random Weights)
        # Initialize a temporary model to generate weights
        temp_model = NanoAttention(self.config).to(self.device).to(self.dtype)
        self.state_dict = temp_model.state_dict()

    def _create_model(self, impl: str):
        """Helper to create a model with specific implementation and loaded weights."""
        # Modify config on the fly (copying config is safer in real apps, but ok here)
        self.config._attn_implementation = impl

        model = NanoAttention(self.config).to(self.device).to(self.dtype)
        model.load_state_dict(self.state_dict)  # Load shared weights
        model.eval()  # Deterministic mode
        return model

    # def test_sdpa_backends_availability(self):
    #     """
    #     Probe which SDPA backends are available and working for the current input.
    #     This forces specific kernels to ensure the shapes/dtypes support Flash Attention.
    #     """
    #     print("\n--- Testing SDPA Backend Availability ---")

    #     # 必须把模式设为 sdpa
    #     model = self._create_model("sdpa")

    #     # PyTorch 提供的上下文管理器，用于强制指定后端
    #     from torch.backends.cuda import sdp_kernel

    #     # 1. 尝试强制使用 Flash Attention
    #     # 要求: GPU 是 Ampere (3090/A100) 或更新, dtype 是 fp16/bf16
    #     print(f"Attempting Flash Attention (enable_flash=True, others=False)...")
    #     try:
    #         with sdp_kernel(
    #             enable_flash=True, enable_math=False, enable_mem_efficient=False
    #         ):
    #             _ = model(self.hidden_states, self.pos_emb, attention_mask=None)
    #         print("✅ Flash Attention Backend: SUCCESS")
    #     except RuntimeError as e:
    #         print(f"❌ Flash Attention Backend: FAILED. Reason: {e}")
    #         # 注意：在 T4/V100 上失败是正常的，因为它们不支持原生 FA
    #         if torch.cuda.get_device_capability()[0] < 8:
    #             print("   (Note: Your GPU architecture is too old for Flash Attention)")

    #     # 2. 尝试强制使用 Memory Efficient Attention (Cutlass)
    #     # 要求: 几乎所有现代 GPU 都支持，通常是 Flash 的备选
    #     print(f"Attempting Memory Efficient Attention...")
    #     try:
    #         with sdp_kernel(
    #             enable_flash=False, enable_math=False, enable_mem_efficient=True
    #         ):
    #             _ = model(self.hidden_states, self.pos_emb, attention_mask=None)
    #         print("✅ Mem Efficient Backend: SUCCESS")
    #     except RuntimeError as e:
    #         print(f"❌ Mem Efficient Backend: FAILED. Reason: {e}")

    #     # 3. 尝试强制使用 Math (C++ Reference)
    #     # 要求: 永远可用，但最慢，显存占用最高
    #     print(f"Attempting Math Backend...")
    #     try:
    #         with sdp_kernel(
    #             enable_flash=False, enable_math=True, enable_mem_efficient=False
    #         ):
    #             _ = model(self.hidden_states, self.pos_emb, attention_mask=None)
    #         print("✅ Math Backend: SUCCESS")
    #     except RuntimeError as e:
    #         print(f"❌ Math Backend: FAILED. Reason: {e}")

    def test_sdpa_vs_eager(self):
        """Compare PyTorch SDPA against Manual Eager implementation."""
        print("\n--- Testing SDPA vs Eager ---")

        # 1. Run Eager
        model_eager = self._create_model("eager")
        with torch.no_grad():
            out_eager, _ = model_eager(
                self.hidden_states,
                self.pos_emb,
                attention_mask=None,  # None implies Causal
            )

        # 2. Run SDPA
        model_sdpa = self._create_model("sdpa")
        with torch.no_grad():
            out_sdpa, _ = model_sdpa(
                self.hidden_states, self.pos_emb, attention_mask=None
            )

        # 3. Compare
        # BF16 precision is tricky, we allow larger tolerance
        max_diff = (out_eager - out_sdpa).abs().max().item()
        # mean_diff = (out_eager - out_sdpa).mean().item()
        print(f"Max Difference (SDPA vs Eager): {max_diff:.6f}")
        # print(f"Mean Difference (SDPA vs Eager): {mean_diff:.6f}")

        # Tolerance: 1e-2 for BF16 is usually acceptable for rigorous accumulation diffs
        self.assertTrue(torch.allclose(out_eager, out_sdpa, atol=2e-2, rtol=1e-2))

    def test_fa2_vs_eager(self):
        """Compare Flash Attention 2 against Manual Eager implementation."""
        print("\n--- Testing FA2 vs Eager ---")

        if not HAS_FLASH_ATTN:
            print("Flash Attention 2 not installed. Skipping.")
            return

        # 1. Run Eager
        model_eager = self._create_model("eager")

        with torch.no_grad():
            out_eager, _ = model_eager(
                self.hidden_states, self.pos_emb, attention_mask=None
            )

        # 2. Run FA2
        model_fa2 = self._create_model("flash_attention_2")
        with torch.no_grad():
            out_fa2, _ = model_fa2(
                self.hidden_states, self.pos_emb, attention_mask=None
            )

        # 3. Compare
        max_diff = (out_eager - out_fa2).abs().max().item()
        # mean_diff = (out_eager - out_fa2).mean().item()
        print(f"Max Difference (FA2 vs Eager): {max_diff:.6f}")
        # print(f"Mean Difference (FA2 vs Eager): {mean_diff:.6f}")

        # FA2 uses different arithmetic order, so diffs are expected in bf16
        self.assertTrue(torch.allclose(out_eager, out_fa2, atol=2e-2, rtol=1e-2))

    def test_fa2_vs_sdpa(self):
        """Compare Flash Attention 2 against SDPA."""
        print("\n--- Testing FA2 vs SDPA ---")

        if not HAS_FLASH_ATTN:
            print("Flash Attention 2 not installed. Skipping.")
            return

        # 1. Run SDPA
        model_sdpa = self._create_model("sdpa")
        with torch.no_grad():
            out_sdpa, _ = model_sdpa(
                self.hidden_states, self.pos_emb, attention_mask=None
            )

        # 2. Run FA2
        model_fa2 = self._create_model("flash_attention_2")
        with torch.no_grad():
            out_fa2, _ = model_fa2(
                self.hidden_states, self.pos_emb, attention_mask=None
            )

        # 3. Compare
        max_diff = (out_sdpa - out_fa2).abs().max().item()
        # mean_diff = (out_sdpa - out_fa2).mean().item()
        print(f"Max Difference (FA2 vs SDPA): {max_diff:.6f}")
        # print(f"Mean Difference (FA2 vs SDPA): {mean_diff:.6f}")

        # FA2 uses different arithmetic order, so diffs are expected in bf16
        self.assertTrue(torch.allclose(out_sdpa, out_fa2, atol=2e-2, rtol=1e-2))

    # def test_gqa_logic(self):
    #     """Specific check to ensure GQA logic (repeat_kv vs broadcasting) holds."""
    #     print("\n--- Testing GQA Logic (Eager vs SDPA) ---")

    #     # Config is already GQA (8 heads vs 4 kv heads)

    #     model_eager = self._create_model("eager")  # Uses manual repeat_kv
    #     model_sdpa = self._create_model(
    #         "sdpa"
    #     )  # Uses broadcasting (if torch>=2.5) or repeat

    #     out_eager, _ = model_eager(self.hidden_states, self.pos_emb, None)
    #     out_sdpa, _ = model_sdpa(self.hidden_states, self.pos_emb, None)

    #     self.assertTrue(torch.allclose(out_eager, out_sdpa, atol=2e-2))
    #     print("GQA consistent across implementations.")


if __name__ == "__main__":
    unittest.main()
