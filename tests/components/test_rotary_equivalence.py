import unittest
import torch
import sys
import os

# 路径设置
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from nanollm.configuration_nanollm import NanoLLMConfig

# 假设你把上面两个类都放在了 nanollm.components.rotary 里
# 如果没放进去，你可以把上面的类定义直接贴在这个文件里测试
from nanollm.components.rotary import NanoRotaryEmbedding  # 你的动态版

# form nanollm.components.rotary import NanoRotaryEmbeddingCached # 你的缓存版 (如果已保存)


# 为了测试方便，我把刚才定义的 Cached 类直接包含在这里，确保能运行
class NanoRotaryEmbeddingCached(torch.nn.Module):
    def __init__(self, config: NanoLLMConfig, device=None):
        super().__init__()
        self.rope_theta = config.rope_theta
        self.dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.max_position_embeddings = config.max_position_embeddings
        inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device)
                / self.dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, device=device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, position_ids):
        if position_ids.max() >= self.max_seq_len_cached:
            self._set_cos_sin_cache(
                int(position_ids.max() + 1024), x.device, torch.float32
            )
        return self.cos_cached[position_ids].to(x.dtype), self.sin_cached[
            position_ids
        ].to(x.dtype)


class TestRotaryEquivalence(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = NanoLLMConfig(
            hidden_size=128,
            num_attention_heads=4,
            head_dim=32,
            rope_theta=10000.0,
            max_position_embeddings=2048,
        )
        print(f"\n[Device] {self.device}")

    def test_equivalence_standard(self):
        """测试标准顺序输入 [0, 1, 2...]"""
        batch_size, seq_len = 2, 128

        # 1. 初始化两个模型
        rope_dynamic = NanoRotaryEmbedding(self.config, device=self.device).to(
            self.device
        )
        rope_cached = NanoRotaryEmbeddingCached(self.config, device=self.device).to(
            self.device
        )

        # 2. 构造输入
        x = torch.randn(batch_size, 4, seq_len, 32, device=self.device)
        position_ids = (
            torch.arange(seq_len, device=self.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )

        # 3. 运行 Forward
        cos_dyn, sin_dyn = rope_dynamic(x, position_ids)
        cos_cache, sin_cache = rope_cached(x, position_ids)

        # 4. 对比
        # 允许极小的浮点误差 (1e-5)
        diff_cos = (cos_dyn - cos_cache).abs().max().item()
        diff_sin = (sin_dyn - sin_cache).abs().max().item()

        print(
            f"[Standard Seq] Max Diff Cos: {diff_cos:.2e} | Max Diff Sin: {diff_sin:.2e}"
        )

        self.assertTrue(torch.allclose(cos_dyn, cos_cache, atol=1e-5))
        self.assertTrue(torch.allclose(sin_dyn, sin_cache, atol=1e-5))

    def test_equivalence_arbitrary_positions(self):
        """测试乱序位置输入 [0, 100, 5...]"""
        batch_size = 2

        # 构造乱序 position_ids
        # Batch 0: [0, 50, 5]
        # Batch 1: [5, 0, 50]
        position_ids = torch.tensor([[0, 50, 5], [5, 0, 50]], device=self.device)
        seq_len = position_ids.shape[1]

        rope_dynamic = NanoRotaryEmbedding(self.config, device=self.device).to(
            self.device
        )
        rope_cached = NanoRotaryEmbeddingCached(self.config, device=self.device).to(
            self.device
        )

        x = torch.randn(batch_size, 4, seq_len, 32, device=self.device)

        cos_dyn, sin_dyn = rope_dynamic(x, position_ids)
        cos_cache, sin_cache = rope_cached(x, position_ids)

        diff_cos = (cos_dyn - cos_cache).abs().max().item()
        print(f"[Arbitrary Pos] Max Diff Cos: {diff_cos:.2e}")

        self.assertTrue(torch.allclose(cos_dyn, cos_cache, atol=1e-5))
        self.assertTrue(torch.allclose(sin_dyn, sin_cache, atol=1e-5))


if __name__ == "__main__":
    unittest.main()
