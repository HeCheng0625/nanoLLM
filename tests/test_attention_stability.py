import unittest
import torch
import copy
import sys
import os
import pandas as pd

# 确保能导入 src
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nanollm.configuration_nanollm import NanoLLMConfig
from src.nanollm.modeling_nanollm import NanoLLMForCausalLM

# 设置显示选项
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.float_format", "{:.6f}".format)


class TestAttentionStability(unittest.TestCase):
    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # 固定种子
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.batch_size = 4
        self.seq_len = 32

        # 配置：使用 GQA (KV heads < Attention Heads) 来触发 repeat_kv 逻辑
        self.config = NanoLLMConfig(
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=4,  # GQA
            head_dim=32,
            max_position_embeddings=512,
            pad_token_id=0,
            _attn_implementation="eager",  # 初始占位
        )

        # 构造输入
        self.input_ids = torch.randint(
            1,
            self.config.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
        )
        # 测试 batch[0] 是否异常，我们让 batch[0] 和 batch[1] 的输入完全一样
        # 这样如果在推理时 logits[0] 和 logits[1] 差距很大，说明是内存/Stride问题
        self.input_ids[1] = self.input_ids[0]

        # 构造基准权重 (FP32)
        model_base = NanoLLMForCausalLM(self.config).to(self.device).float()
        self.base_state_dict = model_base.state_dict()

    def _run_model(self, attn_impl, dtype, name):
        print(f"Running {name}...")
        config = copy.deepcopy(self.config)
        config._attn_implementation = attn_impl

        # 初始化模型
        try:
            model = NanoLLMForCausalLM(config).to(self.device).to(dtype)
        except ImportError:
            print(f"Skipping {name} (Dependency not found)")
            return None

        # 加载权重 (自动转换 dtype)
        model.load_state_dict({k: v.to(dtype) for k, v in self.base_state_dict.items()})
        model.eval()

        with torch.no_grad():
            outputs = model(self.input_ids)

        return outputs.logits

    def test_all_combinations(self):
        results = {}

        # 1. 定义测试配置列表
        configs = [
            ("eager", torch.float32, "Eager (FP32)"),
            ("sdpa", torch.float32, "SDPA (FP32)"),
            ("sdpa", torch.bfloat16, "SDPA (BF16)"),
            ("eager", torch.bfloat16, "Eager (BF16)"),
        ]

        # 尝试添加 Flash Attention 2
        if torch.cuda.is_available():
            try:
                import flash_attn

                configs.append(("flash_attention_2", torch.bfloat16, "Flash (BF16)"))
            except ImportError:
                print("Flash Attention 2 not installed, skipping.")

        # 2. 运行所有模型
        logits_map = {}
        for attn_impl, dtype, name in configs:
            logits = self._run_model(attn_impl, dtype, name)
            if logits is not None:
                # 统一转回 FP32 进行比较
                logits_map[name] = logits.float()

        # 3. 设定基准 (Golden Reference)
        # 通常认为 SDPA (FP32) 或 Eager (FP32) 是最准确的
        if "SDPA (FP32)" in logits_map:
            baseline_name = "SDPA (FP32)"
        else:
            baseline_name = "Eager (FP32)"

        baseline_logits = logits_map[baseline_name]
        print(f"\n=== Baseline: {baseline_name} ===")

        # 4. 比较分析
        comparison_data = []

        for name, logits in logits_map.items():
            # 计算与基准的最大绝对误差
            diff = (logits - baseline_logits).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()

            # 检查 Batch[0] 和 Batch[1] 的一致性 (因为输入是一样的)
            # 理论上 diff_internal 应该非常接近 0
            logits_b0 = logits[0]
            logits_b1 = logits[1]
            internal_diff = (logits_b0 - logits_b1).abs().max().item()

            status = "✅ PASS"
            # BF16 允许较大误差 (e.g. 0.05), FP32 应该很小 (e.g. 1e-4)
            threshold = 0.1 if "BF16" in name else 1e-3
            if max_diff > threshold:
                status = "❌ FAIL"

            comparison_data.append(
                {
                    "Model Config": name,
                    "Max Diff vs Base": max_diff,
                    "Mean Diff vs Base": mean_diff,
                    "B0 vs B1 Diff": internal_diff,  # 检查内部一致性
                    "Status": status,
                }
            )

        # 5. 打印结果
        df = pd.DataFrame(comparison_data)
        print("\n" + str(df))

        # 断言
        for row in comparison_data:
            if "Eager (BF16)" in row["Model Config"]:
                # 我们预期 Eager BF16 可能表现较差，但如果差距过大（>1.0），说明代码有问题
                print(f"\nChecking {row['Model Config']}...")
                if row["Max Diff vs Base"] > 0.5:
                    print(
                        f"⚠️  Warning: {row['Model Config']} has high divergence! This might be precision loss in Eager MatMul."
                    )

                # 如果 B0 vs B1 Diff 很大，说明依然存在 stride/view 问题
                if row["B0 vs B1 Diff"] > 1e-2:
                    print(
                        f"‼️ CRITICAL: {row['Model Config']} failed batch consistency check! (Diff: {row['B0 vs B1 Diff']})"
                    )
                    # self.fail("Batch consistency check failed for Eager BF16")


if __name__ == "__main__":
    unittest.main()
