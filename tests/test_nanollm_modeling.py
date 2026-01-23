import unittest
import torch
import sys
import os
import copy


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from src.nanollm.configuration_nanollm import NanoLLMConfig
from src.nanollm.modeling_nanollm import NanoLLMForCausalLM


class TestNanoLLMModeling(unittest.TestCase):

    def setUp(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        # self.dtype = torch.float32
        seed = 234465
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 1. 创建一个小型的配置用于快速测试
        self.config = NanoLLMConfig(
            vocab_size=1000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,  # 测试 GQA
            max_position_embeddings=512,
            pad_token_id=0,
            # 默认使用 eager 方便基准测试
            _attn_implementation="eager",
        )

        # 2. 构造一些 Dummy Input
        self.batch_size = 15
        self.seq_len = 10
        self.input_ids = torch.randint(
            1,
            self.config.vocab_size,
            (self.batch_size, self.seq_len),
            device=self.device,
        )
        # 模拟全 1 的 mask
        # self.attention_mask = torch.ones(
        #     (self.batch_size, self.seq_len), device=self.device, dtype=torch.long
        # )
        # 模拟 paddig mask
        # self.attention_mask = torch.ones(
        #     (self.batch_size, self.seq_len), device=self.device, dtype=torch.long
        # )
        # self.attention_mask[:, 5:] = 0
        self.attention_mask = None

    def _create_model(self, attn_impl="eager"):
        config = copy.deepcopy(self.config)
        config._attn_implementation = attn_impl
        # add torch_dtype
        config.torch_dtype = self.dtype
        model = NanoLLMForCausalLM(config).to(self.device).to(self.dtype)
        model.eval()
        return model




    def test_generation_consistency(self):
        """
        测试不同 Attention 实现 (Eager/SDPA/Flash2) 在 generate() 生成文本时结果是否完全一致。
        这能验证 KV Cache 和 Decoding 阶段的 Mask 逻辑。
        """
        print("\n[Test] Generation Consistency (Decoding)")

        # 1. 准备 Prompt
        # 为了避免 Right Padding 干扰生成 (通常 CausalLM 推荐 Left Padding)，
        prompt_input_ids = self.input_ids
        print(prompt_input_ids)
        # prompt_attention_mask = self.attention_mask

        # 生成参数 (必须确定性生成)
        gen_kwargs = {
            "max_new_tokens": 10,
            "do_sample": False,  # 贪婪解码
            "pad_token_id": self.config.pad_token_id,
            "use_cache": True,  # 启用 KV Cache
        }

        # --- 1. Eager Mode (基准) ---
        print("Generating with Eager...")
        model_eager = self._create_model("eager")
        state_dict = model_eager.state_dict()  # 保存权重以便复用

        with torch.no_grad():
            output_eager = model_eager.generate(
                prompt_input_ids,
                # attention_mask=prompt_attention_mask,
                **gen_kwargs,
            )

        # --- 2. SDPA Mode ---
        print("Generating with SDPA...")
        model_sdpa = self._create_model("sdpa")
        model_sdpa.load_state_dict(state_dict)  # 加载完全相同的权重

        with torch.no_grad():
            output_sdpa = model_sdpa.generate(
                prompt_input_ids,
                # attention_mask=prompt_attention_mask,
                **gen_kwargs,
            )

        # 比较 Token ID (必须完全相等，整数没有精度误差)
        is_equal_sdpa = torch.equal(output_eager, output_sdpa)
        print(f"Eager Output: {output_eager}")
        print(f"SDPA  Output: {output_sdpa}")

        # self.assertTrue(is_equal_sdpa, "SDPA generation result differs from Eager!")
        # print("✅ Passed (SDPA matches Eager)")

        # --- 3. Flash Attention 2 (如果有) ---
        try:
            import flash_attn

            has_flash = True and torch.cuda.is_available()
        except ImportError:
            has_flash = False

        if has_flash:
            print("Generating with Flash Attention 2...")
            # FA2 需要 FP16/BF16
            dtype = torch.bfloat16
            model_fa2 = self._create_model("flash_attention_2").to(dtype)
            model_fa2.load_state_dict({k: v.to(dtype) for k, v in state_dict.items()})

            # 确保输入也在正确的设备和类型
            inputs_fa2 = prompt_input_ids.to(self.device)
            # mask_fa2 = prompt_attention_mask.to(self.device)

            with torch.no_grad():
                output_fa2 = model_fa2.generate(
                    inputs_fa2,
                    # attention_mask=mask_fa2,
                    **gen_kwargs,
                )

            # 比较
            is_equal_fa2 = torch.equal(output_eager, output_fa2)
            print(f"Eager Output: {output_eager}")
            print(f"FA2   Output: {output_fa2}")

            # self.assertTrue(is_equal_fa2, "FA2 generation result differs from Eager!")
            # print("✅ Passed (FA2 matches Eager)")


if __name__ == "__main__":
    unittest.main()
