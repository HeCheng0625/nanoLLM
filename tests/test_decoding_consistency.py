import torch
import sys
import os
import copy
import pandas as pd
from termcolor import colored  # 可选: pip install termcolor

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.nanollm.configuration_nanollm import NanoLLMConfig
from src.nanollm.modeling_nanollm import NanoLLMForCausalLM

# 设置 Pandas 显示，方便看具体差异
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)


def compare_sequences(ref_seqs, target_seqs, batch_size):
    """
    比较两个生成的序列 (Batch, Seq_Len)，返回详细的差异报告。
    """
    mismatches = []

    # 确保长度一致 (generate 可能会因为 stop token 提前结束，这里假设 max_new_tokens 固定)
    min_len = min(ref_seqs.shape[1], target_seqs.shape[1])
    ref_seqs = ref_seqs[:, :min_len]
    target_seqs = target_seqs[:, :min_len]

    for b in range(batch_size):
        if not torch.equal(ref_seqs[b], target_seqs[b]):
            # 找到第一个不同的位置
            diff_indices = torch.where(ref_seqs[b] != target_seqs[b])[0]
            first_diff_idx = diff_indices[0].item()

            mismatches.append(
                {
                    "Batch Index": b,
                    "Status": "❌ MISMATCH",
                    "First Diff At Token": first_diff_idx,
                    "Ref Token": ref_seqs[b][first_diff_idx].item(),
                    "Got Token": target_seqs[b][first_diff_idx].item(),
                    # "Full Ref": ref_seqs[b].tolist(),
                    # "Full Got": target_seqs[b].tolist()
                }
            )
        else:
            mismatches.append(
                {
                    "Batch Index": b,
                    "Status": "✅ MATCH",
                    "First Diff At Token": "-",
                    "Ref Token": "-",
                    "Got Token": "-",
                }
            )

    return pd.DataFrame(mismatches)


def run_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 234465
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"Running Decoding Consistency Test on {device.upper()}...\n")

    # 1. 配置模型 (使用 GQA 来增加难度，触发 repeat_kv)
    batch_size = 15
    prompt_len = 10
    max_new_tokens = 10

    config = NanoLLMConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
        pad_token_id=0,
        _attn_implementation="eager",
    )

    # 2. 构造 Prompt
    input_ids = torch.randint(
        1, config.vocab_size, (batch_size, prompt_len), device=device
    )
    print("input_ids", input_ids)

    # 3. 初始化基准权重 (FP32)
    print("Initializing Base Weights (FP32)...")
    base_model = NanoLLMForCausalLM(config).to(device).float()
    base_state_dict = base_model.state_dict()

    # 4. 定义要测试的配置
    test_configs = [
        # (Mode, Dtype, Name)
        ("sdpa", torch.float32, "SDPA (FP32)"),  # 通常作为 Golden Reference
        ("eager", torch.float32, "Eager (FP32)"),
        ("sdpa", torch.bfloat16, "SDPA (BF16)"),
        ("eager", torch.bfloat16, "Eager (BF16)"),
    ]

    if torch.cuda.is_available():
        try:
            import flash_attn

            test_configs.append(("flash_attention_2", torch.bfloat16, "Flash (BF16)"))
        except ImportError:
            print("Skipping FlashAttention2 (not installed)")

    # 5. 生成结果收集
    outputs = {}

    # 生成参数
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,  # 必须是 Greedy Search 才能比较
        "use_cache": True,  # 开启 KV Cache
        "pad_token_id": config.pad_token_id,
    }

    print("-" * 60)
    for mode, dtype, name in test_configs:
        print(f"Generating with [{name}]...")

        # 创建模型
        run_config = copy.deepcopy(config)
        run_config._attn_implementation = mode

        try:
            model = NanoLLMForCausalLM(run_config).to(device).to(dtype)
        except Exception as e:
            print(f"Skipping {name}: {e}")
            continue

        # 强制载入相同权重
        model.load_state_dict({k: v.to(dtype) for k, v in base_state_dict.items()})
        model.eval()

        # Generate
        with torch.no_grad():
            # 注意: 输入必须转为对应的 dtype 吗？input_ids 是 long，不需要转
            # 但内部 forward 时的 cache 需要 dtype
            generated_ids = model.generate(input_ids, **gen_kwargs)

        outputs[name] = generated_ids

    # 6. 对比分析
    # 设定基准：优先选 SDPA (FP32)，没有则选 Eager (FP32)
    # ref_name = "SDPA (FP32)" if "SDPA (FP32)" in outputs else "Eager (FP32)"
    ref_name = "SDPA (BF16)"
    ref_seqs = outputs[ref_name]

    print("\n" + "=" * 80)
    print(f"COMPARISON REPORT (Reference: {ref_name})")
    print("=" * 80)

    for name, seqs in outputs.items():
        if name == ref_name:
            continue

        print(f"\nTarget: [{name}]")

        # print seqs
        for i in range(batch_size):
            print(f"Batch {i}: {seqs[i]}")
            print(f"Ref Batch {i}: {ref_seqs[i]}")
            print("-" * 80)

        # 核心对比逻辑
        df = compare_sequences(ref_seqs, seqs, batch_size)

        # 简单的统计
        failed_rows = df[df["Status"] == "❌ MISMATCH"]
        if len(failed_rows) > 0:
            print(
                colored(
                    f"-> FAILED: {len(failed_rows)}/{batch_size} samples mismatched.",
                    "red",
                )
            )
            print(df)

            # 特别检查：如果只有 Batch[0] 挂了
            if 0 in failed_rows["Batch Index"].values and len(failed_rows) == 1:
                print(
                    colored(
                        "⚠️  Warning: Only Batch 0 failed! This strongly suggests a stride/view issue in repeat_kv.",
                        "yellow",
                    )
                )
        else:
            print(colored("-> PASSED: All sequences match perfectly.", "green"))


if __name__ == "__main__":
    run_test()
