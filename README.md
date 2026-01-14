# nanoLLM 🔬

A hands-on research & engineering playground for **learning modern LLM architectures** by **re-implementing**, **ablating**, and **benchmarking** key architectural and optimization variants under **fixed compute (FLOPs)**.

> **Base model**: this project uses the Qwen3 0.6B as baseline model, with training/eval tooling built on **🤗 Transformers** + **Accelerate** (and optional FSDP).

---

## 🧩 Features & Roadmap

### Features

* **Base Architecture:** RMSNorm, SwiGLU, RoPE, Multi-head Attention (MHA), Grouped Query Attention (GQA).

* **Attention**

  * MHA (baseline)
  * MLA (multi-head latent attention)
  * Gated attention (attention output gating / QKV gating variants)
  * Linear attention
  * Sparse attention
  * Sliding-window attention
  * Hybrid architecture

* **MoE**

  * Top-k routing
  * Load balancing losses

* **mHC / Hyper-Connections**

  * multi-stream / hyper residual pathways

* **Engram-style memory**

  * Engram: retrieval / hash memory modules

* **Optimization & Training**

  * AdamW baseline vs. **Muon**


### Roadmap

#### Phase 0 — Foundations

* [ ] Config system (YAML/OmegaConf or dataclasses)
* [ ] Dataset pipeline (streaming + shuffling + packing)
* [ ] Tokenizer integration via 🤗 Transformers (reuse Qwen tokenizer)
* [ ] Baseline decoder-only model (Qwen3-style)
* [ ] Training loop with Accelerate (fp16/bf16, grad accumulation, ckpt)
* [ ] Evaluation harness (perplexity + small task suite)
* [ ] Logging (W&B or TensorBoard) + run manifest export

#### Phase 1 — Baseline + MoE (your stated priority)

* [ ] Dense baseline reproduction: stable loss curve, expected PPL
* [ ] MoE FFN block (Top-2/Top-1 routing)
* [ ] Load-balancing losses (aux loss variants) + metrics (expert usage entropy, overflow)
* [ ] Capacity factor + token dropping policy
* [ ] Fixed-FLOPs comparison scripts (dense vs MoE at matched compute)

#### Phase 2 — Attention variants

* [ ] Sliding-window attention
* [ ] Sparse attention (block/global-local)
* [ ] MLA-style KV compression family
* [ ] Gated attention variants
* [ ] Linear attention baseline (1–2 representative forms)
* [ ] Hybrid configs (e.g., every N layers global attention)

#### Phase 3 — Memory + residual/path tricks

* [ ] Engram memory module (retrieval + gating)
* [ ] mHC / hyper connections (n-stream residual)
* [ ] Combined ablations (MoE + attention + memory)

---

## Repository layout

```text
nanoLLM/
├── configs/                        # YAML configs (model/train/eval)
│   ├── model/
│   │   ├── baseline_0_6B.yaml      # Dense baseline (Qwen3-style)
│   │   └── moe_a0_6B.yaml          # MoE baseline (priority)
│   │   └── experimental_mhc.yaml
│   └── train/                      # Training hyperparameters
│       ├── pretrain_adamw.yaml
│       └── pretrain_muon.yaml
│
├── src/                            # Python package root (HF-compatible)
│   └── nanollm/
│       ├── __init__.py
│       ├── configuration_nanollm.py    # HF Config (PretrainedConfig)
│       ├── modeling_nanollm.py         # HF Model (PreTrainedModel)
│       ├── modeling_blocks.py          # TransformerBlock / MoEBlock assembly
│       │
│       ├── components/                 # Pluggable building blocks
│       │   ├── norms.py                # RMSNorm / LayerNorm variants
│       │   ├── rotary.py               # RoPE utilities
│       │   ├── mlp.py                  # FFN / SwiGLU / gated variants
│       │   ├── attention/
│       │   │   ├── mha.py
│       │   │   ├── mla.py
│       │   │   ├── linear.py
│       │   │   ├── sparse.py
│       │   │   ├── sliding_window.py
│       │   │   └── hybrid.py
│       │   ├── moe/
│       │   │   ├── router.py           # top-k routing
│       │   │   ├── experts.py          # expert MLP
│       │   │   └── losses.py           # load-balancing losses (DeepSeek-style)
│       │   ├── memory/
│       │   │   └── engram.py           # retrieval/hash memory + gating
│       │   └── residual/
│       │       └── mhc.py              # hyper-connections / multi-stream residuals
│       │
│       ├── data/
│       │   ├── datasets.py             # HF datasets / streaming loaders
│       │   ├── packing.py              # sequence packing
│       │   └── collate.py              # batch collation
│       │
│       ├── optim/                      # One true place for optim & schedulers
│       │   ├── adamw.py
│       │   ├── muon.py                 # Custom Muon optimizer
│       │   └── schedulers.py           # cosine/warmup/etc.
│       │
│       ├── train/
│       │   ├── trainer.py              # accelerate-based trainer
│       │   ├── losses.py               # LM loss + aux losses (moe, engram, mtp)
│       │   └── hooks.py                # optional: callbacks (log, ckpt, eval)
│       │
│       ├── eval/
│       │   ├── perplexity.py
│       │   ├── harness.py              # downstream eval glue
│       │   └── metrics.py
│       │
│       └── utils/
│           ├── flops.py                # CRITICAL: FLOPs estimator/accounting
│           ├── metrics.py              # training metrics + EMA, etc.
│           ├── logging.py              # W&B/TB logger adapters
│           ├── checkpoint.py
│           └── seed.py
│
├── scripts/                           # CLI entrypoints
│   ├── train.py
│   ├── eval.py
│   ├── estimate_flops.py
│   └── sweep.py                        # optional
│
├── tests/
│   ├── test_shapes.py
│   ├── test_attention_equivalence.py
│   └── test_moe_routing.py
│
├── requirements.txt
├── pyproject.toml                      # (recommended) or setup.cfg
└── README.md
```

---

## Quickstart

### 1. Train dense baseline

```bash
python scripts/train.py \
  --config configs/model/baseline_0_6B.yaml \
  --train-config configs/train/pretrain_adamw.yaml
```

### 2. Train MoE variant

```bash
python scripts/train.py \
  --config configs/model/moe_a0_6B.yaml \
  --train-config configs/train/pretrain_muon.yaml
```

### 3. Evaluate perplexity

```bash
python scripts/eval.py \
  --config configs/model/baseline_0_6B.yaml \
  --ckpt path/to/checkpoint
```
### 4. Estimate FLOPs

```bash
python scripts/estimate_flops.py \
  --config configs/model/baseline_0_6B.yaml
```

---

## Implementation notes (what to reuse vs write)

Reuse from 🤗 Transformers:

* tokenizer / vocab / special tokens
* dataset loading utilities
* (optionally) weight init conventions / config patterns

Write in nanoLLM:

* a clean, minimal Qwen3-style model (so you understand it)
* attention / MoE / memory variants as components
* fixed-FLOPs accounting + fair benchmarking harness

---

## TODO (starter list)

### Baseline model

* [ ] RMSNorm + RoPE + SwiGLU FFN
* [ ] KV cache support + causal mask correctness tests
* [ ] HF-compatible `Config` + `from_pretrained`-style loading (optional)

### MoE (priority)

* [ ] Router: top-1/top-2, jitter noise, z-loss (optional)
* [ ] Losses: balance/importance/load losses (DeepSeek-like variants)
* [ ] Capacity factor + dispatch implementation
* [ ] Metrics dashboard for expert load

### Attention variants

* [ ] Sliding window attention with KV cache
* [ ] Block sparse attention
* [ ] MLA module + ablation knobs (rank, shared projection, etc.)
* [ ] Gated attention (output gate, Q/K gate variants)
* [ ] Linear attention baseline

### Engram + mHC

* [ ] Retrieval table + hashing + gating API
* [ ] Plug memory into attention context or FFN residual
* [ ] mHC multi-stream residual with fused-friendly layout (later)
