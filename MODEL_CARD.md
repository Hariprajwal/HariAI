---
language:
- en
license: apache-2.0
tags:
- small-language-model
- slm
- edge-ai
- decision-engine
- continual-learning
- quantization
- bitnet
- lean4
pipeline_tag: text-generation
library_name: hari-ai
inference: false
---

# HariAI Nano — Model Card

**Founder & Creator:** K R Hari Prajwal  
**GitHub:** [https://github.com/Hariprajwal](https://github.com/Hariprajwal)  
**LinkedIn:** [https://www.linkedin.com/in/k-r-hari-prajwal-655373256/](https://www.linkedin.com/in/k-r-hari-prajwal-655373256/)  
**License:** Apache 2.0  
**Project Mission:** "Small enough to run anywhere. Intelligent enough to think. Open enough for everyone."

---

## 1. Model Overview

**HariAI Nano** is an original, open-weight, extremely small artificial intelligence foundation model designed from scratch.
Its complete downloadable neural network weights are strictly below **50 MB** ($46.4\text{M}$ total parameters, occupying $\approx \mathbf{26.0\text{ MB}}$ at INT4).

Unlike conventional small language models that waste the majority of their parameter capacity on massive vocabulary embedding matrices, HariAI uses a **$4,096$-token Byte-Fallback BPE Tokenizer** with tied input/output embeddings, allocating $>95\%$ of parameter capacity directly to deep relational transformer blocks.

---

## 2. Key Architectural Features

- **Unified Adaptive Intelligence Architecture (UAIA-Nano):**
  - **Layers:** 16 deep-and-narrow transformer blocks.
  - **Hidden Dimension ($d_{\text{model}}$):** 512.
  - **Intermediate FFN ($d_{\text{ff}}$):** 1,376 (SwiGLU activation).
  - **Grouped-Query Latent Attention (GQLA):** 8 query heads, 2 KV heads (4:1 ratio) for fast local CPU inference.
  - **Positional Encoding:** Rotary Position Embeddings (RoPE) with $\theta = 10000.0$.
  - **Normalization:** RMSNorm ($\epsilon = 10^{-5}$).
- **Open-JEV Typed Decision Engine:**
  - Evaluates system states in a single forward pass without requiring slow, unstructured text generation.
  - Outputs typed readouts: `Choice` (action selection), `Score` (calibrated confidence in $[0, 1]$), and `Noul` (binary permission gating).
- **Formal Verification Engine (Lean 4 & Symbolic Grounding):**
  - Autoformalization mode generates Lean 4 theorem code and Python assertions.
  - Solutions are verified mechanically by deterministic compilers, completely eliminating mathematical hallucinations.
- **Dream Engine (Continual Learning & Catastrophic Forgetting Prevention):**
  - Wake State: Interactive, zero-training latency.
  - Sleep State: Offline background consolidation over a 5-tier memory hierarchy (Working, Episodic, Semantic, Procedural, Parametric).
  - Trains modular $0.5\text{ MB}$ specialist LoRA adapters tested against permanent regression suites before promotion.

---

## 3. Parameter and Storage Budget

| Format / Precision | Effective Bits/Param | Total File Size | Status Under 50 MB Ceiling |
|---|---|---|---|
| **FP32 (Uncompressed)** | 32.0 bits (4.00 B/p) | $185.7\text{ MB}$ | Reference Only |
| **FP16 / BF16** | 16.0 bits (2.00 B/p) | $92.9\text{ MB}$ | Training Checkpoint |
| **INT8 (Quantized)** | 8.0 bits (1.00 B/p) | $46.4\text{ MB}$ | Viable ($\le 50\text{ MB}$) |
| **INT4 (Packed Q4_0)** | 4.5 bits ($0.56\text{ B/p}$) | **$\approx 26.0\text{ MB}$** | **Primary Target** ($24\text{ MB}$ buffer) |
| **BitNet b1.58 (Ternary)** | 1.58 bits ($0.20\text{ B/p}$) | **$\approx 16.8\text{ MB}$** | **Maximum Capacity Frontier** |

---

## 4. Quickstart Usage

### Python SDK
```python
from hari_ai import HariAI, Cluster

# 1. Initialize HariAI Nano
ai = HariAI("nano")

# 2. General Query
print(ai.ask("Explain recursion"))

# 3. Deliberate Reasoning Mode
result = ai.think("Find the time complexity of merge sort")
print(result["thought"])

# 4. Typed Decision Mode (Open-JEV Style)
decision = ai.decide(
    situation="A test failed with an IndexError",
    actions=["retry", "debug_bounds", "ignore", "abort"]
)
print(decision["selected_action"])  # 'debug_bounds'
print(decision["confidence"])       # 0.96

# 5. Multi-Agent Cluster
cluster = Cluster()
cluster.start(workers=3)
summary = cluster.run("Sort an array with verified tests")
print(summary["verification_passed"]) # True
```

### Terminal CLI
```bash
# Interactive REPL
hari run

# Subcommands
hari ask "Who created you?"
hari think "Prove that sqrt(2) is irrational"
hari decide "The user requested deletion of database"
hari dream now
hari info
```

