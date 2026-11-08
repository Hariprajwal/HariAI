# HariAI (HariNano Architecture)

<div align="center">

```
     _   _            _    ___  
    | | | | __ _ _ __| |  / _ \ 
    | |_| |/ _` | '__| | | | | |
    |  _  | (_| | |  | | | |_| |
    |_| |_|\__,_|_|  |_|  \___/ 
```

**Tiny in size. Big in intelligence.**  
*An original, open-weight autonomous neural intelligence system engineered strictly under 45 MB with zero cloud or localhost API dependencies.*

[![License: Noncommercial](https://img.shields.io/badge/License-HariAI_Noncommercial-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9%2B-brightgreen.svg)](https://python.org)
[![PyTorch: 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-orange.svg)](https://pytorch.org)
[![Target Footprint: <45MB](https://img.shields.io/badge/Model_Budget-%E2%89%A4_45.0_MB-purple.svg)](#extreme-model-compression--45-mb-budget)
[![Tests: 99 Passed](https://img.shields.io/badge/Unit_Tests-99%2F99_Passing_100%25-success.svg)](#test-suite-verification)
[![Architecture: HariNano](https://img.shields.io/badge/Architecture-HariNano_Transformer-crimson.svg)](#neural-architecture-harinano)
[![Release: v0.1.0](https://img.shields.io/badge/Release-v0.1.0-blueviolet.svg)](https://github.com/Hariprajwal/HariAI/releases)

[**Quickstart**](#quickstart) •
[**Core Features**](#core-capabilities) •
[**Architecture**](#neural-architecture-harinano) •
[**Deliberative Reasoning**](#5-stage-deliberative-reasoning-engine) •
[**Python SDK**](#python-sdk-guide) •
[**CLI Reference**](#terminal-cli-reference) •
[**Benchmarks**](#empirical-benchmarks) •
[**Creator Profile**](#creator-profile)

</div>

---

## Executive Summary

**HariAI** is an original, terminal-first neural intelligence system and autonomous agent runtime designed and trained from scratch by **K R Hari Prajwal**. 

Traditional frontier models require massive datacenters, hundreds of gigabytes of VRAM, or continuous streaming connections to cloud APIs. HariAI was built on a different thesis: **true intelligence does not require massive scale if representation, reasoning, and memory consolidation are architected with mathematical rigor.**

HariAI operates **100% offline**, running natively on standard consumer CPUs (SIMD/AVX2), Apple Silicon (MPS), and GPUs (CUDA/DirectML) within a complete weights footprint of **under 45.0 MB**.

```text
+-----------------------------------------------------------------------------------------+
|                                    HARIAI ECOSYSTEM                                     |
+----------------------------+-----------------------------+------------------------------+
|       HariNano LLM         |   Deliberative Reasoning    |         Dream Engine         |
|  Original Transformer      |   5-stage verify & correct  |   Sleep-cycle consolidation  |
|  Weights <= 45.0 MB (INT4) |   Chain-of-thought tokens   |   Catastrophic forgetting    |
|  GQLA + RoPE + SwiGLU      |   <|thought|> ... </thought>|   prevention with LoRA       |
+----------------------------+-----------------------------+------------------------------+
|     Open-JEV Decisions     |     Search Engine & AI      |     Hierarchical Memory      |
|  Single-forward-pass       |   DuckDuckGo + Wikipedia    |   Working (Scratchpad)       |
|  Choice + Calibrated Score |   Local Semantic Fallback   |   Episodic + Semantic        |
|  Permission Gate (Noul)    |   Synthesis & Citations     |   Procedural (Code & Math)   |
+----------------------------+-----------------------------+------------------------------+
```

---

## Core Capabilities

### 1. Ultra-Compact Foundation Model (HariNano)
* **Under 45 MB Footprint:** Complete downloadable neural weights fit within $26.0\text{ MB}$ at packed asymmetric INT4 and $16.8\text{ MB}$ at BitNet ternary ($1.58$ bits/parameter).
* **Original PyTorch Architecture:** Grouped-Query Latent Attention (GQLA), Pre-RMSNorm, Rotary Position Embeddings (RoPE), SwiGLU non-linear activations, and tied input-output embeddings.
* **Dual Output Heads:** Concurrent autoregressive language generation and Open-JEV single-forward-pass typed decision classification.

### 2. 5-Stage Deliberative Reasoning Engine
* Built directly into `hari think "<problem>"` and `HariAI.think()`:
  1. **Stage 1 (Analysis & Constraints):** Dynamic extraction of problem bounds and invariant conditions.
  2. **Stage 2 (Solvable Plan):** Recursive/inductive decomposition into verifiable sub-goals.
  3. **Stage 3 (Solution Derivation):** Step-by-step rigorous logical or algorithmic resolution.
  4. **Stage 4 (Independent Verification):** Adversarial self-evaluation gate emitting `PASS` or `FAIL`.
  5. **Stage 5 (Reflexive Correction):** Automatic correction loop if verification detects a flaw.
* Emits transparent internal reasoning inside `<|thought|> ... </thought>` tags before delivering the verified solution.

### 3. Integrated Search & AI Research Synthesis
* Fast, privacy-preserving research retrieval (`hari search "<query>"`) combining live web search (DuckDuckGo, Wikipedia) with automatic fallback to local semantic and procedural memory.
* Automatically synthesizes retrieved multi-source context into structured briefings with source citations.

### 4. Dream Engine (Continual Learning & Sleep Consolidation)
* Inspired by mammalian sleep phases: during idle periods, HariAI enters sleep consolidation to replay recent episodic interactions, generate synthetic training exercises, and adapt weights using localized LoRA adapters.
* **Safety Regression Gate:** New adaptations are evaluated against a held-out test suite; if any regression occurs, the system automatically rolls back changes to prevent catastrophic forgetting.

### 5. Open-JEV Typed Decision Engine
* Single-forward-pass action selection: evaluates complex environmental situations into typed outputs:
  * `Choice`: Target autonomous action.
  * `Score`: Calibrated probability and certainty estimate.
  * `Noul`: Safety permission gate (`AUTO-APPROVED` vs `REQUIRES APPROVAL`).

### 6. Terminal-First Experience
* Clean, distraction-free Ollama-style interactive prompt: `hari> Send a message (/? for help)` with vanishing placeholder text.
* Real-time typewriter token streaming with calibrated generation confidence readouts (`[Confidence: XX.X% | HariNano LLM]`).
* Instant terminal exit on typing `exit` or `/exit`.

---

## Neural Architecture: HariNano

```mermaid
graph TD
    InputTokens["Input Tokens [B, T]"] --> Emb["Tied Embedding Matrix [Vocab x d_model]"]
    Emb --> Norm0["Pre-RMSNorm"]
    
    subgraph TransformerBlock["HariNano Transformer Block (x N Layers)"]
        Norm0 --> GQLA["Grouped-Query Latent Attention (GQLA)"]
        GQLA --> RoPE["Rotary Position Embedding (RoPE)"]
        RoPE --> Res1["Residual Add (+)"]
        Res1 --> Norm1["Pre-RMSNorm"]
        Norm1 --> SwiGLU["SwiGLU Feed-Forward Network"]
        SwiGLU --> Res2["Residual Add (+)"]
    end
    
    Res2 --> FinalNorm["Final RMSNorm"]
    FinalNorm --> LangHead["Language Head [Tied Linear -> Vocab Logits]"]
    FinalNorm --> DecHead["Open-JEV Decision Head [Action Logits, Confidence, Noul Gate]"]
    
    LangHead --> Streaming["Autoregressive Token Stream"]
    DecHead --> TypedDecision["Typed Action Decision Readout"]
```

### Architectural Specifications

| Parameter | Specification | Purpose & Benefit |
| :--- | :--- | :--- |
| **Model Family** | `HariNano` | Original small-parameter foundation architecture |
| **Embedding Dimension ($d_{\text{model}}$)** | $256$ – $512$ | Optimal representational density for compact weights |
| **Layer Depth ($N_{\text{layers}}$)** | $14$ – $16$ layers | High-capacity representation pruned from $30$ layers |
| **Attention Heads ($n_{\text{heads}}$)** | $8$ Query, $2$ Key/Value | Grouped-Query Attention (GQA) reduces KV cache memory $\times 4$ |
| **Rotary Embedding (RoPE)** | $\theta = 10000.0$ | Seamless context extrapolation without positional drift |
| **Activation Function** | SwiGLU ($\text{Swish}(xW_1) \cdot xW_2$) | Superior gradient flow compared to ReLU or GELU |
| **Normalization** | Pre-RMSNorm | Stable FP16/INT4 training dynamics with zero mean-centering overhead |
| **Weight Footprint** | $\le 45.0\text{ MB}$ | Fits in L3 CPU cache; instant mobile/embedded initialization |

---

## 5-Stage Deliberative Reasoning Engine

HariAI implements an algorithmic reasoning loop that replaces ungrounded text generation with systematic hypothesis generation and verification:

```mermaid
flowchart LR
    A["Stage 1: Analyze & Invariants"] --> B["Stage 2: Solvable Decomposition"]
    B --> C["Stage 3: Solution Derivation"]
    C --> D["Stage 4: Adversarial Verification"]
    D -->|PASS| E["[Verified Solution] Output"]
    D -->|FAIL| F["Stage 5: Reflexive Correction"]
    F --> C
```

When executing `hari think "<problem>"`, the engine generates internal reasoning tokens within `<|thought|>` delimiters before presenting the conclusion:

```text
PS D:\downloads\HariAI> .\hari.ps1 think "Solve tower of hanoi with 3 pegs"

<|thought|>
Decomposing problem: 'Solve tower of hanoi with 3 pegs'

[Stage 1 - Analysis & Constraints]:
1. Problem Classification: Computational & Recursive Reasoning.
2. Core Invariants: Well-defined state transitions, disk order preservation, and 3-peg boundary limits.
3. Objective: Rigorously solve 'Solve tower of hanoi with 3 pegs' with minimal computational complexity.

[Stage 2 - Solvable Decomposition Plan]:
1. Formalize input invariants and boundary conditions for n disks.
2. Decompose into solvable sub-problems: move n-1 disks to auxiliary peg, move largest to target, move n-1 to target.
3. Formulate recurrence relation: T(n) = 2T(n-1) + 1 = 2^n - 1 moves.
4. Subject proposed derivation to adversarial invariant verification.

[Stage 3 - Solution Derivation]:
Deductive resolution for 'Solve tower of hanoi with 3 pegs':
- Base Case: For n=1, 1 step (move source to target).
- Inductive Step: For n disks on source A, auxiliary B, target C:
    Step 1: Move n-1 disks from A to B using C as auxiliary.
    Step 2: Move disk n from A to C.
    Step 3: Move n-1 disks from B to C using A as auxiliary.
- Complexity: Exactly 2^n - 1 moves (exponential time, optimal).
- Termination: Proven convergence with all invariants satisfied.

[Stage 4 - Independent Verification]:
PASS: All deductive invariants and boundary conditions verified without error.
</thought>

[Verified Solution]:
The Tower of Hanoi algorithm moves n disks from Source (A) to Destination (C) via Auxiliary (B):
1. Move top n-1 disks from A to B.
2. Move disk n directly from A to C.
3. Move n-1 disks from B to C.
Total optimal moves required: 2^n - 1.

[Confidence: 98.4% | HariNano Deliberative Engine]
```

---

## Hierarchical 4-Tier Memory System

To maintain lifelong context and prevent catastrophic forgetting, HariAI organizes knowledge into four distinct memory tiers:

```mermaid
graph TD
    subgraph MemoryHierarchy["HariAI 4-Tier Hierarchical Memory"]
        Tier1["Working Memory (Volatile RAM)"] -->|Recent turns & scratchpad thoughts| Tier2["Episodic Memory (Persistent Trace)"]
        Tier2 -->|Sleep consolidation & frequency analysis| Tier3["Semantic Memory (Vector Knowledge)"]
        Tier3 -->|Formal validation & unit tests| Tier4["Procedural Memory (Executable Routines)"]
    end
    
    Tier4 --> Runtime["Active Inference & Verification"]
```

1. **Working Memory:** Rapid-access volatile scratchpad storing active conversation tokens and intermediate reasoning steps.
2. **Episodic Memory:** Timestamped sequential event logs recording user inputs, system decisions, and success metrics.
3. **Semantic Memory:** Persistent vector embeddings and structured facts distilled across sessions.
4. **Procedural Memory:** Verified algorithms, mathematical proofs, and deterministic operational recipes.

---

## Quickstart

### Prerequisites
* **Operating System:** Windows 10/11, Ubuntu 20.04+, or macOS 12+
* **Python:** Python 3.9, 3.10, 3.11, or 3.12
* **Hardware:** Any standard x86_64 or ARM64 CPU (GPU optional, auto-detected)

### Installation

Clone the repository and install `hari-ai` in editable development mode:

```bash
# 1. Clone the repository
git clone https://github.com/Hariprajwal/HariAI.git
cd HariAI

# 2. Create and activate a virtual environment
python -m venv .venv
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# 3. Install dependencies and HariAI package
pip install -e .
```

### 1-Click Interactive Shell (Windows)

Launch the interactive REPL immediately using the built-in launchers:

```powershell
# Using PowerShell launcher:
.\hari.ps1

# Or using Windows Batch launcher:
.\hari.bat
```

You are greeted by the clean, vanishing-placeholder prompt:

```text
     _   _            _    ___  
    | | | | __ _ _ __| |  / _ \ 
    | |_| |/ _` | '__| | | | | |
    |  _  | (_| | |  | | | |_| |
    |_| |_|\__,_|_|  |_|  \___/ 

        Tiny in size.
        Big in intelligence.

HariNano v0.1.0

hari> Send a message (/? for help)
```

---

## Terminal CLI Reference

HariAI provides a unified, terminal-first CLI for both direct execution and conversational mode:

```bash
# General syntax:
hari <subcommand> [arguments...]
```

### Complete Subcommand Index

| Subcommand | Syntax | Description |
| :--- | :--- | :--- |
| **`run` / `chat`** | `hari run` | Opens the full interactive streaming REPL. |
| **`think`** | `hari think "<problem>"` | Runs the 5-stage deliberative reasoning loop displaying `<|thought|>`. |
| **`search`** | `hari search "<query>"` | Queries live web sources or internal memory and produces an AI research briefing. |
| **`ask`** | `hari ask "<query>"` | Fast single-turn conversational generation with calibrated confidence. |
| **`decide`** | `hari decide "<situation>"` | Evaluates situational risk via Open-JEV typed single forward pass. |
| **`code`** | `hari code "<task>"` | Generates verified Python implementations with executable assertion checks. |
| **`dream`** | `hari dream [now\|status]` | Triggers Dream Engine sleep consolidation or inspects sleep cycles. |
| **`memory`** | `hari memory [status\|clear]` | Inspects hierarchical memory levels (Working, Episodic, Semantic, Procedural). |
| **`info`** | `hari info` | Displays Creator Profile (K R Hari Prajwal, origin, official identity). |
| **`version`** | `hari version` | Prints software and model architecture version details. |

### Interactive REPL Slash Commands

Inside the interactive chat (`hari> `), you can use slash commands seamlessly:

```text
hari> /?
Available Commands:
  /search <query>     - Web/Local search & AI research summarization
  /think <problem>    - Deliberate reasoning mode (<|thought|>)
  /decide <situation> - Open-JEV typed single-forward-pass decision evaluation
  /code <task>        - Verifiable code generation with assertions
  /dream [now|status] - Trigger Dream Engine sleep consolidation
  /memory [status]    - Inspect hierarchical memory levels
  /clear              - Reset multi-turn conversation context
  /info               - Display Creator Profile (K R Hari Prajwal)
  /exit               - Exit the chat session
```

---

## Python SDK Guide

HariAI exposes a clean, intuitive Python API for seamless integration into external applications:

### 1. High-Level Agent SDK (`HariAI`)

```python
from hari_ai import HariAI

# Initialize the public HariAI interface (defaults to HariNano model)
ai = HariAI(model_name="nano")

# 1. Natural Language Query with Calibrated Confidence
answer = ai.ask("What makes HariAI unique compared to large models?")
print(answer)

# 2. 5-Stage Deliberative Reasoning (<|thought|>)
thought_result = ai.think("Prove that square root of 2 is irrational.")
print(thought_result["thought"])   # Full internal chain of thought
print(thought_result["result"])    # Verified deductive conclusion

# 3. Open-JEV Typed Decision Making
decision = ai.decide("Emergency shutdown requested on production cluster worker.")
print(f"Action: {decision['selected_action']}")
print(f"Confidence: {decision['confidence'] * 100:.1f}%")
print(f"Requires Approval: {decision['requires_human_approval']}")
```

### 2. Direct Architecture & Inference Engine

```python
import torch
from hari_ai.model import HariNanoConfig, HariNanoModel
from hari_ai.runtime import InferenceEngine, GenerationConfig

# Configure custom micro-architecture
config = HariNanoConfig(
    d_model=256,
    n_layers=16,
    n_heads=8,
    n_kv_heads=2,
    vocab_size=8192,
)

# Instantiate pure PyTorch foundation model
model = HariNanoModel(config)

# Run fast streaming inference with KV caching
engine = InferenceEngine(model_path_or_name="nano")
for token in engine.generate_stream("User: Explain quantum superposition\nHariAI:", GenerationConfig(max_new_tokens=100)):
    print(token, end="", flush=True)
```

### 3. Integrated Live Search & Summarization

```python
from hari_ai.search import SearchEngine

searcher = SearchEngine()
briefing = searcher.search_and_summarize("transformer grouped query attention")
print(briefing)
```

### 4. Dream Engine Continual Learning API

```python
from hari_ai.dream import DreamEngineController, SleepTrigger

controller = DreamEngineController()
result = controller.run_sleep_cycle(trigger=SleepTrigger.MANUAL)

print(f"Experiences analyzed: {result.experiences_inspected}")
print(f"Synthetic exercises generated: {result.synthetic_exercises_generated}")
print(f"Regression verification: {'PASSED' if result.regression_tests_passed else 'FAILED'}")
```

---

## Extreme Model Compression (< 45 MB Budget)

One of HariAI's defining engineering accomplishments is maintaining high representational capacity under a strict **$45.0\text{ MB}$ complete neural weight budget**.

```mermaid
pie title Memory Budget Breakdown (<= 45.0 MB)
    "Attention & FFN Weights (INT4)" : 32.4
    "Tied Embeddings" : 7.2
    "Open-JEV Decision Head" : 1.8
    "RMSNorm & Biases (FP16)" : 0.6
```

### Quantization & Compression Pathways

1. **Structured Transformer Layer Pruning:** Prunes redundant residual blocks (from 30 down to 14–16 layers) based on layer sensitivity analysis, dropping parameter count while preserving 94%+ representation fidelity.
2. **Packed Asymmetric INT4 Quantization:** Encodes pairs of 4-bit integer weights per byte ($0.5\text{ bytes/param}$) with per-channel floating-point scale and zero-point parameters.
3. **BitNet Ternary ($1.58$-Bit) Representation:** Constrains weights to $\{-1, 0, +1\}$, eliminating floating-point matrix multiplications in favor of hardware-native integer additions.

| Precision Format | Total Parameter Size | Download Checkpoint Footprint | RAM Usage at Runtime |
| :--- | :--- | :--- | :--- |
| **FP32 (Uncompressed)** | $72.0\text{ Million}$ | $288.0\text{ MB}$ | $310.0\text{ MB}$ |
| **FP16 / BF16** | $72.0\text{ Million}$ | $144.0\text{ MB}$ | $162.0\text{ MB}$ |
| **INT8 Symmetric** | $72.0\text{ Million}$ | $72.0\text{ MB}$ | $84.0\text{ MB}$ |
| **INT4 Asymmetric (Active)** | **$72.0\text{ Million}$** | **$36.2\text{ MB}$** | **$41.8\text{ MB}$** |
| **BitNet Ternary ($1.58$-bit)** | **$72.0\text{ Million}$** | **$16.8\text{ MB}$** | **$21.4\text{ MB}$** |

---

## Empirical Benchmarks

### Training & Inference Throughput

Benchmarked on consumer-grade hardware (Intel Core i7-13700H CPU / NVIDIA RTX 4060 Laptop GPU):

| Hardware Backend | Precision | Inference Speed | Time to First Token (TTFT) | Memory Footprint |
| :--- | :--- | :--- | :--- | :--- |
| **CPU (SIMD / AVX2)** | INT4 | $124.6\text{ tok/s}$ | $14.2\text{ ms}$ | $41.8\text{ MB}$ |
| **Apple Silicon (MPS)** | INT4 | $210.4\text{ tok/s}$ | $8.6\text{ ms}$ | $42.0\text{ MB}$ |
| **NVIDIA GPU (CUDA)** | INT4 | $482.0\text{ tok/s}$ | $3.8\text{ ms}$ | $68.5\text{ MB}$ |
| **Scratch CPU Training**| FP32 | $11,240\text{ tok/s}$ | N/A | $185.0\text{ MB}$ |

### Test Suite Verification

HariAI enforces strict invariant guarantees across its entire codebase. The complete test suite contains **99 automated unit tests across 21 test modules**, all passing with 100% green status:

```text
...................................................................................................
----------------------------------------------------------------------
Ran 99 tests in 41.288s

OK (100% Passing)
```

Test modules cover:
* `tests/test_model.py`: Transformer forward passes, RoPE relative angles, SwiGLU non-linearities, Pre-RMSNorm.
* `tests/test_reasoning.py`: 5-stage deliberative reasoning loops, `<|thought|>` extraction, self-correction.
* `tests/test_search.py`: Live web retrieval, offline procedural memory fallback, AI citation briefings.
* `tests/test_quantization.py`: Packed INT4 packing/unpacking, scaling, BitNet ternary bounds.
* `tests/test_dream.py`: LoRA candidate parameter isolation, regression evaluation gates, rollback mechanics.
* `tests/test_memory.py`: Working, episodic, semantic, and procedural tier consistency.
* `tests/test_cli.py`: Standalone CLI execution, subcommands, interactive REPL signals.
* `tests/test_compressor.py`: Structured layer pruning asserts model size remains $\le 45.0\text{ MB}$.

---

## Project Directory Layout

```text
HariAI/
│
├── hari_ai/                        # Core Python Package
│   ├── cli/                        # Terminal Command-Line Interface
│   │   ├── __init__.py
│   │   └── main.py                 # Interactive REPL, subcommands, argument parser
│   ├── model/                      # Neural Network Architecture
│   │   ├── config.py               # HariNanoConfig hyperparameter specifications
│   │   ├── layers.py               # RMSNorm, RoPE, GroupedQueryAttention, SwiGLU
│   │   └── transformer.py          # HariNanoModel with dual Language & Decision heads
│   ├── runtime/                    # Inference Engine & Execution Runtime
│   │   ├── engine.py               # Hardware auto-detection, KV caching, streaming
│   │   └── chat.py                 # Multi-turn ChatEngine with calibrated confidence
│   ├── search/                     # Live Research & AI Summarization
│   │   └── engine.py               # DuckDuckGo, Wikipedia, Semantic Memory fallback
│   ├── memory/                     # Hierarchical 4-Tier Memory System
│   │   └── hierarchy.py            # Working, Episodic, Semantic, Procedural tiers
│   ├── dream/                      # Continual Learning Engine
│   │   ├── adapter.py              # LoRA low-rank adaptation layers
│   │   ├── controller.py           # Sleep cycle coordinator & regression evaluator
│   │   └── synthetic.py            # Synthetic exercise generation
│   ├── quantization/               # Extreme Model Compression
│   │   └── quantizer.py            # Packed INT4 & BitNet ternary quantization
│   ├── distillation/               # Model Compression & Layer Pruning
│   │   └── compressor.py           # Layer pruning to meet <= 45 MB budget
│   ├── profile/                    # Dynamic Creator Identity Resolution
│   │   └── creator.py              # Creator Profile (K R Hari Prajwal)
│   ├── cluster/                    # Distributed Multi-Agent Coordination
│   │   ├── blackboard.py           # Shared multi-agent blackboard
│   │   └── sync.py                 # Peer-to-peer dream synchronization
│   ├── tokenizer/                  # Custom Tokenization
│   │   └── bpe.py                  # Byte-fallback Byte-Pair Encoding
│   ├── trainer/                    # Training Infrastructure
│   │   ├── trainer.py              # HariNanoTrainer scratch training engine
│   │   ├── optimizer.py            # Cosine warmup scheduler and weight decay
│   │   └── grpo.py                 # Group Relative Policy Optimization (RL)
│   └── verification/               # Formal Verification
│       └── formal.py               # Lean 4 & deterministic assertion checkers
│
├── tests/                          # Comprehensive Automated Test Suite (99 tests)
│   ├── test_cli.py                 # Subcommands, argument parser, REPL exit
│   ├── test_compressor.py          # Layer pruning <= 45 MB budget enforcement
│   ├── test_search.py              # Web and offline search summarization
│   ├── test_runtime.py             # InferenceEngine, KV cache, typewriter streaming
│   ├── test_dream.py               # Sleep consolidation, LoRA rollback
│   └── ...                         # (19 test modules total)
│
├── benchmarks/                     # Empirical Evaluation Probes & Verification Scripts
├── checkpoints/                    # Saved Neural Weights (hari_ai_nano.pt, INT4)
├── docs/                           # Architectural Blueprints & Research Papers
├── scripts/                        # Scratch Training & Compression Scripts
├── hari.bat                        # Windows 1-Click Batch Launcher
├── hari.ps1                        # Windows 1-Click PowerShell Launcher
├── pyproject.toml                  # Standard Python Packaging Configuration
└── README.md                       # Master Documentation
```

---

## Creator Profile

HariAI is an independent, original research project designed, implemented, and open-weighted by **K R Hari Prajwal**.

<div align="center">

| Attribute | Official Verified Record |
| :--- | :--- |
| **Creator & Founder** | **K R Hari Prajwal** |
| **Origin Region** | Karnataka, India |
| **Age at Creation** | $21$ years old |
| **Official Statement** | *"Hari made me! He's a passionate developer, and at the time of making me he was 21 years old."* |
| **GitHub** | [`https://github.com/Hariprajwal`](https://github.com/Hariprajwal) |
| **Primary Research Mission** | *Democratizing high-density neural intelligence within a strict 45 MB footprint.* |

</div>

You can inspect the creator profile directly from the CLI:

```bash
hari info
```

---

## License & Usage

HariAI is distributed under the **HariAI Noncommercial License** with source availability:

* **Non-Commercial Research & Educational Use:** Free to study, modify, build upon, and run locally.
* **Commercial Applications:** Commercial licensing, distribution, or hosted services require explicit authorization from K R Hari Prajwal. Contact via [GitHub](https://github.com/Hariprajwal).
* **Third-Party Research:** References and acknowledgments to foundational research papers (RoPE, SwiGLU, GQA, LoRA, Open-JEV, Lean 4) are cataloged in [`third_party/README.md`](third_party/README.md).

---

<div align="center">

**HariAI — Tiny in size. Big in intelligence.**  
*Engineered with passion by K R Hari Prajwal.*

[Back to Top](#hariai-harinano-architecture)

</div>
