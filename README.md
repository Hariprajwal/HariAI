# HariAI (UAIA-Nano)

<div align="center">

```
     _   _            _    ___  
    | | | | __ _ _ __| |  / _ \ 
    | |_| |/ _` | '__| | | | | |
    |  _  | (_| | |  | | | |_| |
    |_| |_|\__,_|_|  |_|  \___/ 

        Tiny in size.
        Big in intelligence.
```

**An experimental, compact AI runtime and from-scratch model research project. Its prototype checkpoint is small; it is not yet a competitive general-purpose language model.**

[![CI/CD Test Suite](https://img.shields.io/badge/Tests-84%20Passing-brightgreen.svg)](#)
[![Model Size](https://img.shields.io/badge/Model%20Size-26.0%20MB-blue.svg)](#)
[![Architecture](https://img.shields.io/badge/Architecture-UAIA--Nano-orange.svg)](#)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](#)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](#)

</div>

---

## 👨‍💻 Creator & Origin

- **Founder & Creator:** **K R Hari Prajwal**
- **Origin:** Karnataka, India
- **Date of Birth:** 23 December 2004
- **Age at Creation:** 21 years old
- **Official Creator Statement:**
  > *"Hari made me! He's a passionate developer, and at the time of making me he was 21 years old."*
- **GitHub:** [https://github.com/Hariprajwal](https://github.com/Hariprajwal)
- **LinkedIn:** [https://www.linkedin.com/in/k-r-hari-prajwal-655373256/](https://www.linkedin.com/in/k-r-hari-prajwal-655373256/)

---

## 🎯 The Core Mission

Modern frontier AI models require tens of gigabytes of VRAM, massive clusters, and constant cloud connections. **HariAI** is engineered under a radical constraint:

> **The research goal is to maximize useful capability within a small model budget.**
> **Complete downloadable neural network weights strictly $\le 50.00\text{ MB}$.**

HariAI achieves this via an original, from-scratch neural architecture (**UAIA-Nano**) combined with formal mathematical grounding (Lean 4), typed single-forward-pass decision heads (Open-JEV style), multi-agent cluster intelligence, and a continual learning **Dream Engine** inspired by biological sleep consolidation.

---

## 🧠 Architectural Highlights (UAIA-Nano)

- **Total Parameters:** $46,433,280$ parameters (~$46.4\text{M}$).
- **Disk Footprint:**
  - **Symmetric INT4:** **$26.0\text{ MB}$** (0.5 bytes per parameter, well below the 50 MB budget).
  - **BitNet 1.58b Ternary:** **$16.8\text{ MB}$**.
- **Layers & Width:** 16 transformer blocks, $d_{\text{model}} = 512$, $d_{\text{ff}} = 1,376$ (SwiGLU).
- **Grouped-Query Latent Attention (GQLA):** 8 query heads, 2 key/value heads (4:1 compression ratio).
- **Tied Vocabulary Embeddings:** Byte-fallback BPE ($V = 4,096$, lossless UTF-8 roundtrip).
- **Dual-Engine Design:**
  - **Autoregressive Language Modeling:** Generating reasoning, dialogue, and code.
  - **Typed Decision Heads (Open-JEV Style):**
    - `Choice`: Discrete action classification logits.
    - `Score`: Calibrated probability / confidence metric.
    - `Noul`: Binary permission gate (autonomous action vs. human-in-the-loop authorization).
- **Formal Grounding:** Lean 4 formal math kernel verification and deterministic Python assertion execution sandbox to eliminate mathematical and logical hallucinations.

---

## 💤 The Dream Engine (Continual Learning)

HariAI eliminates catastrophic forgetting through a biologically-inspired **Wake/Sleep Lifecycle**:

1. **Wake Phase:**
   - Active interaction and inference.
   - All experiences, queries, and failures are recorded in **Episodic Memory** (Level 2).
2. **Sleep Phase (Background Consolidation):**
   - Automatically triggered during user idle periods or memory threshold triggers.
   - Replays episodic memories and generates synthetic practice curricula from mistakes.
   - Extracts general facts into **Semantic Memory** (Level 3) and verified algorithmic templates into **Procedural Memory** (Level 4).
   - Safe parameter adaptation via isolated **LoRA low-rank adapters** while keeping base weights $\mathcal{W}_0$ permanently frozen.
   - Automated **regression rollback gate**: If candidate adapter degradation exceeds $>1.0\%$ on permanent benchmark suites, the update is rolled back automatically.

---

## ⚡ Quickstart & Installation

### 1. Local Setup
```bash
git clone https://github.com/Hariprajwal/HariAI.git
cd HariAI
pip install -e . --no-build-isolation --no-deps
```

### 2. Standalone Terminal Interface (`hari`)

On Windows, run directly from the repository folder:
```powershell
./hari.bat
# or
.\hari.ps1
```

Or run via Python:
```bash
python -m hari_ai.cli
```

### Use a local Ollama model for chat

HariAI chat uses an installed Ollama chat model when the Ollama service is running. It discovers installed models automatically and prefers common instruction model families. To choose a specific installed model, set `HARIAI_OLLAMA_MODEL` to the exact name shown by `ollama list` (for example, `qwen3:8b`). Set `HARIAI_OLLAMA_HOST` only when Ollama uses a non-default endpoint. Chat sends the system prompt and recent conversation turns.

The small HariNano checkpoint is a from-scratch architecture experiment trained on a tiny synthetic curriculum for 60 steps by default. It has not been pretrained on a large text corpus or meaningfully instruction-tuned, so it should not be presented as a capable standalone assistant. Training larger or more capable weights requires substantial curated data and compute; the Ollama integration provides local inference and does not itself train or distill weights. If Ollama is unavailable, the chat UI identifies responses as offline fallback content and does not report fabricated confidence.

### Arithmetic student research experiment

`scripts/collect_teacher_arithmetic.py` queries a chosen local Ollama teacher and records answers with their provenance and an independent arithmetic check. `scripts/train_arithmetic_student.py` trains a randomly initialized student on the verified teacher answers and a generated arithmetic curriculum, then evaluates held-out expressions before and after training and quantization. It writes a report and enforces a 35,000,000-byte limit on the exported checkpoint. Select a new output directory for each run. These measurements cover only simple arithmetic; they do not measure broad assistant quality.

### 3. Interactive CLI Commands

```powershell
# Interactive Gemma/ChatGPT-style streaming chat
hari

# Fast single-turn question
hari ask "What is quantum computing?"

# Deliberate reasoning mode with <|thought|> chain
hari think "Prove that the square root of 2 is irrational"

# Open-JEV typed single-forward-pass decision evaluation
hari decide "Deploy cluster node to production"

# Verifiable Python code generation with invariant assertions
hari code "Implement binary search"

# Inspect Hierarchical Memory status
hari memory status

# Run Dream Engine sleep consolidation manually
hari dream now

# Inspect Creator Profile
hari info
```

---

## 🐍 Python SDK

```python
from hari_ai import HariAI, Cluster, ChatEngine

# 1. Initialize HariAI Nano
ai = HariAI("nano")
print(ai.ask("Who created you?"))
# Output: "Hari made me! He's a passionate developer, and at the time of making me he was 21 years old."

# 2. Step-by-Step Deliberate Reasoning
thought_result = ai.think("Calculate the determinant of a 2x2 matrix")
print(thought_result["thought"])
print(thought_result["result"])

# 3. Typed Single-Forward-Pass Decision Evaluation
decision = ai.decide("Critical file deletion requested", actions=["proceed", "abort", "confirm"])
print(f"Action: {decision['selected_action']}, Confidence: {decision['confidence']}")
print(f"Human Approval Required: {decision['requires_human_approval']}")

# 4. Multi-Agent Cluster Orchestration
cluster = Cluster()
coordinator = cluster.coordinator
result = coordinator.solve_task("Develop and verify an in-place sorting algorithm")
print(result["final_solution"])

# 5. Multi-Turn Conversational Chat Session
chat = ChatEngine()
for chunk in chat.chat_stream("Hello! Can you help me learn linear algebra?"):
    print(chunk, end="", flush=True)
```

---

## 🧪 Automated Test Suite

HariAI is backed by **84 automated unit tests** covering all 17 system modules:

```bash
python -m unittest discover tests
```

```text
Ran 84 tests in 35.199s
OK
```

Modules tested:
- `test_architecture.py` — Forward/backward passes, GQLA attention, SwiGLU, tied embeddings, decision heads.
- `test_tokenizer.py` — Lossless UTF-8 byte-fallback BPE encoding/decoding.
- `test_data_pipeline.py` — MinHash LSH deduplication, binary memmap streaming.
- `test_trainer.py` — AdamW decoupled weight decay, cosine annealing, checkpointing.
- `test_phase4_intelligence.py` — Multi-teacher distillation router, GRPO verifiable rewards.
- `test_verification.py` — Lean 4 formal math auto-verification, Python assertion sandbox.
- `test_cluster.py` — Blackboard protocol, anti-loop execution budgets, Security Auditor, Custom Roles.
- `test_quantization.py` — Symmetric INT4 packing, BitNet 1.58b ternary scaling, $\le 50\text{ MB}$ budget assertion.
- `test_runtime.py` — Hardware auto-detection (CUDA, DirectML, CPU SIMD), KV cache token generation.
- `test_dream_controller.py` — Wake/Sleep triggers, 5-tier memory consolidation.
- `test_dream_adapter.py` — LoRA linear modules, base weight freezing, regression rollback gates.
- `test_cluster_sync.py` — Cluster-wide learning, Verified Supremacy conflict resolution, FedAvg adapter synchronization.
- `test_chat_engine.py` — Conversational dialogue, multi-turn context retention, streaming generator.
- `test_creator_profile.py` — Dynamic age calculation, official statement preservation.
- `test_cli.py` — Standalone terminal subcommands.
- `test_benchmarks.py` — Pass@1 code verification, decision calibration (ECE), model size limits.

---

## 📦 Cross-Platform Packaging

- **Docker:**
  ```bash
  docker build -t hari-ai:latest .
  docker run -it hari-ai:latest
  ```
- **Windows (WinGet / Scoop):** Manifests provided in `packaging/winget/` and `packaging/scoop/`.
- **macOS / Linux (Homebrew):** Formula provided in `packaging/homebrew/Formula/hari.rb`.

---

## 📜 License & Citation

Licensed under the [Apache License, Version 2.0](LICENSE).

```bibtex
@software{hari_ai_2026,
  author = {K R Hari Prajwal},
  title = {HariAI: Original Open-Weight Autonomous Intelligence under 50 MB},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/Hariprajwal/HariAI}
}
```

<div align="center">
<b>Built with ❤️ by K R Hari Prajwal in Karnataka, India.</b>
</div>

