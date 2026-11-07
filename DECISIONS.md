# HariAI — Architecture Decision Records (ADRs)

This document tracks all formal architectural and technical decisions made during the lifecycle of HariAI.

---

## ADR-0001: Train from Scratch vs Fine-Tune Existing Base
- **Status:** APPROVED
- **Date:** November 2026
- **Context:** Many projects fine-tune or merge existing models (e.g. Llama or Qwen). HariAI's core mission is to establish an original architecture and completely independent open-weight model with novel internal mechanics.
- **Decision:** HariAI will strictly train its weights from scratch using random initialization $\mathcal{W}_0 \sim \mathcal{N}(0, \sigma^2)$. Existing models are permitted only as teachers for knowledge distillation and reference evaluation.
- **Consequences:** Requires custom data ingestion pipeline, custom tokenizer, and end-to-end pre-training loop. Ensures complete legal, licensing, and architectural independence.

---

## ADR-0002: Hard Budget Ceiling of 50 MB on Weights
- **Status:** APPROVED
- **Date:** November 2026
- **Context:** To ensure instant downloads, zero-friction edge execution, and universal operation even on low-end laptops and mobile phones, a strict size limit is mandated.
- **Decision:** The complete downloadable neural network weights for `HariAI Nano` must not exceed $50\text{ MB}$.
- **Consequences:** Constrains parameter count to $\approx 46\text{M}-70\text{M}$ parameters with INT4 / BitNet 1.58b quantization. Requires compact vocabulary ($V=4,096$) and tied embeddings.

---

## ADR-0003: Compact Vocabulary ($V=4,096$) with Byte-Fallback
- **Status:** APPROVED
- **Date:** November 2026
- **Context:** Large vocabularies ($32\text{k}-128\text{k}$) consume up to $65\text{MB}$ in embedding tables alone, exceeding the entire target budget.
- **Decision:** Implement a custom Byte-fallback BPE tokenizer with exactly $4,096$ tokens.
- **Consequences:** Embedding weights occupy only $\approx 1.17\text{ MB}$ at INT4. Sequence length in tokens increases by $\approx 1.3\times - 1.5\times$ relative to 32k tokenizers, but $100\%$ byte coverage is preserved with zero OOV errors.

---

## ADR-0004: Dual-Engine Architecture (Language + Structured Decision Head)
- **Status:** APPROVED
- **Date:** November 2026
- **Context:** Agentic actions, tool selection, and cluster coordination are often unreliable when forced purely through raw text generation on sub-100M parameter models.
- **Decision:** Implement a dedicated linear decision classifier and uncertainty head on top of the shared transformer representations.
- **Consequences:** The model produces both natural language tokens and low-latency structured action signals in a single forward pass.

---

## ADR-0005: Independent Terminal-First Runtime
- **Status:** APPROVED
- **Date:** November 2026
- **Context:** Many projects rely on external runtimes like Ollama or llama.cpp as mandatory dependencies.
- **Decision:** HariAI will provide its own independent CLI (`hari`), local inference runtime with native CPU SIMD acceleration, and standalone Python SDK.
- **Consequences:** Zero external runtime dependencies required to download, run, and chat with HariAI.



---

## ADR-0006: Hierarchical 5-Tier Memory Architecture
- **Status:** APPROVED
- **Date:** November 2026
- **Context:** Storing all interaction history directly into prompt contexts quickly exhausts the 2,048 token window and degrades model focus, while attempting to fine-tune weights on every single user message introduces latency and catastrophic forgetting.
- **Decision:** Establish five distinct memory tiers:
  1. Level 1 (Working Memory): Ephemeral context and scratchpad.
  2. Level 2 (Episodic Memory): Raw session interaction logs in structured append-only format.
  3. Level 3 (Semantic Memory): Deduplicated facts and concepts.
  4. Level 4 (Procedural Memory): Verified code snippets, tool schemas, and decision workflows.
  5. Level 5 (Parametric Memory): Frozen base weights + modular low-rank specialist adapters.
- **Consequences:** Provides clear operational boundaries and allows asynchronous background consolidation without impacting live interactive performance.

---

## ADR-0007: Dream Engine Wake/Sleep Decoupled Continual Learning
- **Status:** APPROVED
- **Date:** November 2026
- **Context:** An AI model must continually improve from its mistakes and user interactions, but running gradient optimization during live chat causes intolerable latency, battery drain, and instability.
- **Decision:** Implement a dual-state lifecycle (Wake State for inference; Sleep State for consolidation). When the machine is idle, plugged into power, or explicitly commanded (`hari dream now`), the system consolidates memory, synthesizes practice problems, trains candidate LoRA adapters, runs permanent regression tests, and promotes or rolls back the candidate.
- **Consequences:** Ensures zero latency overhead during active usage while enabling autonomous continuous capability improvement. Requires preemption handlers so user input instantly resumes interactive mode within 50 ms.


---

## ADR-0008: Integration of Typed Decision Heads Inspired by Open-JEV
- **Status:** APPROVED
- **Date:** November 2026
- **Context:** Language models that rely exclusively on autoregressive text generation to make control choices suffer from syntax formatting failures, slow token generation latency, and poorly calibrated confidence. `kyegomez/open-jev` demonstrates a fast, typed "System One" decision architecture licensed under Apache 2.0.
- **Decision:** Integrate Jev-style typed readouts directly into HariAI's Decision Engine:
  1. `Choice`: Categorical action selection over discrete options (e.g. `[retry, debug, stop, branch]`).
  2. `Score`: Calibrated continuous scalar in $[0, 1]$ measuring uncertainty or reward expectation.
  3. `Noul`: Binary classification head for critical gating (e.g. requiring human permission or terminating execution).
- **Consequences:** Decisions are evaluated in a single forward pass over cached hidden states without generating multi-token JSON strings.
