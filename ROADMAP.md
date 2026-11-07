# HariAI — Execution Roadmap (14-Phase Sequence)

**Project Mission:** Maximum intelligence per megabyte, per parameter, and per unit of computation.  
**Continual Learning Philosophy:** "Learn while active. Consolidate while idle. Improve continuously. Preserve what was already learned."

---

## Complete 14-Phase Execution Sequence

### Phase 0 — Research, Feasibility & Continual Learning Literature [COMPLETED]
- [x] Comprehensive literature review of SLMs, reasoning architectures, and decision systems.
- [x] Comparative evaluation of 15 model architectures.
- [x] Comparative evaluation of 10 reasoning/decision methodologies.
- [x] Comparative evaluation of 5 multi-agent coordination architectures.
- [x] In-depth literature review of 15 continual learning methodologies (Google Sleep Framework, Nested Learning, Titans, Sleep-Time Compute, EWC, Replay).
- [x] Open-weight teacher model & dataset license audit.
- [x] Mathematical feasibility analysis of the $<50\text{ MB}$ parameter budget.
- [x] Compute and financial budget estimation for training from scratch.
- [x] Design of candidate architectures (UAIA-Nano).
- [x] Formulation of testable novelty hypotheses.

### Phase 1 — Original Architecture Design & Memory Hierarchy [ACTIVE]
- [ ] Implement formal mathematical model specifications in PyTorch (`hari_ai/model/`).
- [ ] Rotary Position Embeddings (RoPE) with KV-cache support.
- [ ] RMSNorm with numerical stabilization.
- [ ] Grouped-Query Latent Attention (GQLA) layer.
- [ ] SwiGLU Feed-Forward Network.
- [ ] Tied Vocabulary Embedding & LM Head ($V = 4,096$).
- [ ] Calibrated Multi-Task Decision & Action Classification Head.
- [ ] Hierarchical Memory Interface Definition (Levels 1-5: Working, Episodic, Semantic, Procedural, Parametric).
- [ ] Creator Profile Module with dynamic age computation (`hari_ai/profile/`).
- [ ] Comprehensive unit tests verifying forward/backward pass, shapes, and gradients.

### Phase 2 — Tokenizer and Data Pipeline
- [ ] Train custom Byte-fallback BPE tokenizer on curated corpus ($V = 4,096$).
- [ ] Ingest, deduplicate (MinHash LSH), and filter training corpora.
- [ ] Prepare Python code and algorithm execution datasets.
- [ ] Generate teacher-distilled step-by-step reasoning trajectories (Qwen-2.5-Coder / DeepSeek-R1).
- [ ] Build synthetic structured decision and tool execution dataset.
- [ ] Implement memory-mapped binary dataset shard writer and streaming dataloader.

### Phase 3 — From-Scratch Base Model Training
- [ ] Initialize model parameters from Gaussian distribution $\mathcal{W}_0 \sim \mathcal{N}(0, \sigma^2)$.
- [ ] Implement training loop with Cosine Annealing, Warmup, Weight Decay, and Mixed Precision (BF16/FP16).
- [ ] Implement gradient clipping, checkpointing, and tensorboard/wandb logging.
- [ ] Train base model on foundational token mixture ($1.5\text{B} - 5\text{B}$ tokens).
- [ ] Produce `HariAI Base Alpha` checkpoint.

### Phase 4 — Language, Reasoning and Decision Intelligence
- [ ] Supervised Fine-Tuning (SFT) on conversational instruction dataset.
- [ ] Coding and algorithmic problem-solving specialization.
- [ ] Verifiable Reward RL / GRPO on mathematical equality and deterministic unit test executions.
- [ ] Self-verification and uncertainty calibration training.
- [ ] Produce `HariAI Intelligence Alpha` checkpoint.

### Phase 5 — Multi-Agent Coordination & Cluster Architecture
- [ ] Inter-agent structured communication protocol (Coordinator, Worker, Verifier, Researcher, Coder).
- [ ] Blackboard shared state and task decomposition manager.
- [ ] Consensus voting, conflict resolution, and execution budget guards.
- [ ] Anti-loop safety mechanisms and timeout policies.
- [ ] Produce `HariAI Decision & Cluster Alpha`.

### Phase 6 — Extreme Compression & Nano Model (<50 MB Target)
- [ ] Quantization-Aware Training (QAT) to INT4 and BitNet 1.58-bit ternary representations.
- [ ] Weight sharing ablation and low-rank weight factorization.
- [ ] Export validated model weights to compact binary format ($< 50\text{ MB}$).
- [ ] Evaluate perplexity and benchmark retention after quantization.
- [ ] Produce `HariAI Nano` checkpoint ($\le 50\text{ MB}$).

### Phase 7 — Native Inference Runtime (CPU/SIMD/GPU)
- [ ] Pure C++ / Python SIMD inference kernel (AVX2, AVX-512, NEON).
- [ ] Zero-dependency model binary loader with memory mapping (`mmap`).
- [ ] Fast KV cache manager with sliding window and dynamic memory allocation.
- [ ] Hardware detection engine (automatic fallback: CUDA $\to$ Metal $\to$ CPU SIMD).
- [ ] Streaming token generation interface.

### Phase 8 — Terminal Interface & Python SDK
- [ ] Build standalone `hari` CLI with ASCII banner, interactive REPL, streaming responses, and history.
- [ ] Implement CLI subcommands: `hari ask`, `hari think`, `hari decide`, `hari code`, `hari cluster`.
- [ ] Implement memory and dream subcommands (`hari memory ...`, `hari dream ...`, `hari learn ...`).
- [ ] Build Python SDK: `from hari_ai import HariAI, Cluster`.
- [ ] Build local HTTP REST server (`hari serve`) with OpenAPI specification.

### Phase 9 — Cross-Platform Packaging & Distribution
- [ ] PyPI package build and release (`pip install hari-ai`).
- [ ] Standalone executables for Windows, Linux, and macOS.
- [ ] WinGet and Scoop manifest generation for Windows.
- [ ] Homebrew formula for macOS/Linux.
- [ ] Docker container build with minimal footprint.
- [ ] Hugging Face model repository upload with full model cards.

### Phase 10 — Benchmarking & Research Validation
- [ ] Evaluate on standard benchmarks: GSM8K (math), HumanEval / MBPP (code), ARC (reasoning), MMLU subset.
- [ ] Measure tokens/sec on CPU (x86_64 and ARM64) and GPU.
- [ ] Evaluate calibration error (ECE) of the Decision Engine.
- [ ] Publish `HariAI Technical Report v1`.

### Phase 11 — Dream Engine & Continual Learning
- [ ] Wake/Sleep lifecycle controller and idle-detection scheduler.
- [ ] Hierarchical memory consolidation pipeline (Episodic $\to$ Semantic $\to$ Procedural).
- [ ] Autonomous practice curriculum generator for recorded mistakes.
- [ ] Safe modular parameter updating (LoRA/adapter expansion without base weight contamination).
- [ ] Permanent regression test suite and automatic checkpoint rollback on degradation.
- [ ] Resource-aware sleep governor (CPU, GPU, battery, and thermal limits).

### Phase 12 — Cluster-Wide Learning
- [ ] Distributed experience replay and shared verified semantic memory.
- [ ] Conflict resolution between worker findings.
- [ ] Coordinator-audited parameter-efficient synchronization.
- [ ] Safe knowledge exchange between independent local instances.

### Phase 13 — Public Release & Ongoing Research
- [ ] Release HariAI v1.0.0 with full Dream Engine capabilities.
- [ ] Publish peer-reviewed research paper on UAIA-Nano and Dream Engine sleep consolidation.
- [ ] Launch continuous open-source contribution and community roadmap.
