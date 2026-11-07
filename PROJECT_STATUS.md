# HariAI — Project Status

**Founder & Creator:** K R Hari Prajwal  
**GitHub:** [https://github.com/Hariprajwal](https://github.com/Hariprajwal)  
**LinkedIn:** [https://www.linkedin.com/in/k-r-hari-prajwal-655373256/](https://www.linkedin.com/in/k-r-hari-prajwal-655373256/)  
**Current Phase:** Phase 2 (Tokenizer & Data Pipeline) Completed $\to$ Entering Phase 3 (Base Model Training Infrastructure)  
**Total Phases:** 14 (Phase 0 through Phase 13)  
**Last Updated:** November 2026

---

## 1. 14-Phase Progression Dashboard

| Phase | Description | Status | Completion Date | Key Deliverables |
|---|---|---|---|---|
| **Phase 0** | Research, Feasibility & Continual Learning Review | **COMPLETED** | Nov 2026 | `docs/research/PHASE_0_RESEARCH_SPECIFICATION.md` |
| **Phase 1** | Architecture Design & Hierarchical Memory Spec | **COMPLETED** | Nov 2026 | `ARCHITECTURE.md`, `hari_ai/model/` (UAIA-Nano) |
| **Phase 2** | Tokenizer & Curated Data Pipeline | **COMPLETED** | Nov 2026 | `ByteFallbackBPETokenizer`, `MinHashLSH`, `ShardWriter`, Synthetic Generators |
| **Phase 3** | From-Scratch Base Model Training | **ACTIVE** | Nov 2026 | PyTorch Trainer (`hari_ai/trainer/`), Checkpointing, AdamW, Cosine Scheduler |
| **Phase 4** | Language, Reasoning and Decision Intelligence | PENDING | Dec 2026 | SFT, Lean 4 Verifiable Reward Checkpoint |
| **Phase 5** | Multi-Agent Coordination & Cluster Architecture | PENDING | Jan 2027 | Inter-Agent Protocol & Distributed Blackboard State |
| **Phase 6** | Extreme Compression (<50 MB Target) | PENDING | Jan 2027 | Quantized INT4 / BitNet 1.58b Model Weights ($\le 50\text{ MB}$) |
| **Phase 7** | Native Inference Runtime (CPU/SIMD/GPU) | PENDING | Feb 2027 | Standalone C++/Python Inference Engine without Ollama |
| **Phase 8** | Terminal CLI & Python SDK | PENDING | Feb 2027 | `hari` CLI with Streaming REPL & `import hari_ai` SDK |
| **Phase 9** | Cross-Platform Distribution | PENDING | Mar 2027 | PyPI (`pip install hari-ai`), WinGet, Scoop, Docker |
| **Phase 10** | Benchmarking & Research Validation | PENDING | Mar 2027 | Independent Evaluation Suite & Technical Report v1 |
| **Phase 11** | Dream Engine & Continual Learning | PENDING | Apr 2027 | Wake/Sleep Controller, Autonomous Synthetic Curriculum, LoRA Consolidation |
| **Phase 12** | Cluster-Wide Learning | PENDING | Apr 2027 | Distributed Experience Replay & Audited Parameter Sync |
| **Phase 13** | Public Open-Source Release & Ongoing Research | PENDING | May 2027 | v1.0.0 Release on GitHub & Model Weights on Hugging Face |

---

## 2. Key Metrics & Constraints Tracking

- **Model Disk Size Target:** $\le 50.00\text{ MB}$ (Empirical: $\approx \mathbf{26.0\text{ MB}}$ at INT4 / $\mathbf{16.8\text{ MB}}$ at BitNet 1.58b).
- **Verified Parameter Count:** $46,433,280$ total parameters ($44,336,128$ non-embedding).
- **Vocabulary Size:** $4,096$ tokens (Byte-fallback BPE with lossless UTF-8 roundtrip).
- **Data Pipeline:** MinHash LSH deduplication + memory-mapped uint16 binary sharding with zero-RAM streaming.
- **Decision Engine:** Jev-style typed readouts (`Choice`, `Score`, `Noul`) in a single forward pass.
- **Formal Verification:** Lean 4 autoformalization & deterministic test checks.
- **Hardware Targets:** CPU SIMD (AVX2, AVX-512, ARM NEON) + optional CUDA/Metal.
- **Test Suite:** 29 automated unit tests passing across all components.

---

## 3. Active Work Items (Phase 3)
- [ ] Implement full training loop in `hari_ai/trainer/trainer.py` with AdamW, Cosine Annealing, and Gradient Clipping.
- [ ] Implement checkpoint serialization and validation sanity training loop.
- [ ] Add TensorBoard / CSV experiment logging.
