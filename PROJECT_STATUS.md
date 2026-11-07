# HariAI — Project Status

**Founder & Creator:** K R Hari Prajwal  
**GitHub:** [https://github.com/Hariprajwal](https://github.com/Hariprajwal)  
**LinkedIn:** [https://www.linkedin.com/in/k-r-hari-prajwal-655373256/](https://www.linkedin.com/in/k-r-hari-prajwal-655373256/)  
**Current Phase:** Phase 1 (Architecture Design) Completed $\to$ Entering Phase 2 (Tokenizer & Data Pipeline)  
**Total Phases:** 14 (Phase 0 through Phase 13)  
**Last Updated:** November 2026

---

## 1. 14-Phase Progression Dashboard

| Phase | Description | Status | Completion Date | Key Deliverables |
|---|---|---|---|---|
| **Phase 0** | Research, Feasibility & Continual Learning Review | **COMPLETED** | Nov 2026 | `docs/research/PHASE_0_RESEARCH_SPECIFICATION.md` |
| **Phase 1** | Architecture Design & Hierarchical Memory Spec | **COMPLETED** | Nov 2026 | `ARCHITECTURE.md`, `hari_ai/model/`, 21 Unit Tests Passed |
| **Phase 2** | Tokenizer & Curated Data Pipeline | **ACTIVE** | Nov 2026 | 4096-vocab Byte-fallback BPE & Synthetic Data Pipeline |
| **Phase 3** | From-Scratch Base Model Training | PENDING | Dec 2026 | HariAI Base Alpha Checkpoint |
| **Phase 4** | Language, Reasoning and Decision Intelligence | PENDING | Dec 2026 | SFT, Distilled Reasoning & Verifiable Reward Checkpoint |
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
- **Vocabulary Size:** $4,096$ tokens (Tied Embeddings: $4,096 \times 512 = 2.09\text{M}$ params).
- **Topology:** $16$ layers, $d_{\text{model}} = 512$, $d_{\text{ff}} = 1,376$, $8$ query heads, $2$ KV heads (GQA 4:1).
- **Dual Heads:** Autoregressive Next-Token ($V=4,096$) + 32-Action Decision Classifier & Calibrated Uncertainty.
- **Hardware Targets:** CPU SIMD (AVX2, AVX-512, ARM NEON) + optional CUDA/Metal.
- **Continual Learning:** 5-tier Hierarchical Memory + Wake/Sleep Dream Engine.
- **Test Suite:** 21 automated unit tests passing across architecture, memory, dream engine, and creator profile.

---

## 3. Active Work Items (Phase 2)
- [ ] Implement custom Byte-Fallback BPE Tokenizer training script (`hari_ai/tokenizer/`).
- [ ] Define special control and decision tokens (`<|thought|>`, `<|action|>`, `<|observe|>`, `<|verify|>`, `<|confidence|>`).
- [ ] Build dataset ingestion, filtering, and MinHash LSH deduplication pipeline (`hari_ai/data/`).
- [ ] Build synthetic dataset generators for algorithmic code and verifiable math reasoning.
