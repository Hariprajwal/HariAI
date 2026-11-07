# HariAI — Project Status

**Founder & Creator:** K R Hari Prajwal  
**GitHub:** [https://github.com/Hariprajwal](https://github.com/Hariprajwal)  
**LinkedIn:** [https://www.linkedin.com/in/k-r-hari-prajwal-655373256/](https://www.linkedin.com/in/k-r-hari-prajwal-655373256/)  
**Current Phase:** Phase 0 (Research & Feasibility) Completed $\to$ Entering Phase 1 (Architecture Design)  
**Total Phases:** 14 (Phase 0 through Phase 13)  
**Last Updated:** November 2026

---

## 1. 14-Phase Progression Dashboard

| Phase | Description | Status | Completion Target | Key Deliverable |
|---|---|---|---|---|
| **Phase 0** | Research, Feasibility & Continual Learning Review | **COMPLETED** | Nov 2026 | `docs/research/PHASE_0_RESEARCH_SPECIFICATION.md` |
| **Phase 1** | Architecture Design & Hierarchical Memory Spec | **IN PROGRESS** | Nov 2026 | `ARCHITECTURE.md`, UAIA-Nano Spec, Memory Layer Math |
| **Phase 2** | Tokenizer & Curated Data Pipeline | PENDING | Nov 2026 | 4096-vocab Byte-fallback BPE & Synthetic Data Pipeline |
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

- **Model Disk Size Target:** $\le 50.00\text{ MB}$ (Current estimation: $\approx 26.0\text{ MB}$ at INT4 / $28.5\text{ MB}$ at BitNet 1.58b).
- **Target Parameter Count:** $46.4\text{M} - 68.5\text{M}$ parameters.
- **Vocabulary Size:** $4,096$ tokens (BPE Byte-Fallback, Tied Weights).
- **Context Length:** $2,048$ tokens.
- **Independence:** $100\%$ trained from random initialization $\mathcal{W}_0$. Zero borrowed weights.
- **Continual Learning:** Wake/Sleep decoupled lifecycle with hierarchical 5-tier memory.
- **Hardware Targets:** CPU SIMD (AVX2, AVX-512, ARM NEON) + optional CUDA/Metal.

---

## 3. Active Work Items (Phase 1)
- [x] Establish Phase 0 Research Specification, Feasibility Report, and Continual Learning Survey.
- [x] Create project tracking infrastructure (`ROADMAP.md`, `ARCHITECTURE.md`, `RESEARCH_LOG.md`, `DECISIONS.md`, `BENCHMARKS.md`, `DATA_LICENSES.md`).
- [x] Implement Creator Profile module with dynamic age calculation (`hari_ai/profile/creator.py`).
- [x] Establish Hierarchical Memory interfaces (`hari_ai/memory/`).
- [x] Establish Dream Engine lifecycle state controller stubs (`hari_ai/dream/`).
- [ ] Define exact PyTorch neural layer modules for UAIA-Nano (`hari_ai/model/`).
- [ ] Write layer-level shape validation and unit tests (`tests/test_architecture.py`).
