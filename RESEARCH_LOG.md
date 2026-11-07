# HariAI — Research Log & Hypothesis Registry

This document records research investigations, hypothesis validations, experimental ablations, and literature findings.

---

## Log Entry 001: Phase 0 Feasibility Analysis & Vocabulary Sizing
- **Date:** November 2026
- **Status:** COMPLETED
- **Topic:** Extreme parameter reduction for $<50\text{ MB}$ weights.
- **Hypothesis:** Conventional vocabularies ($V \ge 32,000$) create an intractable parameter bottleneck at sub-100M parameter scales.
- **Evidence:**
  - With $d = 512$, a $32,000$-token vocabulary requires $32,000 \times 512 = 16.38\text{M}$ parameters ($32.8\text{ MB}$ at FP16).
  - A $128,000$-token vocabulary requires $65.5\text{M}$ parameters ($131.0\text{ MB}$ at FP16).
  - Reducing vocabulary to $V = 4,096$ with byte-fallback reduces embedding parameters to $2.09\text{M}$ ($4.19\text{ MB}$ at FP16, $1.17\text{ MB}$ at INT4).
  - Tying the embedding layer with the final language modeling head cuts embedding memory by an additional $50\%$.
- **Conclusion:** Adopt $V = 4,096$ Byte-fallback BPE with tied input/output embeddings as standard architectural policy.

---

## Log Entry 002: Topology Exploration (Deep-and-Narrow vs Shallow-and-Wide)
- **Date:** November 2026
- **Status:** COMPLETED
- **Topic:** Layer depth vs hidden dimension trade-off at $45\text{M}-70\text{M}$ parameters.
- **Evidence from Prior Art:**
  - MobileLLM (Meta, 2024) demonstrated that for $125\text{M}$ models, $30$ narrow layers outperformed $12$ wide layers by $+3.2\%$ average accuracy across reasoning benchmarks.
  - Deep networks learn hierarchical composition of features; shallow networks fail to retain multi-step logical traces.
- **Decision:** Select $L = 16$ layers with $d = 512$ rather than $L = 8$ layers with $d = 724$.

---

## Log Entry 003: Decision Engine Architecture (Prompt-only vs Dual-Head)
- **Date:** November 2026
- **Status:** COMPLETED
- **Topic:** Integration of GLiDE-inspired decision classification.
- **Hypothesis:** An auxiliary decision head operating on the final transformer representations provides calibrated action probabilities without requiring the model to generate multiline JSON strings for atomic control decisions.
- **Design:** Incorporate a linear projection $\mathbf{W}_{\text{action}} \in \mathbb{R}^{512 \times 32}$ with cross-entropy loss, alongside a scalar confidence sigmoid head $\sigma(\mathbf{h} \mathbf{w}_{\text{conf}})$ with binary cross-entropy calibration loss.
- **Next Step:** Implement and benchmark in Phase 1 and Phase 5.



---

## Log Entry 004: Comprehensive Literature Review of 15 Continual Learning Methodologies
- **Date:** November 2026
- **Status:** COMPLETED
- **Topic:** Dream Engine theoretical foundations & catastrophic forgetting prevention.
- **Analyzed Works:**
  1. **Google Research Sleep Framework (Oct 2026):** Decouples real-time inference from asynchronous offline consolidation. Identifies that updating weights during sleep via distilled synthetic tasks prevents degradation on primary tasks while incorporating new facts.
  2. **Google Nested Learning:** Establishes nested optimization loops across distinct timescales (fast inner loop for episodic context, medium loop for procedural strategies, slow outer loop for base parametric consolidation).
  3. **Titans (Behrouz et al., Google 2024):** Introduces a neural long-term memory module that learns to memorize at test time via a surprise metric and momentum-based decay. Informs HariAI's working-to-episodic memory transition.
  4. **Hope Architecture:** Explores dual-memory consolidation where high-confidence episodic traces are progressively merged into permanent weights.
  5. **Test-Time Training (TTT, Sun et al., 2024):** Formulates the hidden state as an internal model trained with self-supervised loss during inference. High compute overhead; reserved for offline sleep rather than live inference in HariAI Nano.
  6. **Sleep-Time Compute (arXiv:2504.13171):** Formalizes idle-period compute budgeting, demonstrating that 30 minutes of idle compute per day can recover 95% of knowledge degradation in small language models.
  7. **Continual Learning & Catastrophic Forgetting (Kirkpatrick et al., French):** Classic plasticity-stability dilemma. Concludes that unconstrained full-weight fine-tuning invariably causes catastrophic forgetting in compact networks.
  8. **Online and Incremental Learning:** Requires bounded data streams; without replay buffers, performance degrades monotonically over time.
  9. **Experience Replay (ER):** Preserving representative historical samples and interleaving them with new experiences during gradient updates is the single most effective baseline against forgetting.
  10. **Elastic Weight Consolidation (EWC):** Measures parameter importance via diagonal Fisher Information Matrix $F_i$, penalizing updates to critical weights: $\mathcal{L}(\theta) = \mathcal{L}_{\text{new}}(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_{A, i}^*)^2$.
  11. **Progressive Neural Networks (Rusu et al.):** Instantiates new columns for new tasks while keeping old columns frozen. Too memory-intensive for a strict 50 MB budget.
  12. **Mixture-of-Experts Expansion:** Dynamically routes new tasks to new expert modules. Feasible if new experts are highly low-rank.
  13. **Low-Rank Adaptation (LoRA, Hu et al.):** Decomposes weight deltas $\Delta W = B A$. Provides parameter isolation, allowing modular addition, removal, and rollback of specialist capabilities without touching base weights.
  14. **On-Policy Knowledge Distillation:** Uses the prior checkpoint as a frozen teacher while training the candidate checkpoint on synthesized replay data.
  15. **Reinforcement Learning for Continual Improvement (GRPO / Verifiable Rewards):** Continual refinement guided by deterministic compiler and test case successes rather than human labelers.

- **Synthesis for HariAI:** 
  The optimal approach for HariAI Nano is a **hybrid of Decoupled Sleep Consolidation (Google Sleep / Sleep-Time Compute), Replay Buffering, and Modular LoRA Parameter Updates**. Base weights are frozen; sleep updates train compact low-rank adapters verified by deterministic regression benchmarks before promotion.
