# Dream Engine: current research status

The attached design calls for opt-in learning from approved experiences, offline
candidate training, replay, independent evaluation, rollback and resource-aware
scheduling. These remain research goals. The interactive Dream controller now
performs inspection only; it neither invents a checkpoint nor marks chat output
as a verified procedure. Conversation data is not fed into the experiment below.

## First measured offline candidate

`scripts/dream_arithmetic_candidate.py` uses checked teacher examples and
deterministic arithmetic replay. It restores the existing arithmetic student,
freezes its base parameters, trains low-rank attention adapters, evaluates on
the original held-out arithmetic split and separate held-out educational text,
merges the candidate, quantizes and measures the exported file. It writes a new
candidate and report to a fresh directory; it never promotes the candidate.

Run on 7 November 2026, seed 42, 200 updates:

- Qwen: 13 checked training examples admitted. SmolLM: all 12 attempted calls
  returned HTTP 500, so no SmolLM answers were admitted. These are tiny samples, not
  comprehensive teacher comparisons. Local SmolLM model layer SHA-256:
  `eb2c714d40d4b35ba4b8ee98475a06d51d8080a17d2d2a75a23665985c739b94`.
- Held-out arithmetic: FP32 base 34/42; quantized base 33/42; FP32 candidate
  36/42; quantized candidate 35/42. The comparison reuses a previously measured small split;
  it is useful for regression tracking, not an unbiased generalization claim.
- Held-out language loss: FP32 base 12.584 nats/token; quantized base 12.641;
  quantized candidate 12.726. The allowed increase from the quantized base was
  0.05 nats, so the candidate **failed** the
  gate. Its file is 581,146 bytes, within 35,000,000 bytes.
- No active checkpoint changed. The base checkpoint remains available for
  rollback. See `checkpoints/dream_candidate_v2/report.json` for all answers.

The gate compares the quantized candidate against the quantized base. The
candidate is retained for inspection, not presented as an improvement. A stronger
evaluation must include more held-out examples, conversation, coding and
factuality. LoRA freezing protects base parameters during fitting but does not
guarantee retention of behavior after merging.

## Further work

1. Build larger, licensed, independently checked teacher data and a held-out
   general assistant suite; use review or independent validators for claims
   that cannot be mechanically scored. Never run generated code directly on
   the host or call formatting checks factual verification.
2. Add persistent, versioned candidate history and a real rollback pointer,
   explicit opt-in for learning from private experiences, deletion controls,
   per-user isolation, and resource-aware pause/resume.
3. Compare replay and LoRA against matched no-replay and frozen-model controls,
   measuring old and new capabilities and serialized model-plus-adapter bytes.
4. Add configurable idle scheduling only after the candidate workflow can be
   interrupted safely and audited. Automatic self-improvement is not currently
   implemented and should not be claimed.

Background: [Google Sleep Framework](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/),
[Nested Learning](https://research.google/blog/introducing-nested-learning-a-new-ml-paradigm-for-continual-learning/),
[Sleep-time Compute](https://arxiv.org/abs/2504.13171), and
[self-generated replay research](https://arxiv.org/abs/2605.26097). These works
do not prove this project's candidate retains general capabilities.
