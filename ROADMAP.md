# Roadmap

HariAI is a research prototype. A checked item means code exists; it does not
claim production readiness or general intelligence.

## Foundation

- [x] From-scratch Transformer, tokenizer, training and INT4 export experiments.
- [x] Local Ollama chat integration and a clear unavailable-model response.
- [x] Model-size inspection and small, reproducible prompt probes.
- [ ] Train and release a broadly useful standalone model under 35 MB.
- [ ] Establish independent held-out language, reasoning, code and safety evaluations.
- [ ] Audit datasets and teacher outputs with reproducible provenance.

## Dream Engine

- [x] Offline LoRA candidate experiment with replay and measured regression checks.
- [x] Opt-in local memory add, inspect and delete commands.
- [ ] Persistent and auditable candidate history, promotion and rollback.
- [ ] Robust idle scheduling, resource limits, pause and preemption.
- [ ] User-approved learning from private experiences with retention controls.
- [ ] Broader evaluation and evidence that a candidate retains prior capabilities.
- [ ] Safe, authorized cluster knowledge sharing.

## Distribution

- [x] Public source repository and commercial licensing contact.
- [ ] Reproducible public model artifact and verified dataset permissions.
- [ ] Hardware benchmarks, optimized inference and installation packages.

Details and measured results: [Dream Engine status](docs/research/DREAM_ENGINE_STATUS.md),
[research plan](docs/research/35MB_GENERAL_ASSISTANT_PLAN.md), and
[benchmarks](BENCHMARKS.md).
