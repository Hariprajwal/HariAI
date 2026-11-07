# Project status

**Stage:** Research prototype. **Updated:** November 2026.

- Small Transformer, tokenizer and research training scripts: implemented.
- Local Ollama chat: implemented; requires a separately installed model and service.
- General-purpose standalone assistant under 35 MB: not yet demonstrated.
- Manual Dream candidate training with replay and held-out checks: experimental;
  the first measured candidate was rejected for language regression.
- Persistent local memory: explicit add, inspect and delete; plaintext, local,
  and excluded from model training.
- Automatic improvement, idle scheduling, interruption-safe training, full
  privacy controls and production checkpoint rollback: not implemented.
- Model and data artifacts: local and Git-ignored; not included in the repo.

See [README.md](README.md), [BENCHMARKS.md](BENCHMARKS.md), and
[Dream Engine status](docs/research/DREAM_ENGINE_STATUS.md) for evidence and
limitations.
