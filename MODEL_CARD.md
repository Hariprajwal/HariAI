> **Historical research document:** Some capabilities and size claims below are unverified or obsolete. See [README.md](README.md) and [the live evaluation](docs/research/LIVE_CHAT_EVALUATION.md) for current results.

---
language: en
license: other
pipeline_tag: text-generation
library_name: hari-ai
---

# HariNano model card

HariNano is a research architecture for small, from-scratch language models.
There is no generally capable chat checkpoint bundled in this repository. The
goal of a model under 35,000,000 bytes is an experimental target, not a claim of
frontier-model performance.

The local language pilot trained a six-layer, 192-wide model for 1,200 updates on
360 educational documents and assessed loss on 40 held-out documents. Its INT4
file was 1,473,688 bytes and scored 0/8 on a small mixed exact-match prompt
probe. A separate arithmetic student and a rejected Dream candidate are
documented in [BENCHMARKS.md](BENCHMARKS.md) and
[DREAM_ENGINE_STATUS.md](docs/research/DREAM_ENGINE_STATUS.md). These small
probes are not representative benchmarks for language, code or factuality.

Do not rely on these research checkpoints for general advice, safety decisions,
or accurate facts. The repository does not include the sampled data or weights;
data sources and scripts are described in [DATA_LICENSES.md](DATA_LICENSES.md).

Current repository software licensing is described in [LICENSE](LICENSE).
Earlier Apache 2.0 grants and third-party model or dataset terms continue to
apply where relevant. Consult those sources for rights in weights or data;
the software license does not automatically license third-party materials.
