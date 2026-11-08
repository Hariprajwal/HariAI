# HariAI model and training provenance

This is a factual record of sources used in the local experiments, not a blanket
legal clearance for future models or datasets. Creator: K R Hari Prajwal.

## Standalone chat adaptation (8 November 2026)

- Foundation: [HuggingFaceTB/SmolLM-135M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct).
- Local quantized source: SHA-256 `eb2c714d40d4b35ba4b8ee98475a06d51d8080a17d2d2a75a23665985c739b94`, 91,727,296 bytes.
- The source was read from an existing local model cache and copied into the
  workspace. Runtime inference does not access that cache or Ollama.
- The accompanying Apache 2.0 license is preserved at `hari_ai/assets/LICENSE-SMOLLM`.
- Adaptation data: 36 examples authored for this task in `scripts/adaptation_data.py`,
  plus eight development validation examples. No private conversation logs,
  saved memories or downloaded web corpus were added to this adaptation run.
- Training code writes the actual output hash and statistics. Model modifications
  and the original foundation must be attributed separately.

## Open-JEV operation experiment

- Architecture: [kyegomez/open-jev](https://github.com/kyegomez/open-jev),
  revision `93843ef288f3d35fd91263377c40b6f917d77da0`, Apache 2.0.
- Vendored source and license are under `hari_ai/vendor/open_jev/`.
- The public reference ships random weights. HariAI trains its own small
  operation-policy checkpoint using `scripts/train_operation_jev.py`.
- The synthetic teacher is an explicit deterministic function in that script.
  It produces 96 finite-schema records, divided into 72 training and 24
  development test records before training. Labels are smoothed for training;
  they are not empirical teacher-ensemble probabilities.
- The confidence head is not trained or represented as calibrated. The task is
  advisory and cannot replace permission checks or validate arbitrary answers.

## Earlier experiments retained for research

The arithmetic Dream experiment used checked outputs from a locally installed
Qwen 3 4B model. Its model-layer SHA-256 was
`3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f`;
its accompanying license layer contained Apache 2.0. The experiment admitted
13 verified teacher examples and did not promote its candidate after a language
regression. See [Dream status](docs/research/DREAM_ENGINE_STATUS.md).

The earlier language pilot sampled 400 rows of
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
`sample-10BT`. Its dataset card lists ODC-By 1.0. That label does not independently
establish rights in every underlying webpage. Sample records retained source
URLs, row identifiers and content hashes locally. The sampled data is not part
of the standalone chat adaptation or its package.

## Before using additional sources

Check the exact model revision, dataset version, applicable terms, attribution,
and redistribution requirements. Preserve provenance and keep evaluation data
separate. Do not infer permission from a model family name. Earlier broad lists
of "approved" and "prohibited" models were not a source-by-source legal review
and should not be treated as one.
