# HariAI: research plan for a general assistant under 35 MB

## Goal and boundaries

Develop a useful general assistant whose serialized model weights are strictly
smaller than 35,000,000 bytes. Tokenizer, executable, working memory and any
external index must be reported separately. General ability and parity with any
specific assistant are hypotheses requiring independent evaluation; neither is
guaranteed by a parameter count or architecture.

Existing best measured student: 581,380-byte checkpoint, 33/42 on a narrow
arithmetic split and 1/8 on a small mixed smoke probe. Those results do not
demonstrate general conversation.

## Stages and promotion gates

1. **Measurement.** Preserve versioned evaluation sets for ordinary dialogue,
   instruction following, factual questions, arithmetic, code and uncertainty.
   Separate unseen evaluation data from training. Log raw generations, actual
   weight bytes, tokenizer hash, memory and latency. Retire any measurement that
   scores reference code instead of model output. Current eight-item probe is a
   smoke check, not a reliable general benchmark.
2. **Language foundation.** Compare random-initialized 5–20M parameter models
   with a shared, license-audited training corpus and fixed compute. Begin with
   a small reproducible pilot to validate language loss and generation, then
   scale data and compute. A 35 MB limit permits substantially more parameters
   than the current 0.84M experiment. Compare byte and modest BPE vocabularies,
   accounting for embedding bytes and throughput. Save train and held-out splits
   by document, and track sources and dataset revisions.
3. **Instruction training and distillation.** Use short, diverse dialogues and
   licensed teachers; check arithmetic and code with independent oracles where
   safe isolation exists, and sample other answers for human review. Compare
   a matched non-teacher control. Train response tokens, retain general language
   replay, and reject runs that regress on held-out tasks.
4. **Compression at equal quality.** Compare FP32, INT4 and a properly trained
   ternary experiment, including embeddings and metadata. Measure each exported
   and reloaded checkpoint against identical held-out tasks. Enforce the exact
   byte ceiling. Do not label 4-bit-packed ternary values as 1.58-bit storage.
5. **Systems capability.** Test retrieval and deterministic tools as optional
   separately sized components, with model-only and augmented scores, latency,
   index size, and error cases reported. A calculator or retrieval store can
   supplement a model but does not move their information into the weights.
6. **Runtime.** Profile CPU and the available AMD GPU. Compare a supported
   optimized inference backend with the existing implementation only after a
   quality baseline is established. Rust, Go, C++ and Lean cannot by themselves
   improve the information encoded by the weights. Lean can check actual formal
   statements and proofs, not automatically validate arbitrary language output.

## First experiment and decision rule

Obtain a small, version-pinned, attributable sample of educational text, train
an isolated student from random initialization, and measure held-out language
loss before and after training. Use an independent prompt probe to record actual
generations before and after quantization. A falling language loss is evidence of
learning text distribution only; instruction following requires separate tuning.
Keep the existing checkpoints and do not promote the new model automatically.
Record failed runs and rejected datasets as well as successes.

### Pilot result

- A six-layer, 192-wide student was initialized randomly and trained for 1,200
  updates on 360 FineWeb-Edu documents (1,228,800 byte tokens processed). Forty
  other documents were held out by document hash. The held-out language loss
  fell from 5.714 to 1.855 nats per byte token; the INT4 export scored 1.858.
  The 1,473,688-byte checkpoint scored 0/8 on the mixed smoke probe. This shows
  limited text-distribution learning, not general conversation.
- A tokenizer pilot trained 128 BPE merges on 20 training document prefixes.
  On 20 different document prefixes it used 10,401 tokens versus 18,150 byte
  tokens, 42.7% fewer in this tiny sample. At width 192 the extra FP32 embedding
  rows cost 98,304 bytes; the tokenizer JSON is 28,024 bytes. This indicates a
  potentially useful context-efficiency tradeoff, not an assistant-quality gain.
  A model must be trained with the new tokenizer before it can be deployed.
- The short complete-dialogue filter retained 23 examples out of 2,000 sampled
  rows. That is too little for meaningful instruction tuning. A subsequent data
  request encountered HTTP 429. Work should use a robust, resumable data pipeline
  and a larger context/tokenizer before treating dialogue tuning as validated.

Artifacts: `checkpoints/language_pilot_v1/report.json`,
`checkpoints/language_pilot_v1/general_probe.json`, and
`checkpoints/tokenizer_pilot_v1/report.json`. These files and the sampled data
are local and Git-ignored; retain them separately when reproducing this run.

Published reference points: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
[SmolLM](https://huggingface.co/HuggingFaceTB/SmolLM-135M),
[TinyStories](https://arxiv.org/abs/2305.07759),
[BitNet b1.58](https://arxiv.org/abs/2402.17764), and
[pruning with distillation](https://arxiv.org/abs/2407.14679).
