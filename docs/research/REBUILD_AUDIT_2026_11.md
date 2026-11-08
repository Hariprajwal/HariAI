# HariAI: independent model rebuild audit

Research date: 8 November 2026. Project creator: K R Hari Prajwal.

## Current state and gaps

| Area | What exists | Important gap |
| --- | --- | --- |
| Default chat | Local llama.cpp inference using a 91,727,296-byte GGUF derived from SmolLM-135M-Instruct. It does not require Ollama. | This is a pretrained derivative, not a foundation model trained from random initialization. Its general-answer accuracy and confidence are limited. |
| Scratch model | `HariNanoModel` has randomly initialized embeddings, decoder blocks, GQA, RoPE, RMSNorm, SwiGLU and a language-model loss. | Its trained research checkpoints are small pilots, not capable general assistants. The decision sigmoid is not calibrated merely because it exists. |
| Training | The repository contains real PyTorch optimizers, a byte-tokenizer language pilot, checkpoint save/load, and a small teacher-answer experiment. | The pilot's tokenizer is byte-only; the trainer does not yet provide a complete, reproducible large-corpus, deterministic resume and evaluation pipeline. Saved state does not include all RNG and data-loader progress. |
| Data | Local FineWeb-Edu sample has document hashes, source information and a document-level holdout. The pilot now verifies text hashes and rejects exact duplicates. | Near duplicates, source-level leakage, per-source permissions, data quality and benchmark contamination need broader controls. |
| Distillation | A separate random student experiment used a small number of independently checked teacher arithmetic examples. | It is not a broad teacher-distilled assistant and no side-by-side, matched-budget general evaluation exists. |
| Quantization | Experimental linear-layer INT4 packing saves real files; the research runtime can load them. | Embeddings and some other tensors stay floating point, linear layers dequantize during forward, and the custom checkpoint is not a production GGUF export. An ideal bit count is not the actual model file size. |
| Dream and memory | Explicit local memory and manual research candidate/rollback code exist. | Automatic idle training, independently validated promotion, private-data controls for training, and durable production rollback are not implemented. |
| Terminal, tools and cluster | A terminal, CPU local chat, manual search, memory commands and prototype cluster code exist. | The supplied screenshot is a design goal: it is not evidence that background learning, multiple live agents, GPU use or autonomous tool execution are active. `hari status` reports the observable state. |

The prototype Nano runtime has a DirectML enum but no corresponding detection
or execution path; it now explicitly rejects DirectML rather than silently
running on CPU. The MPS device mapping has been corrected, but MPS execution
has not been hardware-tested here and should not be advertised as verified.

The older `scripts/train_custom_llm.py` includes scripted examples that make
unsupported claims about capabilities. Do not use it as a general instruction
corpus without replacing and auditing those examples. The older Phase 0
licensing table also contains overbroad claims and is marked historical.

## Candidate 125M architecture and size

Start with a decoder-only research candidate: **16,384 vocabulary, width 576,
32 layers, 9 attention heads, 3 KV heads, 1,600-wide SwiGLU feed-forward,
tied input/output embeddings, RoPE and RMSNorm**. Train language modeling
first; the small optional decision head remains untrained until separately
supervised and evaluated. This is a candidate for controlled experiments, not
an architecture proven best for HariAI. The code's current decoder attention
mask now also supports cached chunks containing multiple tokens. The proposed
16K tokenizer has not yet been trained: its smaller embedding budget trades
against longer sequences, code and multilingual coverage and must be measured.

`python scripts/parameter_budget.py` calculates counts without allocating the
full model. Its formula was checked against actual small model instances.

| Component | Parameters |
| --- | ---: |
| Token embeddings, shared with output | 9,437,184 |
| Attention projections | 28,311,552 |
| SwiGLU feed-forward layers | 88,473,600 |
| Layer norms | 37,440 |
| Optional decision head | 19,586 |
| **Total** | **126,279,362** |

At an ideal four bits for every parameter, weights alone require
**63,139,681 bytes**; actual scales, tokenizer and container metadata add more.
The ideal three-bit floor is **47,354,761 bytes**, before those costs and with
unknown quality. Therefore, 125M parameters and a complete under-50-MB file
are separate goals that require an evaluated compression experiment. A smaller
69,344,770-parameter comparison has an ideal Q4 floor of 34,672,385 bytes,
also before overhead; it is not proof of a sub-35-MB deployable model.

[SmolLM2's published configuration](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/raw/main/config.json)
and [MobileLLM](https://arxiv.org/abs/2402.14905) motivate testing deep, narrow
decoders and GQA. Compare these candidates at matched data and compute before
selecting a final architecture. No claim of novelty or equivalent quality is
made from matching architectural features.

## Two separate experiments and a baseline

1. **Scratch:** randomly initialize every parameter, train next-token prediction
   on license-reviewed documents, then instruction-tune on a distinct approved
   corpus. Save tokenizer, model config, data and code versions, random seeds,
   optimizer, scheduler, RNG states and sampler position. Evaluate a separate
   fixed document split and task benchmarks.
2. **Teacher-distilled:** start a second student from random initialization.
   Generate versioned examples from a specifically identified, permitted
   teacher. Verify math answers and execute code only in an appropriate isolated
   evaluator. Compare short-answer and reasoning-trace data, with the same base
   architecture, compute budget and held-out evaluations as the scratch run.
   Teacher labels and any judge outputs remain fallible.
3. **SmolLM baseline:** retain the existing SmolLM-derived local GGUF as an
   explicitly attributed baseline. A merged adapter or rename does not make it
   a scratch-trained foundation model.

For future typed decisions, [AnyJev](https://github.com/nokia-applied-research/AnyJev)
reads option probabilities from a causal model and studies rotation and label
bias; its published benchmarks do not establish calibration of HariAI's chat
answers. Our earlier simple four-rotation experiment is not a reproduction of
its full algorithm and did not establish useful discrimination on HariAI.
[Laya's model card](https://huggingface.co/convaiinnovations/laya) describes a
421M-parameter encoder plus decision head, a separate model whose ideal four-bit
weights alone exceed the size target. These are research references, not new
runtime dependencies or claims of equivalence to TypeSafe's Jev.

Pipeline: **source/rights registry → content validation and deduplication →
document-level split → train-only tokenizer → tokenized shards → scratch
pretraining → held-out validation → checkpoint → instruction/teacher training
→ task evaluations → quantization → post-quantization regression and size
checks → explicitly selected release**. Research artifacts never automatically
replace the running chat model.

## Candidate data and teacher terms

* The [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
  lists ODC-By 1.0 and Common Crawl terms. Its underlying pages have their own
  provenance and rights. The existing sample is used locally for a pilot;
  review attribution and applicable rights before further use or redistribution.
* [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) lists
  CDLA-Sharing-1.0. It is useful for a language-learning pilot, not a sufficient
  general assistant corpus; review the exact version and source terms.
* [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) and
  [Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) list Apache 2.0; pin and
  review the exact weights, dataset and synthetic-output use before training.
  **Qwen2.5-3B-Instruct is different:** its [official license](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct/raw/main/LICENSE)
  is a Qwen Research license that limits use of the materials to noncommercial
  purposes without a separate license. Do not rely on a family-wide claim.
* [Gemma 2 terms](https://ai.google.dev/gemma/terms) and the specific
  [Llama 3.1 license](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
  need individual review, including restrictions on derivatives and outputs.
  A model's license does not automatically establish the rights of a dataset.
* The Stack v2 contains source files with [multiple original licenses and
  access conditions](https://huggingface.co/datasets/bigcode/the-stack-v2);
  it is not a single Apache-licensed corpus. Filter and retain file-level
  provenance before considering it. Hold out task benchmarks such as
  [GSM8K](https://huggingface.co/datasets/openai/gsm8k) from training if used
  for independent evaluation.

## Hardware and feasibility

At inspection, this machine reported **16 logical CPU threads, 23.34 GiB
total RAM, about 2.68 GiB available RAM**, and about 956.64 GiB free disk.
PyTorch 2.5.1 reported `cuda.is_available() == False`. A local DirectX report
identified integrated AMD Radeon graphics and a **Radeon RX 5500M** with WDDM
3.1, but did not report a reliable dedicated-memory value. The current PyTorch
training run and llama.cpp configuration used the CPU; AMD acceleration requires
its own compatibility and numerical validation. Available memory changes with
other programs.

For the 126M candidate, FP32 parameters, gradients and two Adam moments alone
are roughly **2.02 GB** before activations, temporary tensors, data, runtime
and checkpoint overhead. It is not safe to infer that a full-size training run
will fit in the currently available memory. Training time must be measured on
the target hardware and data; the tiny pilot below cannot predict the speed or
quality of a trillion-token run. [SmolLM2's model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
reports 2 trillion pretraining tokens and 64 H100 GPUs for its training setup.

Use PyTorch for experiments and the existing optimized llama.cpp runtime for
the GGUF baseline. Rust or Go can be reconsidered for measured orchestration,
packaging or deployment bottlenecks; changing language alone neither compresses
weights nor improves a model's learned knowledge. Lean is appropriate for
specific formal-verification tasks, not as a general training accelerator.

## First new scratch pilot: actual result

The 120-step run at `checkpoints/scratch_audit_120_v1/` initialized a
**2,414,786-parameter** model from seed `20261108`, used 360 training and 40
held-out documents from the existing FineWeb-Edu sample, processed 46,080
training byte tokens, and never loaded SmolLM weights. The saved FP32
checkpoint is 9,682,282 bytes. A seeded reconstruction confirmed the trained
embedding changed from initialization; the training script and data hash are
recorded in `report.json`.

A second, separate 20-step timing smoke run with the same tiny architecture
and batch/sequence settings recorded about **4,656 byte tokens/second** during
the training loop on CPU. This is one short measurement for the 2.4M model;
it is not a throughput estimate for the 126M design.

Held-out byte-token loss decreased from **5.6288** to **2.6253** nats; after
experimental INT4 packing it was **2.6259**. The quantized checkpoint was
**1,473,688 bytes**. On an eight-question public exact-match smoke probe it
scored **0/8**, often producing repeated text. The byte-token loss is not
directly comparable with subword-token model losses, and this tiny run is
neither an instruction-tuned assistant nor evidence of general competence.
The holdout and smoke questions have been used by earlier experiments; they are
development checks, not a fresh independent final test. Artifacts remain local
and Git-ignored; the default chat model was not replaced.

## Milestones and acceptance criteria

1. Harden data provenance, near-deduplication, tokenizer training and packing;
   verify train/eval/source separation and deterministic resume.
2. Run a matched scratch-versus-distilled pilot and keep the SmolLM baseline
   separate. Record seeds, weights, dataset versions, learning curves and cost.
3. Evaluate language, instruction following, code execution, math, factuality,
   refusal behavior and quantization regression on fresh independent sets.
4. Train and evaluate uncertainty and typed decisions against explicit labels;
   never call an untrained sigmoid or token likelihood calibrated correctness.
5. Add staged, opt-in Dream candidates, privacy controls and rollback only after
   those evaluation gates exist. Test CPU packaging and optional acceleration
   before any release or claims resembling the supplied terminal mock-up.
