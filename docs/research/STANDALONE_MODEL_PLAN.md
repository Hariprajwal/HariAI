# HariAI standalone model: research and implementation

Research date: 8 November 2026. Creator: K R Hari Prajwal.

## What "our own model" means here

The deployable artifact must contain the weights it needs. It must run when
Ollama is stopped, the original model cache is absent, and the network is off.
A merged fine-tune meets that deployment requirement. It remains a derivative
of its pretrained foundation, whose authorship and license must stay visible.
It is different from a new foundation model trained from random initialization.

HariAI supports a local GGUF file on the development machine. A model-bearing
wheel is planned but has not been built or published because the candidates
failed quality gates. The training script merges low-rank updates into a
complete independent file. Inference from that file needs no adapter service,
Hugging Face login, or base-model download.

## Why the previous Nano failed

The earlier sub-megabyte model was trained on a tiny repeated curriculum. Its
0/8 mixed probe and whitespace output were model-quality failures. Changing the
CLI backend did not supply missing language knowledge. In comparison,
[SmolLM2-135M's model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
reports 2 trillion pretraining tokens. The
[SmolLM2 paper](https://arxiv.org/abs/2502.02737) emphasizes data composition,
training duration, and staged instruction training. These are substantial
training inputs, not properties obtained by renaming an architecture.

The locally available foundation is **SmolLM-135M-Instruct**, not SmolLM2.
Its [model card](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct)
identifies Hugging Face as its publisher and Apache 2.0 as its license.
The exact local quantized source is identified by SHA-256 in the model manifest.
The attempted newer SmolLM2 download was not executed because approval review
could not run after the workspace spending cap was reached.

## Size accounting

135 million parameters at exactly four bits each would occupy 67.5 million
bytes. Real files also contain scales, higher-precision tensors, tokenizer
tables and metadata. The installed SmolLM GGUF occupies **91,727,296 bytes**.
The pinned SmolLM2 Q4_K_M artifact inspected on Hugging Face is
**105,454,432 bytes**. These are model files, not peak RAM or installed dependency
sizes. PyTorch and the compiled inference runtime require additional space.

35 MB would allow at most 70 million weights at ideal four-bit storage before
overhead. This arithmetic is not a prediction of quality. Moving the runtime to
Rust or Go can change execution and distribution costs; it does not by itself
compress learned weights or replace pretraining.

[BitNet b1.58](https://arxiv.org/abs/2402.17764) investigates trained ternary
weights. It is a serious compression research direction, but the paper does not
establish that converting this already trained 135M checkpoint into 35 MB will
preserve its capabilities. That needs a separate training and evaluation study.
This release must not advertise the old 35 MB or 45 MB goal as achieved.

## What combining SmolLM and Open-JEV can actually do

[Open-JEV](https://github.com/kyegomez/open-jev) is an unofficial reference
implementation. The upstream repository explicitly provides random weights and
a hash tokenizer, not TypeSafe AI's production checkpoint. Its state encoder
and typed readouts differ from the causal language decoder in SmolLM. Adding
their weight tensors is undefined, and shipping two files does not establish
an improvement in conversation.

The concrete integration has two components in the same package:

- A complete SmolLM-derived GGUF handles language generation.
- The attributed Open-JEV implementation supports separately trained typed
  decisions. The bounded operation-policy experiment uses a deterministic
  synthetic teacher and evaluates unseen combinations of structured inputs.
  It does not assess free-form questions, improve chat, enforce permissions,
  or calibrate the truth of language-model answers.

The longer-term way to reduce duplicated weights is to investigate sharing
pretrained language representations with typed query/readout heads. That requires
a designed attention interface and joint or staged training. It is a proposed
experiment, not an implemented fused model or a discovered breakthrough.

## Training and evidence

`scripts/adapt_smollm.py` performs a small identity adaptation inspired by
[LoRA](https://arxiv.org/abs/2106.09685). Base parameters are frozen, attention
updates are trained, and the updates are merged before export. Training uses
36 original examples in `scripts/adaptation_data.py`, with identity examples
sampled more often and basic instructional replay. This is enough to test an
adaptation mechanism, not to claim general instruction-tuning coverage.

The first 120-step candidate failed identity probes and was not promoted.
A stronger candidate adapted identity, but its exported model regressed on general prompts and was rejected. Reports record
the exact source hash, output hash, bytes, trained parameter count, validation
loss, and generated answers. These few prompts are development probes. Keyword
matches can pass even when surrounding text is wrong; they are not accuracy
scores for a general assistant. A separate fresh prompt report showed both baseline and candidate at 0/12 strict exact matches; the candidate often repeated creator identity instead of answering. See [live conversations](LIVE_CHAT_EVALUATION.md). Broader benchmarks remain needed.

For Open-JEV, `scripts/train_operation_jev.py` splits exact structured records
before training and checks held-out accuracy plus invariance to field/option
ordering. Its neural model is compared against the majority baseline. The
deterministic labeling policy is already an exact solution to this small task;
the neural experiment demonstrates integration rather than superiority over it.
The confidence head is untrained and is not exposed by the bounded-task API.

## What would improve general capability next

First establish a stronger pretrained baseline and evaluate representative
conversation, instruction following, arithmetic, factuality, and code behavior.
Then perform supervised adaptation on substantially more diverse, attributed
instruction data, preserving held-out tests. Multi-teacher distillation can
provide additional targets, but teachers' errors, license terms and duplicated
evaluation data need explicit handling. Teachers are training tools; the
released student can still run without them.

### Findings from the live tests

The [20-question conversation transcript](LIVE_CHAT_EVALUATION.md) shows that
the baseline often fails to follow basic instructions: it did not answer
12 + 7, confused Moon phases with an eclipse, retained subtraction when asked
to fix an addition function, and fabricated details in a summary. The identity
fine-tune learned creator responses but also emitted creator information for
unrelated arithmetic and factual questions. This is consistent with
overfitting or interference from an extremely small, identity-heavy adaptation
set; the evidence does not prove a particular architectural defect.

To distinguish these causes, the next controlled experiment should:

1. Verify that training and runtime use matching chat templates, special tokens,
   loss masking and stopping rules, and compare unmerged PyTorch and exported
   GGUF outputs on the same prompts. Evaluate quantization separately.
2. Compare the original base and adaptation on a fixed, held-out, diverse set
   with independent arithmetic answers, executable code tests in an appropriate
   sandbox, factual questions, multi-turn dialogue, and instruction following.
   Record raw answers and score each capability separately. The 12 strict
   exact-match probes here measure format as well as content.
3. Add substantially more varied, properly licensed general instruction data
   and replay; keep creator examples a small portion and include non-identity
   prompts to test whether the identity answer leaks into unrelated tasks.
   Compare several adaptation strengths against the unchanged base and reject
   any candidate that regresses on held-out capabilities.
4. Compare the installed SmolLM 135M with a stronger, appropriately licensed
   foundation under the same prompts, runtime and precision. A larger model or
   increased model-storage budget may be needed; quality gains cannot be
   inferred from parameter counts alone.

The current comparison is a development investigation, not a proof that a
different architecture will reach ChatGPT quality at 35 MB. Open-JEV's typed
heads also do not independently establish correctness of generated prose.

Compression must be measured after export, with tokenizer and auxiliary weights
counted. Compare each candidate against its unmodified foundation and reject
regressions. For coding tests, execute only reviewed tasks in a suitable sandbox;
do not equate syntactic code output with correctness.

The [Google Sleep framework](https://research.google/blog/the-sleep-framework-a-new-way-for-llms-to-retain-knowledge-in-continual-learning/)
addresses retention during continual learning. It does not supply unlimited
self-improvement or missing pretrained knowledge. HariAI's Dream command remains
inspection-only; background training, privacy controls, scheduling, promotion
and rollback remain separate work. Chat is not automatically used as training data.

## Packaging and release

The model-bearing wheel should be checked from an installation outside the
source tree. It must include the merged GGUF, manifest, model card, upstream
licenses, and attributed Open-JEV code. The build script validates hashes and
sizes before creating an artifact. No live service is required for chat.

Publication follows the [Python Packaging guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).
Creating a wheel locally is not publishing on PyPI. Package-name availability,
account ownership, authentication and the intended release version must be
established before uploading. A small passing unit-test suite verifies code
behavior; it does not certify model quality.
