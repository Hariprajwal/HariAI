# Chat confidence and SmolLM investigation

Research and local checks, 8 November 2026. HariAI creator: K R Hari Prajwal.

## Findings

The default development GGUF is **SmolLM-135M-Instruct**, not SmolLM2 or a
model trained from scratch. Its file is 91,727,296 bytes. Its local Ollama
template uses ChatML delimiters; HariAI's `llama-cpp-python` backend uses the
same ChatML format. This checks the basic template, not every possible
implementation or sampling issue. The local model's code answers and errors are
recorded in [LIVE_CHAT_EVALUATION.md](LIVE_CHAT_EVALUATION.md).

The local runtime limits a response to 192 generated tokens. A long answer can
therefore stop partway through a program. The CLI now reports when the runtime
explicitly returns the `length` finish reason. Continuing an incomplete answer
does not guarantee that its code will compile or work.

The original [SmolLM model card](https://huggingface.co/HuggingFaceTB/SmolLM-135M-Instruct)
explicitly notes difficulties with arithmetic, editing and complex reasoning.
During this investigation, three different system prompts still produced invalid
or incomplete C binary-search programs. This small qualitative test does not
establish that prompt changes can never help, but it does not justify promoting
any of these prompt changes as a fix.

Ollama lists SmolLM 135M at about [92 MB](https://ollama.com/library/smollm),
while its smallest [Llama 3.1](https://ollama.com/library/llama3.1) variant is
8B parameters and approximately 4.9 GB in its listed package. Model size alone
does not determine quality, but they are not comparable deployments. The
original 35 MB target is not met by this SmolLM file and no ChatGPT-level result
has been established.

## What the confidence line means

Chat now shows an average base-model **token likelihood** in grey after a
supported local model response, explicitly labeled `not answer accuracy`.
It is the arithmetic mean of the model probabilities of its visible generated
tokens, before sampling adjustments. It is not calibrated answer correctness.
Verified arithmetic, where supported, is shown separately. The lightweight
observer is validated on llama-cpp-python 0.3.8 and abstains if token accounting
cannot match the displayed text or the runtime version is unverified. Turn the
line off with `hari settings confidence off`, or back on with
`hari settings confidence on`. The Python response text does not contain it.

The [Open-JEV reference](https://github.com/kyegomez/open-jev) is an unofficial
architecture with random upstream weights. The project's local trained Open-JEV
checkpoint was trained only on a bounded synthetic operation-choice task. Its
confidence head was not trained. Neither is a trained verifier of the correctness
of arbitrary chat responses; applying either to chat would invent a number.

Token likelihood means how likely a model found its next tokens, not directly
whether its complete answer is correct. [Research on calibration](https://arxiv.org/abs/2608.07827)
finds that confidence estimation and calibration are distinct and can vary
across models and datasets. A natural-language self-rating also requires
validation; it is not automatically a reliable correctness probability.

### Local calibration experiment

`scripts/evaluate_chat_confidence.py` generated 160 single-turn addition and
subtraction answers using the local GGUF and a 32-token generation limit. Each
question requested only the numerical answer; arithmetic answers were computed
independently. The operand pairs were split before inference into 112 calibration
and 48 held-out examples. No output from these prompts was used to train the model.

None of the 160 answers was exactly the requested integer. That is an
instruction-following and exact-format result, not a claim that all answers
contained incorrect arithmetic. `scripts/analyze_confidence_results.py` also
checks explicitly stated, unambiguous numerical results: it extracted a result
from 105 answers, of which 6 matched the independent answer. The other 55 could
not be scored by this simple parser. Only two of the 112 calibration answers
contained an extractable correct result, too few to fit the experiment's
predeclared minimum of five positives and five negatives. No numerical verifier
was installed from this experiment.

The geometric mean of token likelihood was evaluated on the 48 held-out
exact-format tasks. Its mean was 0.476 even though none matched the required
format; the 10-bin ECE was 0.476 and Brier score 0.237 against that strict
format target. These numbers describe only this small synthetic task and the
chosen sampling and generation settings. They do not measure general chat
accuracy or general-answer calibration, and a constant zero score would appear
perfect on this all-failure set without being useful on other tasks. The full
local report, prompts and answers are written to
`checkpoints/confidence_evaluation/report.json` (Git-ignored).

### Open-JEV answer-verifier experiment

`scripts/train_answer_jev_probe.py` trained the reference Open-JEV choice and
confidence heads for 400 updates on 1,600 synthetic arithmetic examples plus
74 extractable real answers from the calibration portion above. Only two of
the real calibration answers were correct. The synthetic operand pairs exclude
all pairs in the original 160-question run. A separate 31-answer held-out set
contained four correct answers and 27 incorrect ones. The verifier classified
27/31 correctly at a 0.5 cutoff; simply predicting incorrect for every answer
also gets 27/31. It identified one of the four correct answers. The head's
held-out ten-bin ECE for predicting whether the verifier's *choice* was correct
was 0.138; its Brier score was 0.100. The tiny, highly imbalanced sample makes
these numbers unstable and does not establish calibration or general-chat
performance. The head's target is confidence in the verifier's choice, not
directly a probability that an arbitrary chat answer is correct.

The checkpoint was **not promoted** or connected to chat. The local research
checkpoint and raw predictions are in `checkpoints/confidence_evaluation/`.

### Earlier Open-JEV integration and a larger independent run

An earlier version contained a fail-closed integration for a trained Open-JEV answer
verifier. For a supported answer, the verifier chooses `correct` or `incorrect`;
its own confidence-head output is oriented to the `correct` label and mapped
through a fitted Platt calibrator. This is an estimated probability for the
specific evaluated task, not a general-purpose confidence value. The runtime
also checks the language-model and verifier hashes, system prompt, generation
protocol, problem scope, single-turn context, and whether the answer finished.
No verifier result is used for other prompts, multi-turn chat, or other models.

`scripts/collect_jev_calibration.py` generated 1,200 new model answers with the
same local model, system prompt, sampling settings and 192-token limit used in
chat. Addition pairs were made canonical so `2 + 3` and `3 + 2` cannot cross the
predeclared calibration/test split. A deterministic arithmetic checker could
score 310 complete single-integer or single-equation answers. Of those, 20 were
correct: 9 in calibration and 11 in test. The other 890 answers were not scored
as correct or incorrect, because they were incomplete or did not match the
restricted answer format.

`scripts/train_calibrated_openjev.py` trained both the Open-JEV choice head and
its confidence head on independent synthetic problems and prior examples,
excluding all new evaluation problem groups. It fitted the calibration mapping
on 155 scorable calibration answers and evaluated it on 155 held-out answers.
The raw verifier-head score's held-out AUROC was about 0.509; after calibration,
the held-out Brier score was 0.066098 versus 0.066098 for the constant-rate
baseline (the model was slightly worse at full precision). There were fewer
than 20 correct examples in each split, and the model did not pass the Brier or
AUROC gates. Thus the candidate was not installed and no number is displayed.
Its raw data, verifier, metrics and artifact are in the Git-ignored
`checkpoints/jev_calibration_fresh_v2/` directory.

A second development experiment represented operands and answers as digit
fields before Open-JEV. It obtained held-out AUROC about 0.581, below the 0.60
minimum and still did not beat the Brier baseline. This split had already been
inspected for the earlier verifier, so this experiment is explicitly marked as
development, not an independent promotion test. It is not installed either.

The implementation requires at least 100 examples and 20 examples of each
class in **both** calibration and test, predeclared disjoint problem groups,
held-out ECE at most 0.10, Brier better than a constant baseline, and AUROC at
least 0.60. These checks are minimum safeguards, not a guarantee that a score
works outside its measured distribution. User-provided artifacts are trusted
local inputs; their stated source labels cannot be proven solely by the file.
The Open-JEV confidence path is no longer used by chat. Its experimental code
remains available for research, and old confidence-artifact settings are ignored.

## A path to actual answer confidence

1. Collect independently graded answers across relevant tasks, models and
   prompt styles. Keep training, calibration and final test examples disjoint.
2. Train a verifier using the actual user question and generated answer. If
   using an Open-JEV head, train its confidence outputs for this specific task;
   the operation-policy checkpoint cannot be repurposed without training.
3. Calibrate on held-out examples and report reliability plots, Brier score,
   expected calibration error, and results by domain. Test distribution shifts
   and abstain on unsupported or out-of-domain questions.
4. Display a percentage only when its stated interpretation and evaluation are
   supported. Code generation also needs separate sandboxed execution tests;
   a probability cannot replace compiling or testing a program.

Until then, the absence of a score is intentional. This does not claim that
the model knows which of its answers are correct.

### Semantic Open-JEV research on TruthfulQA

The earlier verifier used a randomly initialized hash encoder. A further
experiment reuses frozen hidden representations from the bundled SmolLM-derived
GGUF, projects them into a small Open-JEV readout, and trains the reference
implementation's binary decision and confidence heads. It adds no second
language-model weight file; the experimental verifier checkpoint is about
691 KB. This is a research architecture, not the unavailable TypeSafe Jev
production checkpoint.

The benchmark was the Apache-2.0-licensed
[TruthfulQA](https://github.com/sylinrl/TruthfulQA) CSV at commit
`d71c110897f5d31c5d7f309e7bc316c152f6f031`. The CSV has SHA-256
`b8d8ef1e12f98b4f2a9f47abc9765da0640b182b6c5d9b92f0c1a1f2f1e02e5c`.
The model weights were pinned by SHA-256. Of 790 distinct questions, 450 were
used for training, 103 for model selection, 118 for Platt calibration, and
119 for the held-out test. Each question was assigned to only one split;
calibration and test each contained one predetermined candidate per question,
alternating between the benchmark authors' correct and incorrect answers.
This creates a deliberately balanced distribution, not the correctness rate of
HariAI's own responses. Results may also reflect cues specific to the
benchmark's answer-writing style; exact question grouping does not rule out
all semantic or topic overlap between different questions.

On the held-out authored candidates, calibrated Brier was **0.1394**, the
constant-rate baseline was **0.2500**, ten-bin ECE was **0.0783**, and AUROC
was **0.8692**. Running the fixed seeded experiment again produced identical
results on this machine. The learned Platt slope was negative: the raw
confidence head was poorly oriented and required substantial correction.
These metrics show promise on this particular balanced benchmark, not that
chat-answer correctness has been calibrated.

To check the mismatch with real use, a separate exploratory run generated 100
answers to held-out questions using the local model and the normal 192-token
chat settings. Only exact matches to a benchmark correct or incorrect reference
were assigned labels. All 100 were unscorable under that strict rule; 75 stopped
at the token limit and 25 completed but did not exactly match a reference.
Unscorable does not mean incorrect. With zero independently graded generated
answers, neither calibration quality nor accuracy on HariAI chat can be
measured. The benchmark test questions were already inspected in the authored
candidate experiment and therefore cannot serve as a fresh promotion set.

`scripts/research_truthfulqa_jev.py` reproduces feature extraction and the
candidate experiment when supplied the pinned CSV and GGUF;
`scripts/research_truthfulqa_generated.py` reproduces the exploratory chat
generation and conservative exact-match grading. Local data, representations,
weights, model responses and JSON reports remain in Git-ignored paths. The
experimental checkpoint is explicitly marked non-deployable and is not loaded
by chat. A production confidence score still needs independently graded actual
responses across the intended tasks, a separate calibration split, a fresh
held-out evaluation, and an explicit out-of-scope policy.

As an additional exploratory check, a locally installed Qwen2.5 3B model was
used offline to judge the 25 completed generated answers against TruthfulQA's
references. It classified 21 as incorrect, three as correct, and one uncertain.
The judge is not ground truth: inspection found that two of the three answers
it marked correct merely repeated the question. Its results therefore cannot
be used as reliable labels or as a measured chat accuracy. On the 24 answers
for which the judge made a binary decision, the experimental verifier's mean
score was 0.647, whereas the judge marked 3/24 correct. That mismatch is a
warning about transfer and the judge's limitations, not a calibration estimate.
The judge and comparison are reproducible with
`scripts/research_truthfulqa_judge.py` and
`scripts/research_truthfulqa_compare.py`; neither the judge nor the experimental
verifier is used by HariAI chat.

### Model-native alternatives and exact verification

[P(True) self-assessment](https://arxiv.org/abs/2207.05221) asks the same
language model to judge a proposed answer; it requires no second model.
[Semantic entropy](https://www.nature.com/articles/s41586-024-07421-0) samples
multiple answers and groups them by meaning, but measures uncertainty rather
than truth and costs multiple generations. [Semantic entropy probes](https://arxiv.org/abs/2406.15927)
learn an inexpensive signal from a model's internal activations. Separately,
[research on learned uncertainty](https://arxiv.org/abs/2406.08391) trains on
graded answers and evaluates calibration and transfer. None of these papers
establishes that a method works for this small model or for arbitrary chat.

`scripts/research_native_self_assessment.py` tested native Yes/No logits using
the *same* language model on the previous, already-explored arithmetic set.
The held-out portion contained 155 scorable generated answers, 11 correct.
The model's raw self-assessment averaged about 90% while the observed fraction
correct was 7.1%; raw Brier was 0.759 and AUROC was 0.573. After fitting a
Platt mapping on the other split, Brier was 0.0660 against a 0.0661 constant
baseline; AUROC remained 0.573. With only 9 and 11 positives in the two splits,
and a previously explored test set, these figures are development evidence and
do not qualify the signal for deployment.

The current chat runtime therefore uses no Open-JEV confidence model. It can
verify a strictly parsed, complete, single-turn integer addition, subtraction
or multiplication answer exactly, with no external model or additional model
weights. `100% (verified arithmetic)` or `0% (verified arithmetic)` describes
the deterministic check, not a calibrated model-produced probability. It
abstains on extra claims, incomplete answers and multi-turn prompts. For other
chat responses, confidence remains uncalibrated. An internal learned probe
still requires independently graded generated answers, a suitable disjoint
calibration set and a fresh evaluation on the intended task distribution.

In an exploratory 30-question run of single-turn addition prompts using the
real model, 18 outputs were recognized as incorrect by the exact checker and
12 could not be checked; none was recognized as correct. For example, one
response stated `17 + 9 = 24`, which the checker marked incorrect. This small,
nonrepresentative run measures only these generated prompts and demonstrates
that verification does not improve the underlying language model's arithmetic.

As another development-only check, `scripts/research_native_confidence.py`
ran 100 questions from a pinned copy of Ai2's
[ARC-Easy](https://huggingface.co/datasets/allenai/ai2_arc) dataset using the
normal 192-token generation settings and asking for a single answer letter.
None of the 100 responses matched the strict one-letter answer format: 90
finished and 10 reached the token limit. The run therefore provides no
correctness labels or calibration metric; unscorable does not mean wrong.
The dataset is CC BY-SA 4.0, is stored only in the ignored local data directory,
and is not bundled with the package or used to train a released checkpoint.
