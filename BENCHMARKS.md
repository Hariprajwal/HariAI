# HariAI evaluation status

The model weights, tokenizer, runtime, optional knowledge stores, peak RAM,
and training compute are separate quantities. The original 35,000,000-byte
research target is not met by the new approximately 92 MB standalone chat
model. See [the current model card](hari_ai/assets/MODEL_CARD.md) and
[research report](docs/research/STANDALONE_MODEL_PLAN.md) for the new work.

The local Open-JEV operation experiment scored 21/24 on a finite synthetic
development split (72 training records). Its majority baseline was 12/24.
Canonical field ordering fixed a field-order sensitivity found in the upstream
reference. The checkpoint is an advisory research component, not general
reasoning or an authorization mechanism. Its confidence head is untrained.

## What has actually been measured

- The 4,000-step arithmetic student scored 34/42 exact matches on a small held-out
  arithmetic set before quantization and 33/42 after quantization. Its quantized
  checkpoint is 581,380 bytes. This tests a restricted range of addition and
  subtraction; it does not measure general assistant capability.
- `benchmarks/evaluate_checkpoint.py` evaluates the actual checkpoint against
  `benchmarks/general_probe.jsonl` and records raw generations, dataset and
  checkpoint hashes, and exact checkpoint bytes. The arithmetic student scored
  1/8 on this public, hand-written smoke probe. It is far too small to estimate
  overall quality and can be contaminated if added to training data.
- The former 100% code-generation result measured four handwritten reference
  solutions. It did not evaluate model-generated code and is withdrawn. The
  legacy code-generation benchmark now explicitly reports `not_evaluated`.
- The legacy decision confidence error is mean absolute error per example, not
  expected calibration error. The INT4 size in the legacy system benchmark is
  an estimate, not a measured checkpoint file size. Previous throughput,
  calibration, and other release metrics lack reproducible artifacts and
  should not be relied on.

Example (choose a new output file for every run):

```powershell
python benchmarks/evaluate_checkpoint.py checkpoints/arithmetic_student_long_v1/hari_student_int4.pt --output checkpoints/arithmetic_student_long_v1/probe2.json
```

Before claiming broader progress, evaluate independently held-out language,
instruction following, factuality, reasoning, arithmetic and code tasks, with
the dataset versions, licenses, splits, prompts and scoring methods recorded.
Run generated code only in a properly isolated execution environment. Measure
quality before and after quantization on identical examples and report actual
files, hardware, latency, memory use and runtime dependencies.
