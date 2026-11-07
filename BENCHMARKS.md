# HariAI — Benchmarking Suite & Target Metrics

This document defines the quantitative evaluation suite, baselines, and empirical performance metrics for HariAI.

---

## 1. Core Evaluation Principles
1. **Zero Hallucinated Numbers:** Never report estimated or synthetic benchmark scores as empirical findings.
2. **Reproducible Test Harness:** All benchmark datasets, prompts, evaluation scripts, and log dumps must be version-controlled in `benchmarks/`.
3. **Contamination Auditing:** Pre-training corpora must be scrubbed against test sets with 13-gram overlap filtering.

---

## 2. Benchmark Suites by Capability Domain

| Capability Domain | Primary Benchmark | Metric | Baseline Comparator (SmolLM-135M) | Target for HariAI Nano (<50MB) |
|---|---|---|---|---|
| **Mathematical Reasoning** | GSM8K (8-shot / CoT) | Accuracy (%) | ~12.5% | $\ge 15.0\%$ |
| **Code Generation** | HumanEval (0-shot) | Pass@1 (%) | ~9.8% | $\ge 12.0\%$ |
| **Elementary Code & Syntax** | MBPP (3-shot) | Pass@1 (%) | ~14.2% | $\ge 16.5\%$ |
| **Scientific & Logic Reasoning** | ARC-Challenge (25-shot) | Accuracy (%) | ~28.4% | $\ge 30.0\%$ |
| **General Commonsense** | HellaSwag (10-shot) | Accuracy (%) | ~38.0% | $\ge 40.0\%$ |
| **Decision & Action Selection** | Synthetic ToolBench Subset | Action Accuracy (%) | N/A (Standard LLMs fail format) | $\ge 85.0\%$ |
| **Action Confidence Calibration** | Calibration Test Set | Expected Calibration Error (ECE) | N/A | $\le 0.08$ |

---

## 3. Systems & Efficiency Benchmarks

| System Metric | Hardware Platform | Target Metric |
|---|---|---|
| **Model Weight File Size** | Disk Storage | $\le 50.00\text{ MB}$ (Current target: $\approx 26.0\text{ MB}$ - $38.3\text{ MB}$) |
| **Inference Latency (CPU Single Thread)** | Intel Core i7 / AMD Ryzen (AVX2) | $\ge 35\text{ tokens/sec}$ |
| **Inference Latency (ARM64)** | Apple Silicon M-series (NEON) | $\ge 60\text{ tokens/sec}$ |
| **Peak RAM Consumption during Generation** | Standard 64-bit OS | $\le 120\text{ MB}$ Total Working Set |
| **Cold Start Time to First Token** | Native binary launch | $\le 150\text{ ms}$ |
| **Multi-Agent Message Throughput** | Local IPC (Shared Memory) | $\ge 1,000\text{ msg/sec}$ |

