# HariAI — Data, Teacher Models & Licensing Register

This document maintains full legal provenance, licensing classification, and commercial-use compliance for all datasets, teacher models, and synthetic generation sources used in HariAI.

---

## 1. Compliance Policy & Distillation Rules
- **Rule 1 (Zero License Contamination):** No weights or data under restrictive licenses (e.g., non-commercial CC-BY-NC, or terms prohibiting model training) shall be ingested into the training pipeline.
- **Rule 2 (Synthetic Distillation Terms):** Outputs from models whose licenses restrict synthetic generation for competing models (specifically Llama 3.x Community License Section 1.b) are strictly banned from training HariAI's weights.
- **Rule 3 (Commercial Permissibility):** Only Apache 2.0, MIT, Open Data Commons (ODC-By), or explicit open licenses allowing derivative model training are permitted.

---

## 2. Approved Teacher Models (For Distillation & Synthetic Traces)

**Local research run, November 2026:** The Ollama Qwen 3 4B model layer used by
`collect_teacher_arithmetic.py` has SHA-256
`3e4cb14174460404e7a233e531675303b2fbf7749c02f91864fe311ab6344e4f`.
Its accompanying local license layer contains Apache License 2.0. Teacher
answers are individually checked against Python arithmetic and are recorded with
their source and verification result. This is a small research dataset, not a
general assessment of model quality. The locally installed Qwen 2.5 3B package
contains a separate research-only license and was not used for this run.

| Model Name | Creator | Checkpoint / Source | License | Permissible Uses in HariAI | Verification URL |
|---|---|---|---|---|---|
| **Qwen 2.5 Coder** | Alibaba Cloud | Qwen2.5-Coder-7B-Instruct | Apache 2.0 | Synthetic code reasoning traces, algorithm explanations, test case generation. | [https://huggingface.co/Qwen](https://huggingface.co/Qwen) |
| **DeepSeek-R1-Distill-Qwen** | DeepSeek AI | 1.5B / 7B Checkpoints | MIT License | Mathematical step-by-step reasoning rationales, verification logic. | [https://github.com/deepseek-ai/DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) |
| **Mistral NeMo** | Mistral AI | 12B-Instruct | Apache 2.0 | Instruction-following dialogue pairs, syntactic reformulation. | [https://huggingface.co/mistralai](https://huggingface.co/mistralai) |
| **Gemma 2** | Google | 2B / 9B-IT | Gemma Terms of Use | Comparative benchmarking and evaluation baseline; permitted for distillation under terms. | [https://ai.google.dev/gemma](https://ai.google.dev/gemma) |

---

## 3. Approved Datasets (Pre-Training & SFT)

**Language pilot:** Up to 400 rows were sampled from
[HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
`sample-10BT`, via the Hugging Face datasets server. The dataset card lists
ODC-By 1.0; this label does not independently establish the rights of every
underlying web page. The local JSONL records each available source URL, original
row identifier and content SHA-256. Document hashes determine the local train
and validation split. This is an experimental local sample; dataset-server
rows are not pinned to a repository revision, so the retained JSONL hash is the
reproducibility identifier for this run. Review underlying provenance and terms
before redistributing data or treating this sample as a cleared release corpus.

| Dataset | Publisher | Domain | License | Verification Status |
|---|---|---|---|---|
| **FineWeb-Edu (Sampled)** | Hugging Face | High-quality educational text | ODC-By 1.0 | Approved |
| **The Stack v2 (Permissive)** | BigCode Project | Python, Shell, C, Algorithms | Apache 2.0 / Permissive | Approved (Opt-out list filtered) |
| **GSM8K** | OpenAI / Academic | Grade School Math Reasoning | MIT License | Approved |
| **MBPP & HumanEval** | Academic Authors | Python Programming Problems | CC-BY-4.0 / MIT | Approved (Evaluation & Training Splits Kept Strictly Disjoint) |
| **Synthetic ReAct & Decision Logs**| Internal / HariAI Team | Tool trajectories & action schemas | Apache 2.0 (Original) | Internally Generated & Certified |

---

## 4. Prohibited / Restricted Sources (Blacklist)

| Source | Creator | Reason for Blacklist |
|---|---|---|
| **Llama 3.1 / 3.2 Output Traces** | Meta AI | License Section 1.b prohibits training competing models for general distribution unless based on Llama weights. |
| **OpenAI ChatGPT / GPT-4 Direct Outputs** | OpenAI | Terms of Service prohibit using outputs to train competing models. |
| **Any CC-BY-NC Datasets** | Various | Non-commercial restriction conflicts with HariAI's open-source commercial viability. |



| **Open-JEV (Kye Gomez / TypeSafe AI Concept)** | Kye Gomez | GitHub Repository | Apache 2.0 | Architectural reference & typed decision readout heads (`Choice`, `Score`, `Noul`). | [https://github.com/kyegomez/open-jev](https://github.com/kyegomez/open-jev) |
