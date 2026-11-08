# HariAI

**Local AI by K R Hari Prajwal — a standalone language model and experimental typed decisions.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Noncommercial-orange)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experimental_alpha-yellow)](docs/research/STANDALONE_MODEL_PLAN.md)

HariAI runs a local GGUF model directly in Python. On the development machine,
the model file is already available and chat needs no Ollama service or network.
The model-bearing release wheel has **not** been published or approved: the
current models failed quality checks and model files are excluded from Git.

The language model is derived from **Hugging Face SmolLM-135M-Instruct**. HariAI's
adaptation, runtime and packaging are by **K R Hari Prajwal**. Upstream model
credit and licensing are preserved. This is an experimental small assistant,
not a ChatGPT-equivalent foundation model trained from scratch.

## Start chatting

Requires Python 3.10+ and the native `llama-cpp-python` runtime. The development checkout can use an existing compatible model file:

```powershell
python -m pip install -e .
python -m hari_ai.cli ask "Hello!"
python -m hari_ai.cli
```

After changing the local code, enter `/exit` and start the app again. Enter
`/version` inside chat to check the running app version. The version number is
for the application; it does not imply that a new language model was trained.

No wheel or model is published yet. A fresh clone requires a compatible ChatML
GGUF file configured with `hari settings model-file PATH`; Git does not include it.
For the development checkout with the local model present:

```bash
python -m pip install -e .
python -m hari_ai.cli ask "Explain a Python list"
```

If compiling the native runtime is inconvenient, its maintainers document
[CPU wheels](https://github.com/abetlen/llama-cpp-python#installation):

```bash
python -m pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

Large model files are excluded from normal Git commits. A plain source clone
needs a release model or an explicitly configured compatible ChatML GGUF:

```bash
hari settings model-file /path/to/hariai-chat.gguf
hari settings backend local
```

## What is included

| Component | Purpose | Current limit |
| --- | --- | --- |
| Local language model | Experimental conversation | About 92 MB; failed basic arithmetic, factual and coding checks |
| Open-JEV reference | Typed decision architecture, packaged with attribution | Not TypeSafe AI's production Jev model |
| Operation-policy experiment | A small trained Open-JEV classifier for a fixed JSON schema | Checkpoint is local and excluded from Git; advisory only |
| Memory | Explicit local save, inspect and delete | Plaintext; not automatically used for training |
| Dream research | Inspect experiences and run separate candidate experiments | No automatic self-improvement or production checkpoint promotion |
| Original Nano | Archived research implementation, explicitly selectable | Failed general-chat probes; no longer the default |

The original 35 MB target is **not met** by the chat model. File size, installed
Python dependencies, and peak memory are different measurements. See the
[standalone model research](docs/research/STANDALONE_MODEL_PLAN.md) and the
[live conversation report](docs/research/LIVE_CHAT_EVALUATION.md) for actual responses.

The original README from `master` is preserved as [a historical document](docs/MASTER_README_HISTORICAL.md). It contains outdated capability claims; this README records the current results.

## Python API

```python
from hari_ai import HariAI

ai = HariAI()  # Requires a local GGUF file; no Ollama service
print(ai.ask("Explain recursion in two sentences."))
print(ai.ask("Give a small example."))  # Retains this session's context

# Requires the locally trained operation checkpoint, which is not in Git.
# This separate, bounded typed-decision task never performs the operation.
result = ai.decide_operation({
    "operation": "delete",
    "scope": "all",
    "authorized": "yes",
    "backup": "no",
})
print(result)
```

The operation model accepts `read`, `list`, `create`, `update`, `delete`,
`format`, `overwrite`, or `export`; `scope` is `single`, `batch`, or `all`;
`authorized` and `backup` are `yes` or `no`. It returns a choice from `proceed`,
`review`, and `stop`. Its probabilities are not calibrated correctness scores.
Use actual program permissions to control actions. General `decide` calls need
a separately trained checkpoint matching the intended task.

## Settings and local memory

The default backend is `local`. An existing Ollama installation remains optional:

```bash
hari settings show
hari settings list-models
hari settings ollama-model your-installed-model
hari settings backend ollama
hari settings backend local
hari memory add "I prefer concise explanations"
hari memory inspect
hari memory delete 1
```

Settings and saved memories live in `~/.hari_ai`, or `HARIAI_DATA_DIR` when set.
Chat history is held in the session. Saved memories are plaintext and are not
fed into training automatically. The separate `search` command uses the network;
ordinary local chat does not.

The CLI shows the **average base-model token likelihood** after local
model-generated replies. For example, `Token likelihood: 83% (not answer
accuracy)` means the model assigned its generated tokens an average probability
of 83% before sampling adjustments. This is a language-prediction signal, not
a calibrated probability that the answer is correct. The line is grey in an
interactive terminal. Hard-coded replies or unsupported runtimes show
`Token likelihood: unavailable`. The lightweight observer is currently checked
against `llama-cpp-python` 0.3.8; other versions still run chat but omit this
score until their sampling interface is verified. Use
`hari settings confidence off` to hide it or `hari settings confidence on` to
show it again. See [confidence research](docs/research/CHAT_CONFIDENCE.md) and
the [calibration system design](docs/research/CALIBRATED_CONFIDENCE_DESIGN.md).
The [independent-model rebuild audit](docs/research/REBUILD_AUDIT_2026_11.md)
separates the current SmolLM-derived chat baseline from scratch-training work.
Run `hari status` or `/status` to see the active backend, model file and the
actual state of the experimental learning and cluster features.
Use `hari settings confidence-status` to see the current confidence coverage.
For a complete, single-turn integer arithmetic question and an unambiguous
number or matching equation, HariAI checks the calculation directly and shows either
`100% (verified arithmetic)` or `0% (verified arithmetic)`. This is an exact
check of that restricted answer, not the language model's self-confidence;
when available, the separate token-likelihood line is shown as well.
General-chat confidence has not been validated and remains unavailable.
For an underspecified request such as “give me the C program”, HariAI asks what
the program should do. The local language model can still generate invalid code.

## Reproduce and evaluate

```bash
python -m unittest discover tests
python scripts/adapt_smollm.py --base /path/to/original-smollm.gguf --output checkpoints/my-candidate --steps 300
python scripts/train_operation_jev.py
python scripts/assemble_release.py /path/to/passing-candidate --fresh-report /path/to/report.json
python scripts/build_release.py
```

Training is offline once the base file and dependencies are installed. Candidate
scripts write separate checkpoints and reports. They do not replace the active
model. Small development probes are not evidence of general benchmark accuracy.
Candidates must pass release gates, and model files must match their manifest hashes before building. No candidate has passed these gates yet.

The new model work supersedes earlier claims about a verified general assistant
under 35/45 MB. Historical experiments are retained in [BENCHMARKS.md](BENCHMARKS.md)
and [Dream research status](docs/research/DREAM_ENGINE_STATUS.md).

## Attribution and license

Created and maintained by [K R Hari Prajwal](https://github.com/Hariprajwal).
HariAI software and HariAI-specific modifications use the
[HariAI Noncommercial License](LICENSE). Commercial use of those portions
requires a separate written agreement with the creator.

SmolLM's upstream weights and the vendored Open-JEV implementation retain their
Apache 2.0 terms. See [third-party notices](THIRD_PARTY_NOTICES.md), the bundled
model license, and [data provenance](DATA_LICENSES.md). The HariAI commercial
restriction does not revoke upstream rights or earlier Apache grants. This
project is source-available, not OSI open source.

Report problems through [GitHub Issues](https://github.com/Hariprajwal/HariAI/issues),
including package version, platform, model hash and a reproducible prompt.
