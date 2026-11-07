# HariAI

**A research project exploring compact language models and controlled continual learning.**

HariAI aims to build a useful local assistant with neural model weights smaller
than 35 MB. The repository contains a small Transformer, training and
quantization experiments, a terminal interface, and the beginnings of a Dream
Engine for evaluating candidate improvements.

> **Research prototype:** HariAI is not yet a general-purpose assistant and has
> not demonstrated ChatGPT-level performance. The experimental checkpoints are
> not included in this repository. The Dream controller does not automatically
> train or improve the model.

## What is here today

- **HariNano:** a from-scratch Transformer and byte-fallback tokenizer for
  experiments with small models and INT4 storage.
- **Local chat integration:** when Ollama and a compatible local model are
  available, chat can use that model. Ollama models are separate downloads and
  can be much larger than HariNano's 35 MB target.
- **Dream research:** an offline LoRA candidate experiment with arithmetic
  replay and held-out regression checks. Candidates are never automatically
  promoted. The interactive `dream now` command currently inspects memory only.
- **Explicit local memory:** save, inspect and delete individual memories.
  Saved memories are stored in plaintext on the local machine and are not
  automatically used for model training.
- **Reproducible probes:** scripts record checkpoint size, generated answers,
  data and checkpoint hashes, and narrow held-out scores.

## Quick start

Requires Python 3.9 or newer, PyTorch and NumPy. From a clone of this repository:

```bash
python -m pip install -e .
python -m hari_ai.cli info
python -m hari_ai.cli dream now
```

On Windows you can start the interactive CLI with `./hari.bat`. For general
chat, start [Ollama](https://ollama.com/) and install a compatible chat model,
then run `python -m hari_ai.cli`. Select a model explicitly if needed:

```powershell
$env:HARIAI_OLLAMA_MODEL = "your-installed-model-name"
python -m hari_ai.cli ask "Explain recursion"
```

The name must match `ollama list`. Set `HARIAI_OLLAMA_HOST` if the service uses
a non-default URL. Without a reachable model, the chat command explains that it
cannot answer general questions. Responses from a model can be wrong.

### Local memory

```bash
python -m hari_ai.cli memory add "I prefer concise explanations"
python -m hari_ai.cli memory inspect
python -m hari_ai.cli memory delete 1
```

These are deliberate saves, not automatic collection from chat. Memory is stored
as plaintext in `~/.hari_ai/memory.sqlite3`; set `HARIAI_DATA_DIR` to choose a
different directory. Do not save passwords or secrets. `memory clear` clears
session working memory; delete saved entries individually with their IDs.

## Research results

| Experiment | Result | What it shows |
| --- | --- | --- |
| Arithmetic student | 33/42 held-out arithmetic questions after INT4; 581,380-byte checkpoint | Narrow addition and subtraction only |
| Language pilot | Held-out loss 5.714 → 1.855 nats per byte token; 1,473,688-byte INT4 checkpoint | Learned text patterns; scored 0/8 on a small mixed prompt probe |
| Dream candidate | 33/42 → 35/42 arithmetic after INT4; language regression exceeded the gate | Candidate rejected; active checkpoint unchanged |

These are small local experiments, not standard model benchmarks. Checkpoints
and sampled training data are Git-ignored; results cannot be independently
reproduced from this repository alone until their source data is obtained and
the training scripts are run. See [BENCHMARKS.md](BENCHMARKS.md), the
[35 MB research plan](docs/research/35MB_GENERAL_ASSISTANT_PLAN.md), and the
[Dream Engine status](docs/research/DREAM_ENGINE_STATUS.md).

## Project layout

```text
hari_ai/model/          Transformer configuration and layers
hari_ai/tokenizer/      Byte-fallback BPE tokenizer
hari_ai/trainer/        Training infrastructure
hari_ai/dream/          Experimental adaptation and sleep controller
hari_ai/memory/         Session memory and opt-in local storage
hari_ai/quantization/   Experimental packed INT4 export
benchmarks/             Checkpoint inspection and small probes
scripts/                Data collection and training experiments
```

Run the test suite with `python -m unittest discover tests`. Research scripts
that collect data require network access and record source metadata; review the
relevant dataset and teacher terms before use or redistribution. Model files
loaded with PyTorch must come from a trusted source.

## Limits and roadmap

HariAI still needs broad language and instruction training, reliable evaluation
on held-out conversation and coding tasks, true resource-aware scheduling,
privacy and retention controls, and a robust checkpoint promotion and rollback
system. It does not currently verify arbitrary answers or execute generated
code in a security sandbox. See [DREAM_ENGINE_STATUS.md](docs/research/DREAM_ENGINE_STATUS.md)
for the implemented experiment and remaining work.

## License and contact

The current software is available under the [HariAI Noncommercial License](LICENSE).
Commercial use requires a separate written agreement with K R Hari Prajwal;
contact the creator through [GitHub](https://github.com/Hariprajwal). The project is source-available; the commercial
restriction means it is not open source under the OSI definition. Previous
versions were offered under Apache 2.0 and earlier license grants are not revoked.
Third-party models, datasets and dependencies have their own terms.

Created by [K R Hari Prajwal](https://github.com/Hariprajwal).
