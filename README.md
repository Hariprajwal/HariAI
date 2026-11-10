# HariAI

**Local AI by K R Hari Prajwal — Standalone Edge Intelligence, Autonomous Desktop Control, and Universal Document Systems.**

<img width="1917" height="1057" alt="image" src="https://github.com/user-attachments/assets/8f370fec-154a-4f9e-86c5-146961681abd" />

[![PyPI version](https://img.shields.io/pypi/v/hari-ai.svg?color=c17aff)](https://pypi.org/project/hari-ai/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Package Size](https://img.shields.io/badge/Wheel-82.9%20MB%20(<100MB)-success)](https://pypi.org/project/hari-ai/)
[![MCP Ready](https://img.shields.io/badge/MCP-Supported-blueviolet)](https://modelcontextprotocol.io)
[![License](https://img.shields.io/badge/License-Noncommercial-orange)](LICENSE)
[![Creator](https://img.shields.io/badge/Creator-K%20R%20Hari%20Prajwal-purple)](https://github.com/Hariprajwal)

HariAI is a completely local, self-contained AI system engineered to run directly on consumer hardware without mandatory cloud APIs, subscription keys, or network connectivity. Packaged into a lightweight wheel (<100 MB), it provides standalone conversational inference, an autonomous hands-free desktop agent with floating takeover controls, universal multi-format document conversion, Model Context Protocol (MCP) tool integration, and continual memory consolidation.

---

## ⚡ Quick Start

### Installation
Install the complete standalone package from PyPI:

```bash
pip install hari-ai
```

*(Optional extras for computer vision and desktop control)*:
```bash
pip install "hari-ai[desktop,ocr]"
```

### Launch Interactive Dashboard
```bash
hari
```
*(or `hari run`)*

### ☁️ Google Colab & Jupyter Notebooks
HariAI automatically detects Google Colab, Kaggle, and Jupyter notebook kernels:
* **Zero UI Glitches**: Automatically switches to optimized silent/line REPL mode so notebook cell outputs stream cleanly without alternate-screen TUI buffer errors.
* **Safe Input Handling**: Bypasses terminal raw-mode conflicts in IPython event loops.
```bash
!pip install hari-ai
!hari ask "Hello from Google Colab!"
```

---

## 🖥️ Live Terminal Dashboard

HariAI features a keyboard- and mouse-friendly dual-pane terminal interface built on Textual and Rich with an aesthetic purple theme.

![HariAI live terminal dashboard](docs/images/terminal-dashboard.png)

### Key Dashboard Highlights
* **Active Capability Indicators**: Live verification showing `✓ Automatic training`, `✓ Autonomous tool execution`, `✓ Web search via /search`, and `✓ Explicit local memory`.
* **Streaming Inference with Token Likelihood**: Real-time token generation with token probability and verified arithmetic checks.
* **⌘ Multi-HariAI Cluster**: Swarm debate and consensus engine (`/swarm <topic>`).
* **✦ Sacred Mahakaal Representation**: High-resolution unicode Braille, Trinetra, and Trishul sacred art renderer (`hari mahakal`).
* **⌁ Quick Commands Toolbar**: One-click execution for `/status`, `/convert`, `/tools`, `/auto`, `/takeover`, `/swarm`, `/dream`, and `/settings`.
* **Shortcuts**:
  * `F2`: Toggle side panels on compact or narrow displays.
  * `Esc`: Instantly interrupt in-flight generation or active tool tasks.
  * `Ctrl+L`: Clear screen and start a fresh session.
  * `Ctrl+Q`: Exit application.

---

## 🚀 Core Capabilities

### 1. 🔄 Universal Document Converter (`hari convert`)
Convert any document format to any target format locally without data leaking to external servers.

```bash
# Convert PDF to Word document
hari convert document.pdf docx

# Convert PowerPoint presentation to PDF
hari convert presentation.pptx pdf

# Convert Excel spreadsheet to CSV or Markdown
hari convert financials.xlsx csv
hari convert data.csv md

# Absolute path support with in-place output
hari convert "D:\Projects\Report.pdf" docx
```

**Supported Formats:** `PDF`, `DOCX`, `PPTX`, `PPT`, `XLSX`, `XLS`, `TXT`, `MD`, `HTML`, `CSV`, `TSV`, `RTF`.

---

### 2. 🤖 Autonomous Desktop Agent & Takeover Pill (`hari autonomous`)
A bounded, safe desktop agent that perceives screen state via local OCR and vision models, then plans and executes keyboard and mouse actions.

```bash
# Launch a goal-driven desktop task
hari autonomous "Open browser and check PyPI release status"

# Test or activate the floating Take Over control window directly
hari takeover
```

* **Tri-Level Architecture**:
  * **System 1**: Open-JEV sub-millisecond input execution and bounds checking.
  * **System 2**: HariAI deliberate planning and step evaluation.
  * **System 3**: Open web search fallback for dynamic real-time information.
* **Safety & Control**:
  * **Floating Takeover Pill**: On-screen top-center floating widget (`● HariAI has control   [ Pause ]   [ Take over ]`) with a purple screen edge glow.
  * **Instant Return of Control**: Clicking `[ Take over ]` or pressing `<Esc>` immediately halts all automation and surrenders control back to the human operator.
  * **Fail-Safe**: Moving the mouse to any screen corner immediately aborts execution.

---

### 3. 🔌 Model Context Protocol (MCP) Integration (`hari mcp`)
Connect external tools, databases, enterprise systems, and custom scripts through the open standard Model Context Protocol:

```bash
# List connected MCP servers and available tools
hari mcp list

# Add an external stdio MCP server
hari mcp add filesystem npx -y @modelcontextprotocol/server-filesystem "D:\workspace"
hari mcp add github npx -y @modelcontextprotocol/server-github

# Connect and discover tools
hari mcp connect

# Let HariAI autonomously plan and call tools for a task
hari tools run "List all files in the current directory and find recent changelogs"
```

---

### 4. 🧠 Continual Learning & Dream Engine (`hari dream`)
Inspired by mammalian sleep consolidation, HariAI features an autonomous background dream process that triggers during idle periods:

* **Episodic Replay**: Reviews recent session experiences during inactive idle intervals.
* **Synthetic Rehearsal Buffer**: Automatically synthesizes self-supervised practice exercises to prevent catastrophic forgetting.
* **Offline & Private**: Runs 100% locally on your machine without transmitting user memories or telemetry over the internet.

```bash
# Inspect dream engine status and rehearsal counts
hari dream status

# Trigger immediate memory consolidation
hari dream now

# Toggle automatic background idle sleep worker
hari dream auto on
hari dream auto off
```

---

### 5. 🔱 Sacred Mahakaal Artwork (`hari mahakal`)
Renders high-density sacred representations of Mahakaal (Lord Shiva) in the terminal:

```bash
hari mahakal braille    # High-density unicode Braille silhouette with crescent moon
hari mahakal trinetra   # Sacred Third Eye (Trinetra) & Tripundra vibhuti
hari mahakal trishul    # Divine Trishul & Damru representation
hari mahakal text       # 256-color purple ANSI block representation
```

---

## 🛠️ CLI Command Reference

| Command | Usage | Description |
| :--- | :--- | :--- |
| `hari` / `hari run` | `hari run [--plain]` | Launches the purple live dashboard (or minimal line REPL) |
| `hari ask` | `hari ask "<prompt>"` | Single-shot local model query directly in terminal |
| `hari convert` | `hari convert <file> [format]` | Universal document conversion (PDF, DOCX, PPTX, XLSX, etc.) |
| `hari autonomous` | `hari autonomous "<goal>"` | Hands-free bounded autonomous computer control |
| `hari takeover` | `hari takeover` | Launches the floating Take Over pill window with Pause/Resume |
| `hari tools` | `hari tools run "<task>"` | Autonomous tool planning across local & MCP tools |
| `hari mcp` | `hari mcp [list\|add\|connect]` | Manage Model Context Protocol server connections |
| `hari swarm` | `hari swarm "<topic>"` | Multi-bot debate & consensus engine |
| `hari mahakal` | `hari mahakal [style]` | Renders sacred Mahakaal artwork (Braille/Trinetra/Trishul) |
| `hari status` | `hari status` | Live hardware, memory, model, and runtime state |
| `hari memory` | `hari memory [add\|inspect\|delete]` | Explicit local memory storage (`~/.hari_ai`) |
| `hari settings` | `hari settings show` | Inspect and configure local model and backend options |
| `hari info` | `hari info` | Displays creator profile and system provenance |

---

## 🐍 Python SDK API

HariAI provides a clean Python interface for embedding within applications or automated workflows:

```python
from hari_ai import HariAI
from hari_ai.tools.converter import convert_document
from hari_ai.agent.autonomous import AutonomousAgent

# 1. Standalone Conversational Inference
ai = HariAI()
reply = ai.ask("Explain the difference between synchronous and asynchronous execution.")
print(reply)

# 2. Universal Document Conversion
output_file = convert_document("annual_report.pptx", target_format="pdf")
print(f"Document saved to: {output_file}")

# 3. Autonomous Computer Control
agent = AutonomousAgent()
result = agent.run("Open settings and check display resolution")
print(f"Status: {result.status}, Actions: {result.total_steps}")
```

---

## 🤖 Agent Discovery Specification

HariAI is built to be discoverable and invocable by external AI coding agents (Claude Code, Cursor, Codex, Gemini CLI, LangChain, AutoGen):

```json
{
  "name": "hari-ai",
  "version": "0.2.0a5",
  "description": "Standalone edge SLM, autonomous desktop agent, and document converter",
  "entrypoint": "hari",
  "capabilities": {
    "document_converter": {
      "cli": "hari convert <input> <target_format>",
      "supported_formats": ["pdf", "docx", "pptx", "xlsx", "txt", "md", "html", "csv"]
    },
    "autonomous_desktop": {
      "cli": "hari autonomous \"<goal>\"",
      "overlay_pill": "hari takeover",
      "safety": "Esc or Take Over button aborts immediately"
    },
    "mcp_server": {
      "cli": "hari mcp",
      "protocol": "Model Context Protocol v1.0"
    },
    "local_inference": {
      "cli": "hari ask \"<prompt>\"",
      "model_type": "GGUF Q4_K_M (Offline edge SLM)"
    }
  }
}
```

---

## 🔬 Architecture & Model Provenance

* **Core Language Model**: Quantized ChatML GGUF derived from Hugging Face SmolLM-135M-Instruct, optimized with `llama-cpp-python` 0.3.8 for edge CPU deployment.
* **Zero Cloud Dependence**: Ordinary local inference, document transformation, and memory retrieval never make external network calls.
* **Data Privacy**: All local memories and settings are stored locally in `~/.hari_ai` (or `$HARIAI_DATA_DIR`).

---

## 👤 Attribution & License

* **Creator & Author**: [K R Hari Prajwal](https://github.com/Hariprajwal)
* **License**: Software modifications, runtime, and architecture are distributed under the [HariAI Noncommercial License](LICENSE).
* **Third-Party Notices**: Upstream model weights and vendored components preserve their respective Apache 2.0 licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [DATA_LICENSES.md](DATA_LICENSES.md).

For inquiries, bugs, or feature proposals, open an issue on the [GitHub Issues](https://github.com/Hariprajwal/HariAI/issues) page.
