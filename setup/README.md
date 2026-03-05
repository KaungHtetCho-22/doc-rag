# DocRAG Intelligence Platform

> End-to-end Document Intelligence: ingest → parse → embed → fine-tune → serve

A production-grade RAG system that parses academic documents, fine-tunes a
domain-specific LLM on consumer hardware, and serves intelligent Q&A via
hybrid local/cloud inference.

---

## Tech Stack

| Layer | Tool | Role |
|-------|------|------|
| Ingestion | **Docling** | Parse PDFs → structured Markdown |
| Vector Store | **Qdrant** | Hybrid search + metadata filtering |
| Data Generation | **Groq** | Synthetic QA pair generation |
| Fine-tuning | **Unsloth** | QLoRA on RTX 3060 6GB |
| Local Inference | **llama.cpp** | Offline GGUF model serving |
| Cloud Inference | **Groq** | Ultra-fast API inference |
| MLOps | **W&B + DVC + Ragas** | Tracking, versioning, evaluation |

---

## Architecture

```
arXiv / Papers With Code
        │
        ▼
    [Docling]  ──→  DVC
        │
        ▼
    [Qdrant]  ←── Hybrid Search (dense + sparse)
        │
        ▼
   [Groq API]  ──→  Synthetic QA pairs
        │
        ▼
    [Unsloth]  ──→  W&B tracking
        │
        ▼
  GGUF Export  ──→  [llama.cpp]
        │
        ▼
   [FastAPI]  ←── mode switch: Groq ☁️ / llama.cpp 🖥️
        │
        ▼
   [Gradio UI]
```

---

## Quick Start

### Prerequisites
- Python 3.10+
- Docker Desktop
- NVIDIA GPU with 6GB+ VRAM (RTX 3060 recommended)
- CUDA 12.1 drivers

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/docrag-platform.git
cd docrag-platform
bash setup.sh
```

### 2. Configure API Keys

```bash
# Edit .env with your keys
nano .env
```

Required keys:
- `GROQ_API_KEY` — free at [console.groq.com](https://console.groq.com)
- `WANDB_API_KEY` — free at [wandb.ai](https://wandb.ai)

### 3. Login to W&B

```bash
source venv/bin/activate
wandb login
```

### 4. Verify Setup

```bash
python verify_setup.py
```

Expected output:
```
  ✅  Python 3.10+
  ✅  PyTorch              →  torch 2.4.0 | NVIDIA GeForce RTX 3060 (6.0GB VRAM)
  ✅  Unsloth
  ✅  Docling
  ✅  Qdrant (Docker)      →  Connected at localhost:6333
  ✅  Groq API             →  API key valid
  ✅  Weights & Biases     →  API key found
  ✅  DVC
  ...
  12/12 checks passed  |  All systems go! 🚀
```

### 5. Run the Pipeline

```bash
# Stage 1: Collect & parse documents
python ingestion/collect.py
python ingestion/parse.py

# Or run full DVC pipeline
dvc repro
```

---

## Project Structure

```
docrag-platform/
├── docs/                    ← architecture, roadmap, objectives
├── ingestion/               ← data collection + Docling parsing
├── vectorstore/             ← Qdrant setup + indexing
├── training/                ← Unsloth fine-tuning + data generation
├── inference/               ← Groq + llama.cpp serving
├── evaluation/              ← Ragas evaluation harness
├── api/                     ← FastAPI backend
├── ui/                      ← Gradio frontend
├── data/                    ← managed by DVC (not in Git)
├── models/                  ← model weights (not in Git)
├── .github/workflows/       ← CI/CD pipelines
├── setup.sh                 ← one-command environment setup
├── verify_setup.py          ← health check for all components
├── docker-compose.yml       ← Qdrant container
├── dvc.yaml                 ← reproducible data pipeline
├── requirements.txt
└── .env.example
```

---

## Hardware

Built and tested on:
- **GPU:** NVIDIA RTX 3060 Laptop (6GB VRAM)
- **RAM:** 16GB
- **OS:** Ubuntu 22.04 / Windows 11 WSL2

---

## Docs

- [Architecture](docs/architecture.md)
- [Project Stages](docs/stages.md)
- [Tech Stack](docs/tech_stack.md)
- [Objectives](docs/objectives.md)
- [Business Values](docs/business_values.md)
- [Roadmap](docs/roadmap.md)
