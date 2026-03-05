# DocRAG — Complete Setup & Run Guide

> Follow this guide top to bottom. Each step must succeed before moving to the next.

---

## Prerequisites

| Requirement | Check |
|---|---|
| NVIDIA GPU (6GB+ VRAM) | `nvidia-smi` |
| Docker + Docker Compose | `docker compose version` |
| Python 3.11 or 3.12 | `python3 --version` |
| `uv` package manager | `uv --version` |
| Groq API key | https://console.groq.com (free) |

Install `uv` if missing:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc
```

---

## Step 0 — Project structure check

Make sure you are in the project root and all scripts are present:

```bash
cd ~/portfolio/doc-rag
ls
```

You should see:
```
ingestion/   vectorstore/   training/   inference/   evaluation/   ui/
setup.sh     verify_setup.py   requirements.txt   docker-compose.yml
```

---

## Step 1 — Environment Setup

### 1a. Create virtual environment and install dependencies

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -r requirements.txt
```

### 1b. Set up environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your key:
```
GROQ_API_KEY=your_groq_api_key_here
```

Get a free key at https://console.groq.com — no credit card needed.

### 1c. Start Qdrant

```bash
docker compose up -d
```

Verify it's running:
```bash
curl http://localhost:6333/healthz
# Expected: {"title":"qdrant - vector search engine"}
```

### 1d. Verify everything

```bash
python verify_setup.py
```

Expected: **14/14 checks passed ✅**

If any check fails, fix it before continuing.

---

## Step 2 — Data Collection

Collect ~150 Computer Vision papers from arXiv and Papers With Code.

```bash
python ingestion/collect.py
```

**What it does:**
- Searches arXiv for CV papers (object detection, segmentation, transformers, etc.)
- Enriches with GitHub URLs and conference info from Papers With Code
- Downloads PDFs to `data/raw/pdfs/`
- Saves metadata to `data/raw/metadata.jsonl`

**Expected output:**
```
Collected: 76 unique papers
PDFs downloaded: 72
```

**Time:** ~5–10 minutes (polite rate limiting)

> ⚠️ If you see Cloudflare errors for JobsDB/JobStreet — that's expected, ignore them.

---

## Step 3 — PDF Parsing

Parse all PDFs into section-aware chunks using Docling.

```bash
python ingestion/parse.py
```

**What it does:**
- Reads each PDF with Docling (preserves section structure)
- Splits into chunks of ~200–350 words with 2-sentence overlap
- Saves to `data/processed/chunks.jsonl`
- Saves parse report to `data/processed/parse_report.json`

**Expected output:**
```
PDFs parsed: 72
Total chunks: ~2100–2200
Avg words/chunk: ~250
```

**Time:** 15–30 minutes (Docling is CPU-heavy)

Check the report:
```bash
cat data/processed/parse_report.json
```

---

## Step 4 — Vector Store Setup

### 4a. Create Qdrant collection

```bash
python vectorstore/setup_collection.py
```

Expected:
```
Status   : green
Points   : 0
Dense dim: 384
Sparse   : BM25
```

### 4b. Embed and index all chunks

```bash
python vectorstore/index.py
```

**What it does:**
- Loads `all-MiniLM-L6-v2` onto your GPU for dense embeddings
- Generates BM25 sparse embeddings via FastEmbed
- Upserts 2,141 chunks into Qdrant with deterministic UUIDs
- Resume-safe: skips already-indexed chunks if interrupted

**Expected output:**
```
Chunks indexed this run: 2141
Total points in Qdrant : 2141
```

**Time:** ~6 seconds on RTX 3060

### 4c. Test hybrid search

```bash
python vectorstore/search.py
```

Then benchmark:
```bash
python vectorstore/search.py --benchmark
```

Expected latency: **avg ~5ms, p95 <8ms** ✅

---

## Step 5 — Synthetic Training Data

Generate ~4,000 QA pairs from the indexed chunks using Groq.

### 5a. Quick test first (10 chunks)

```bash
python training/generate_data.py --max-chunks 10
```

Check the output looks good:
```bash
head -n 1 data/training/qa_pairs.jsonl | python -m json.tool
```

You should see a `conversations` array with `human` and `assistant` turns.

### 5b. Full generation

```bash
python training/generate_data.py
```

**What it does:**
- Skips References and Acknowledgement sections (low quality)
- Skips chunks under 50 words
- Generates 2 QA pairs per chunk via `llama-3.3-70b-versatile`
- Deduplicates questions
- Saves 80/20 train/eval split in ShareGPT format

**Expected output:**
```
Raw QA pairs   : ~3,800–4,200
After dedup    : ~3,600–4,000
Train split    : ~2,900–3,200
Eval split     : ~700–800
```

**Time:** ~35–45 minutes (Groq free tier rate limits)

> ⚠️ If interrupted, restart with `--resume` to skip already-processed chunks:
> ```bash
> python training/generate_data.py --resume
> ```

---

## Step 6 — Fine-tuning

Fine-tune Llama 3.2 3B with QLoRA on your RTX 3060.

### 6a. Dry run first (verify setup, 10 steps only)

```bash
python training/finetune.py --dry-run
```

Expected: `Fine-tuning Complete ✅` with no OOM errors.

If you see `CUDA out of memory` — the memory settings are already tuned for 6GB:
- `MAX_SEQ_LENGTH = 1024`
- `BATCH_SIZE = 1`
- `PYTORCH_CUDA_ALLOC_CONF = expandable_segments:True`

### 6b. Full training run

```bash
python training/finetune.py
```

**What it does:**
- Loads Llama 3.2 3B in 4-bit (QLoRA) — fits in ~5.5GB VRAM
- Attaches LoRA adapters (11M trainable of 1.25B params = 0.9%)
- Trains for 3 epochs with cosine LR schedule
- Saves adapter to `models/adapter/`
- Merges and exports GGUF Q4_K_M to `models/docrag-3b-q4.gguf`
- Logs all metrics to MLflow

**Expected training loss:** should drop from ~1.5 → ~0.6 over 3 epochs

**Time:** ~4–5 hours on RTX 3060

> ⚠️ **If you stop training mid-run:**
> The script auto-resumes from the latest checkpoint. Just run the same command again:
> ```bash
> python training/finetune.py
> ```
> It will print: `Resuming from checkpoint: models/adapter/checkpoint-XXX`

### 6c. Monitor training

In a second terminal:
```bash
source .venv/bin/activate
mlflow ui --port 5000
# Open: http://localhost:5000
```

You'll see live loss curves updating every 10 steps.

---

## Step 7 — Inference Server

### 7a. Start the server (Groq mode — works before training finishes)

```bash
# Terminal 1 — keep this running
python inference/serve.py
```

Expected startup:
```
Qdrant connected — 2141 points
Groq client ready — model: llama-3.1-8b-instant
Inference mode: GROQ
Server ready ✅
```

### 7b. Test queries

```bash
# Terminal 2
python inference/test_query.py
```

Custom queries:
```bash
python inference/test_query.py --query "how does DETR work?"
python inference/test_query.py --benchmark
```

Expected: **~1.1s total latency** (50ms retrieval + ~1s Groq generation)

### 7c. Switch to local mode (after training finishes)

Verify the GGUF exists:
```bash
ls -lh models/docrag-3b-q4.gguf
# Expected: ~2GB file
```

Restart server in local mode:
```bash
# Stop the server (Ctrl+C), then:
python inference/serve.py --mode local
```

Expected:
```
Local GGUF model loaded ✅
Inference mode: LOCAL
```

---

## Step 8 — Evaluation

Evaluate RAG quality with LLM-as-judge (requires server running).

```bash
# Quick test (5 questions, ~3 min)
python evaluation/run_ragas.py --n 5

# Full evaluation (20 questions, ~15 min)
python evaluation/run_ragas.py --n 20
```

**Metrics and targets:**

| Metric | Target | What it measures |
|--------|--------|-----------------|
| Faithfulness | > 0.80 | Answer grounded in retrieved context? |
| Answer Relevancy | > 0.75 | Answer actually addresses the question? |
| Context Recall | > 0.70 | Context contains needed information? |
| Context Precision | > 0.70 | Retrieved chunks are useful? |

After training, compare Groq vs local:
```bash
# Groq mode
python evaluation/run_ragas.py --mode groq --n 20

# Local mode (fine-tuned model)
python evaluation/run_ragas.py --mode local --n 20
```

Results saved to:
- `evaluation/results/ragas_scores.json`
- `evaluation/results/ragas_report.md`

---

## Step 9 — Gradio UI

```bash
# inference/serve.py must be running in another terminal first

python ui/app.py
# Open: http://localhost:7860
```

The UI has 4 tabs:
- **Query** — ask questions with optional filters (year, section, has_code)
- **Live Parsing** — upload any PDF to see Docling parse it in real time
- **Architecture** — system diagram and tech stack
- **Evaluation** — shows latest Ragas scores (auto-updates after running run_ragas.py)

---

## Full Run Order (summary)

```bash
# One-time setup
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env          # fill in GROQ_API_KEY
docker compose up -d
python verify_setup.py        # must show 14/14 ✅

# Data pipeline
python ingestion/collect.py
python ingestion/parse.py
python vectorstore/setup_collection.py
python vectorstore/index.py
python vectorstore/search.py --benchmark   # verify ~5ms latency

# Training data + fine-tune
python training/generate_data.py
python training/finetune.py --dry-run      # verify no OOM
python training/finetune.py               # ~4-5h, auto-resumes if stopped

# Serve + evaluate + UI
python inference/serve.py                  # Terminal 1, keep running
python inference/test_query.py             # Terminal 2
python evaluation/run_ragas.py --n 20      # Terminal 2
python ui/app.py                           # Terminal 2, open localhost:7860

# MLflow dashboard
mlflow ui --port 5000                      # open localhost:5000
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `CUDA out of memory` | Already tuned for 6GB — check nothing else is using GPU: `nvidia-smi` |
| `Address already in use :8000` | `kill $(lsof -t -i:8000)` |
| `Address already in use :7860` | `python ui/app.py --port 7861` |
| `model decommissioned` (Groq) | Update model name in `.env`: `GROQ_DATAGEN_MODEL=llama-3.3-70b-versatile` |
| Qdrant connection refused | `docker compose up -d` |
| `No chunks found` | Run `ingestion/collect.py` and `ingestion/parse.py` first |
| Training stops mid-run | Just run `python training/finetune.py` again — auto-resumes |
| `rm -rf data/` accident | Start from Step 2 — all scripts are intact, only data needs regenerating |