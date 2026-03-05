"""
verify_setup.py
─────────────────────────────────────────────────────────
Run after setup.sh to confirm every component is working.

Usage:
    python verify_setup.py
"""

import sys
import os
import importlib.metadata
from dotenv import load_dotenv

load_dotenv()

PASS = "✅"
FAIL = "❌"

results = []

def check(label, fn):
    try:
        msg = fn()
        results.append((PASS, label, msg or ""))
    except Exception as e:
        results.append((FAIL, label, str(e)))

# ── Python version ────────────────────────────────────────────────────
def check_python():
    v = sys.version_info
    assert v >= (3, 10), f"Need 3.10+, got {v.major}.{v.minor}"
    return f"Python {v.major}.{v.minor}.{v.micro}"

check("Python 3.10+", check_python)

# ── PyTorch + CUDA ────────────────────────────────────────────────────
def check_torch():
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        return f"torch {torch.__version__} | {name} ({vram:.1f}GB VRAM)"
    return f"torch {torch.__version__} | CPU only (no CUDA)"

check("PyTorch", check_torch)

# ── Unsloth ───────────────────────────────────────────────────────────
def check_unsloth():
    version = importlib.metadata.version("unsloth")
    return f"unsloth {version}"

check("Unsloth", check_unsloth)

# ── Docling ───────────────────────────────────────────────────────────
def check_docling():
    from docling.document_converter import DocumentConverter
    return "docling import OK"

check("Docling", check_docling)

# ── Qdrant ────────────────────────────────────────────────────────────
def check_qdrant():
    from qdrant_client import QdrantClient
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", 6333))
    client = QdrantClient(host=host, port=port, timeout=3)
    info = client.get_collections()
    return f"Connected at {host}:{port} | {len(info.collections)} collections"

check("Qdrant (Docker)", check_qdrant)

# ── Groq ──────────────────────────────────────────────────────────────
def check_groq():
    from groq import Groq
    key = os.getenv("GROQ_API_KEY", "")
    assert key and key != "your_groq_api_key_here", "GROQ_API_KEY not set in .env"
    version = importlib.metadata.version("groq")
    client = Groq(api_key=key)
    models = client.models.list()
    return f"groq {version} | {len(models.data)} models available"

check("Groq API", check_groq)

# ── MLflow ────────────────────────────────────────────────────────────
def check_mlflow():
    import mlflow
    return f"mlflow {mlflow.__version__} | local tracking ready"

check("MLflow", check_mlflow)

# ── DVC ───────────────────────────────────────────────────────────────
def check_dvc():
    import dvc
    assert os.path.exists(".dvc"), ".dvc directory not found — run: dvc init"
    return f"dvc {dvc.__version__} | .dvc directory exists"

check("DVC", check_dvc)

# ── sentence-transformers ─────────────────────────────────────────────
def check_st():
    import sentence_transformers
    return f"sentence-transformers {sentence_transformers.__version__}"

check("sentence-transformers", check_st)

# ── Ragas ─────────────────────────────────────────────────────────────
def check_ragas():
    version = importlib.metadata.version("ragas")
    return f"ragas {version}"

check("Ragas", check_ragas)

# ── llama-cpp-python ──────────────────────────────────────────────────
def check_llama_cpp():
    version = importlib.metadata.version("llama-cpp-python")
    return f"llama-cpp-python {version}"

check("llama-cpp-python", check_llama_cpp)

# ── FastAPI ───────────────────────────────────────────────────────────
def check_fastapi():
    import fastapi
    return f"fastapi {fastapi.__version__}"

check("FastAPI", check_fastapi)

# ── Gradio ────────────────────────────────────────────────────────────
def check_gradio():
    import gradio
    return f"gradio {gradio.__version__}"

check("Gradio", check_gradio)

# ── arXiv SDK ─────────────────────────────────────────────────────────
def check_arxiv():
    import arxiv
    version = importlib.metadata.version("arxiv")
    client = arxiv.Client()  # smoke test
    return f"arxiv {version} | client OK"

check("arXiv SDK", check_arxiv)

# ── Print results ─────────────────────────────────────────────────────
print()
print("═" * 60)
print("  DocRAG Platform — Setup Verification")
print("═" * 60)

passed = sum(1 for r in results if r[0] == PASS)
failed = sum(1 for r in results if r[0] == FAIL)

for icon, label, msg in results:
    suffix = f"  →  {msg}" if msg else ""
    print(f"  {icon}  {label:<30}{suffix}")

print()
print("═" * 60)
print(f"  {passed}/{len(results)} checks passed", end="")
if failed:
    print(f"  |  {failed} failed — fix before proceeding")
else:
    print("  |  All systems go! 🚀")
print("═" * 60)
print()

if failed:
    sys.exit(1)