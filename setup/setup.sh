#!/bin/bash
# ─────────────────────────────────────────────────────────────────────
# DocRAG Platform — Environment Setup (uv)
# Run from the project root: bash stage1/setup.sh
# ─────────────────────────────────────────────────────────────────────

set -e

# ── Resolve project root (where this script lives) ───────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"
echo "  ℹ️  Working directory: $PROJECT_ROOT"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║     DocRAG Platform — Environment Setup     ║"
echo "╚══════════════════════════════════════════════╝"

# ── 1. Install uv if not present ─────────────────────────────────────
echo ""
echo "▶ Checking uv..."
if ! command -v uv &> /dev/null; then
    echo "  ℹ️  uv not found — installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
    echo "  ✅ uv installed"
else
    echo "  ✅ uv $(uv --version) found"
fi

# ── 2. Python version check ──────────────────────────────────────────
echo ""
echo "▶ Checking Python version..."
if uv python find 3.11 &> /dev/null; then
    echo "  ✅ Python 3.11 available"
else
    echo "  ℹ️  Installing Python 3.11 via uv..."
    uv python install 3.11
    echo "  ✅ Python 3.11 installed"
fi

# ── 3. Create virtual environment ────────────────────────────────────
echo ""
echo "▶ Creating virtual environment..."
if [ ! -d ".venv" ]; then
    uv venv .venv --python 3.11
    echo "  ✅ .venv created"
else
    echo "  ℹ️  .venv already exists — skipping"
fi

source .venv/bin/activate

# ── 4. Install PyTorch (CUDA 12.1 for RTX 3060) ──────────────────────
echo ""
echo "▶ Installing PyTorch with CUDA 12.1 support..."
uv pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121
echo "  ✅ PyTorch installed"

# ── 5. Install Unsloth ───────────────────────────────────────────────
echo ""
echo "▶ Installing Unsloth..."
uv pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
echo "  ✅ Unsloth installed"

# ── 6. Install llama-cpp-python with CUDA ────────────────────────────
echo ""
echo "▶ Installing llama-cpp-python with CUDA support..."
CMAKE_ARGS="-DLLAMA_CUDA=on" uv pip install llama-cpp-python
echo "  ✅ llama-cpp-python installed with CUDA"

# ── 7. Install remaining requirements ────────────────────────────────
echo ""
echo "▶ Installing remaining requirements..."
REQUIREMENTS="$PROJECT_ROOT/requirements.txt"
if [ ! -f "$REQUIREMENTS" ]; then
    echo "  ❌ requirements.txt not found at $REQUIREMENTS"
    exit 1
fi
grep -v -E "^(torch|unsloth|llama-cpp|#|\s*$)" "$REQUIREMENTS" \
    | uv pip install -r /dev/stdin
echo "  ✅ All requirements installed"

# ── 8. Initialize Git + DVC ──────────────────────────────────────────
echo ""
echo "▶ Initializing Git..."
if [ ! -d ".git" ]; then
    git init
    echo "  ✅ Git initialized"
else
    echo "  ℹ️  Git already initialized"
fi

echo ""
echo "▶ Initializing DVC..."
DVC_BIN="$PROJECT_ROOT/.venv/bin/dvc"
if [ ! -d ".dvc" ]; then
    "$DVC_BIN" init
    git add .dvc .dvcignore
    echo "  ✅ DVC initialized"
else
    echo "  ℹ️  DVC already initialized"
fi

# ── 9. Create .env from template ─────────────────────────────────────
echo ""
echo "▶ Setting up environment variables..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "  ✅ .env created from .env.example"
    echo "  ⚠️  IMPORTANT: Edit .env and add your API keys before running"
else
    echo "  ℹ️  .env already exists — skipping"
fi

# ── 10. Create required directories ──────────────────────────────────
echo ""
echo "▶ Creating project directories..."
mkdir -p data/raw data/processed data/training models evaluation/results
echo "  ✅ Directories created"

# ── 11. Start Qdrant ─────────────────────────────────────────────────
echo ""
echo "▶ Starting Qdrant vector database..."
if command -v docker &> /dev/null; then
    docker compose up -d qdrant
    echo "  ✅ Qdrant started at http://localhost:6333"
    echo "  ℹ️  Dashboard: http://localhost:6333/dashboard"
else
    echo "  ⚠️  Docker not found. Start Qdrant manually:"
    echo "      docker compose up -d qdrant"
fi

# ── 12. Verify GPU ───────────────────────────────────────────────────
echo ""
echo "▶ Checking GPU availability..."
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'  ✅ GPU found: {gpu} ({vram:.1f} GB VRAM)')
else:
    print('  ⚠️  No GPU found — fine-tuning will be very slow on CPU')
"

# ── Done ─────────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║           Setup Complete! ✅                 ║"
echo "╚══════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "  1. Edit .env  →  add GROQ_API_KEY and WANDB_API_KEY"
echo "  2. Run: wandb login"
echo "  3. Run: python verify_setup.py"
echo "  4. Run: python ingestion/collect.py"
echo ""
echo "Activate .venv in future sessions:"
echo "  source .venv/bin/activate"