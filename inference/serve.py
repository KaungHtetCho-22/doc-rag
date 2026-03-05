"""
inference/serve.py
─────────────────────────────────────────────────────────────────────
Stage 5 — RAG Inference Pipeline

Retrieves relevant chunks from Qdrant, assembles context, and
generates answers using either:
  - Cloud mode  : Groq API (ultra-fast, requires internet)
  - Local mode  : llama.cpp with fine-tuned GGUF (offline, private)

Exposes a FastAPI backend at http://localhost:8000

Endpoints:
    POST /query          ← main RAG query
    GET  /health         ← health check
    GET  /mode           ← current inference mode
    POST /mode/{mode}    ← switch mode (groq | local)
    GET  /stats          ← collection + model stats

Usage:
    python inference/serve.py
    python inference/serve.py --mode local
    python inference/serve.py --port 8000
"""

import os
import time
import argparse
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from loguru import logger

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedVector, NamedSparseVector, SparseVector,
    Filter, FieldCondition, MatchValue, Range,
)
from groq import Groq

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
QDRANT_HOST        = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT        = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME    = os.getenv("QDRANT_COLLECTION", "research_papers")
DENSE_MODEL_NAME   = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPARSE_MODEL_NAME  = "Qdrant/bm25"
DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"

GROQ_API_KEY       = os.getenv("GROQ_API_KEY")
GROQ_MODEL         = os.getenv("GROQ_INFERENCE_MODEL", "llama-3.1-8b-instant")
LOCAL_MODEL_PATH   = os.getenv("LOCAL_MODEL_PATH", "models/docrag-3b-q4.gguf")
INFERENCE_MODE     = os.getenv("INFERENCE_MODE", "groq")  # "groq" | "local"

TOP_K              = 5
MAX_CONTEXT_CHARS  = 4000
RRF_K              = 60

# ── System prompt ─────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert research assistant specializing in computer vision
and machine learning. Answer questions based strictly on the provided research paper excerpts.

Guidelines:
- Ground your answer in the provided context
- Cite which paper each piece of information comes from
- If the context doesn't contain enough information, say so clearly
- Be concise but complete
- Use technical terminology appropriate for ML/CV researchers"""


# ── Global state (loaded once at startup) ─────────────────────────────
class AppState:
    qdrant_client:  QdrantClient        = None
    dense_model:    SentenceTransformer = None
    sparse_model:   SparseTextEmbedding = None
    groq_client:    Groq                = None
    llama_model:    object              = None   # llama_cpp.Llama
    inference_mode: str                 = INFERENCE_MODE
    query_count:    int                 = 0
    total_latency:  float               = 0.0

state = AppState()


# ── Lifespan: load all models at startup ──────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("═" * 55)
    logger.info("  DocRAG Inference Server — Starting Up")
    logger.info("═" * 55)

    # Qdrant
    state.qdrant_client = QdrantClient(
        host=QDRANT_HOST, port=QDRANT_PORT, timeout=10
    )
    info = state.qdrant_client.get_collection(COLLECTION_NAME)
    logger.info(f"Qdrant connected — {info.points_count} points")

    # Embedding models (always needed for retrieval)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading dense embedder on {device}...")
    state.dense_model  = SentenceTransformer(DENSE_MODEL_NAME, device=device)
    logger.info("Loading sparse embedder (BM25)...")
    state.sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    # Groq client
    if GROQ_API_KEY and GROQ_API_KEY != "your_groq_api_key_here":
        state.groq_client = Groq(api_key=GROQ_API_KEY)
        logger.info(f"Groq client ready — model: {GROQ_MODEL}")
    else:
        logger.warning("GROQ_API_KEY not set — cloud mode unavailable")

    # llama.cpp (only load if local mode requested or GGUF exists)
    gguf_path = Path(LOCAL_MODEL_PATH)
    if gguf_path.exists():
        logger.info(f"Loading local GGUF model: {gguf_path}")
        try:
            from llama_cpp import Llama
            state.llama_model = Llama(
                model_path   = str(gguf_path),
                n_ctx        = 4096,
                n_gpu_layers = -1,   # offload all layers to GPU
                verbose      = False,
            )
            logger.info("Local GGUF model loaded ✅")
        except Exception as e:
            logger.warning(f"Failed to load GGUF model: {e}")
            logger.warning("Local mode will be unavailable")
    else:
        logger.warning(f"GGUF not found at {gguf_path} — local mode unavailable")
        if state.inference_mode == "local":
            logger.warning("Falling back to Groq mode")
            state.inference_mode = "groq"

    logger.info(f"Inference mode: {state.inference_mode.upper()}")
    logger.info("Server ready ✅")
    logger.info("═" * 55)

    yield  # server runs here

    logger.info("Shutting down...")


# ── FastAPI app ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "DocRAG Intelligence Platform",
    description = "Hybrid RAG over CV/ML research papers",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)


# ── Request / Response models ─────────────────────────────────────────
class QueryRequest(BaseModel):
    query:    str
    top_k:    int           = TOP_K
    year:     Optional[int] = None
    has_code: Optional[bool] = None
    section:  Optional[str] = None
    mode:     Optional[str] = None   # override global mode for this request

class SourceChunk(BaseModel):
    title:      str
    arxiv_id:   str
    year:       Optional[int]
    section:    str
    conference: Optional[str]
    has_code:   bool
    text:       str
    score:      float

class QueryResponse(BaseModel):
    answer:        str
    sources:       list[SourceChunk]
    mode:          str
    latency_ms:    float
    retrieval_ms:  float
    generation_ms: float
    query:         str


# ── Retrieval ─────────────────────────────────────────────────────────
def retrieve(query: str, top_k: int, qdrant_filter: Filter | None) -> tuple[list, float]:
    """Hybrid search: dense + sparse with RRF fusion."""
    dense_vec  = state.dense_model.encode(query, normalize_embeddings=True).tolist()
    sparse_vec = list(state.sparse_model.embed([query]))[0]

    t0 = time.perf_counter()

    dense_hits = state.qdrant_client.search(
        collection_name = COLLECTION_NAME,
        query_vector    = NamedVector(name=DENSE_VECTOR_NAME, vector=dense_vec),
        limit           = top_k * 3,
        query_filter    = qdrant_filter,
        with_payload    = True,
    )

    sparse_hits = state.qdrant_client.search(
        collection_name = COLLECTION_NAME,
        query_vector    = NamedSparseVector(
            name   = SPARSE_VECTOR_NAME,
            vector = SparseVector(
                indices = sparse_vec.indices.tolist(),
                values  = sparse_vec.values.tolist(),
            ),
        ),
        limit        = top_k * 3,
        query_filter = qdrant_filter,
        with_payload = True,
    )

    # RRF fusion
    scores  : dict = {}
    hit_map : dict = {}
    for rank, hit in enumerate(dense_hits):
        pid = str(hit.id)
        scores[pid]  = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        hit_map[pid] = hit
    for rank, hit in enumerate(sparse_hits):
        pid = str(hit.id)
        scores[pid]  = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)
        hit_map[pid] = hit

    top_ids = sorted(scores, key=lambda x: -scores[x])[:top_k]
    results = []
    for pid in top_ids:
        hit       = hit_map[pid]
        hit.score = scores[pid]
        results.append(hit)

    retrieval_ms = (time.perf_counter() - t0) * 1000
    return results, retrieval_ms


# ── Context assembly ──────────────────────────────────────────────────
def assemble_context(hits: list) -> str:
    """Build a context string from retrieved chunks with source labels."""
    parts = []
    total = 0
    for i, hit in enumerate(hits, 1):
        p     = hit.payload
        title = p.get("title", "Unknown")[:60]
        year  = p.get("year", "?")
        sec   = p.get("section", "")
        text  = p.get("text", "")

        chunk = f"[{i}] {title} ({year}) — {sec}\n{text}"

        if total + len(chunk) > MAX_CONTEXT_CHARS:
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n\n---\n\n".join(parts)


# ── Generation: Groq ──────────────────────────────────────────────────
def generate_groq(query: str, context: str) -> tuple[str, float]:
    if not state.groq_client:
        raise HTTPException(503, "Groq client not available — check GROQ_API_KEY")

    messages = [
        {"role": "system",  "content": SYSTEM_PROMPT},
        {"role": "user",    "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ]

    t0       = time.perf_counter()
    response = state.groq_client.chat.completions.create(
        model       = GROQ_MODEL,
        messages    = messages,
        temperature = 0.3,
        max_tokens  = 1024,
    )
    gen_ms = (time.perf_counter() - t0) * 1000
    return response.choices[0].message.content.strip(), gen_ms


# ── Generation: llama.cpp ─────────────────────────────────────────────
def generate_local(query: str, context: str) -> tuple[str, float]:
    if not state.llama_model:
        raise HTTPException(503, "Local GGUF model not loaded — run training first")

    prompt = (
        f"<|system|>\n{SYSTEM_PROMPT}\n"
        f"<|user|>\nContext:\n{context}\n\nQuestion: {query}\n"
        f"<|assistant|>\n"
    )

    t0     = time.perf_counter()
    output = state.llama_model(
        prompt,
        max_tokens  = 1024,
        temperature = 0.3,
        stop        = ["<|user|>", "<|system|>"],
    )
    gen_ms = (time.perf_counter() - t0) * 1000
    answer = output["choices"][0]["text"].strip()
    return answer, gen_ms


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":     "ok",
        "mode":       state.inference_mode,
        "qdrant":     state.qdrant_client is not None,
        "groq":       state.groq_client is not None,
        "local_model": state.llama_model is not None,
    }


@app.get("/mode")
def get_mode():
    return {"mode": state.inference_mode}


@app.post("/mode/{mode}")
def set_mode(mode: str):
    if mode not in ("groq", "local"):
        raise HTTPException(400, "Mode must be 'groq' or 'local'")
    if mode == "local" and not state.llama_model:
        raise HTTPException(503, "Local model not loaded — GGUF file not found")
    if mode == "groq" and not state.groq_client:
        raise HTTPException(503, "Groq client not available")
    state.inference_mode = mode
    logger.info(f"Mode switched → {mode.upper()}")
    return {"mode": mode, "status": "ok"}


@app.get("/stats")
def stats():
    info = state.qdrant_client.get_collection(COLLECTION_NAME)
    avg_latency = (
        state.total_latency / state.query_count
        if state.query_count > 0 else 0
    )
    return {
        "collection":    COLLECTION_NAME,
        "total_points":  info.points_count,
        "queries_served": state.query_count,
        "avg_latency_ms": round(avg_latency, 1),
        "inference_mode": state.inference_mode,
        "dense_model":   DENSE_MODEL_NAME,
        "groq_model":    GROQ_MODEL,
        "local_model":   LOCAL_MODEL_PATH,
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t_total = time.perf_counter()

    # Build metadata filter
    conditions = []
    if req.year:
        conditions.append(FieldCondition(key="year",     range=Range(gte=req.year, lte=req.year)))
    if req.has_code is not None:
        conditions.append(FieldCondition(key="has_code", match=MatchValue(value=req.has_code)))
    if req.section:
        conditions.append(FieldCondition(key="section",  match=MatchValue(value=req.section)))
    qdrant_filter = Filter(must=conditions) if conditions else None

    # Retrieve
    hits, retrieval_ms = retrieve(req.query, req.top_k, qdrant_filter)
    if not hits:
        raise HTTPException(404, "No relevant chunks found for this query")

    # Assemble context
    context = assemble_context(hits)

    # Generate
    mode = req.mode or state.inference_mode
    if mode == "groq":
        answer, gen_ms = generate_groq(req.query, context)
    else:
        answer, gen_ms = generate_local(req.query, context)

    total_ms = (time.perf_counter() - t_total) * 1000

    # Update stats
    state.query_count  += 1
    state.total_latency += total_ms

    logger.info(
        f"Query [{mode}] | retrieval={retrieval_ms:.0f}ms "
        f"gen={gen_ms:.0f}ms total={total_ms:.0f}ms | '{req.query[:50]}'"
    )

    # Build source list
    sources = [
        SourceChunk(
            title      = h.payload.get("title", "")[:80],
            arxiv_id   = h.payload.get("arxiv_id", ""),
            year       = h.payload.get("year"),
            section    = h.payload.get("section", ""),
            conference = h.payload.get("conference"),
            has_code   = h.payload.get("has_code", False),
            text       = h.payload.get("text", "")[:300],
            score      = round(h.score, 4),
        )
        for h in hits
    ]

    return QueryResponse(
        answer        = answer,
        sources       = sources,
        mode          = mode,
        latency_ms    = round(total_ms, 1),
        retrieval_ms  = round(retrieval_ms, 1),
        generation_ms = round(gen_ms, 1),
        query         = req.query,
    )


# ── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    parser = argparse.ArgumentParser(description="DocRAG inference server")
    parser.add_argument("--mode",  default=INFERENCE_MODE, choices=["groq", "local"],
                        help="Inference mode (default: from .env)")
    parser.add_argument("--port",  type=int, default=8000,
                        help="Port to listen on (default: 8000)")
    parser.add_argument("--host",  default="0.0.0.0",
                        help="Host to bind (default: 0.0.0.0)")
    args = parser.parse_args()

    # Override mode from CLI
    state.inference_mode = args.mode

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")