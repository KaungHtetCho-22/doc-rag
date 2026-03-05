"""
vectorstore/index.py
─────────────────────────────────────────────────────────────────────
Milestone 2.2 — Embedding & Indexing

Reads chunks.jsonl, generates dense + sparse vectors for each chunk,
and upserts them into Qdrant in batches.

Dense  : sentence-transformers/all-MiniLM-L6-v2  (384 dims)
Sparse : FastEmbed BM25

Usage:
    python vectorstore/index.py
    python vectorstore/index.py --batch-size 64 --reindex
"""

import os
import json
import uuid
import argparse
from pathlib import Path

import torch
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
QDRANT_HOST        = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT        = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME    = os.getenv("QDRANT_COLLECTION", "research_papers")
DENSE_MODEL_NAME   = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPARSE_MODEL_NAME  = "Qdrant/bm25"

PROCESSED_DIR      = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
CHUNKS_FILE        = PROCESSED_DIR / "chunks.jsonl"
INDEXED_IDS_FILE   = PROCESSED_DIR / ".indexed_ids"   # tracks already-indexed chunk_ids

DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"
DEFAULT_BATCH_SIZE = 32


# ── Load models ───────────────────────────────────────────────────────
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading dense model on {device}: {DENSE_MODEL_NAME}")
    dense_model = SentenceTransformer(DENSE_MODEL_NAME, device=device)

    logger.info(f"Loading sparse model: {SPARSE_MODEL_NAME}")
    sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)

    return dense_model, sparse_model


# ── Load chunks ───────────────────────────────────────────────────────
def load_chunks(reindex: bool) -> list[dict]:
    if not CHUNKS_FILE.exists():
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        logger.error("Run ingestion/parse.py first.")
        raise SystemExit(1)

    # Load already-indexed chunk IDs to support resuming
    indexed_ids = set()
    if not reindex and INDEXED_IDS_FILE.exists():
        with open(INDEXED_IDS_FILE) as f:
            indexed_ids = set(line.strip() for line in f if line.strip())
        logger.info(f"Resuming — {len(indexed_ids)} chunks already indexed")

    chunks = []
    with open(CHUNKS_FILE) as f:
        for line in f:
            chunk = json.loads(line)
            if chunk["chunk_id"] not in indexed_ids:
                chunks.append(chunk)

    logger.info(f"Chunks to index: {len(chunks)}")
    return chunks


# ── Build Qdrant points ───────────────────────────────────────────────
def build_points(
    chunks: list[dict],
    dense_model: SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    batch_size: int,
) -> list[PointStruct]:
    """
    Embed a batch of chunks and return Qdrant PointStructs.
    Each point has:
      - a UUID derived from chunk_id (deterministic)
      - dense vector (semantic embedding)
      - sparse vector (BM25 keyword weights)
      - full metadata payload for filtering
    """
    points = []
    texts  = [c["text"] for c in chunks]

    # ── Dense embeddings (GPU-accelerated) ───────────────────────────
    dense_vecs = dense_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine sim = dot product after normalisation
        convert_to_numpy=True,
    )

    # ── Sparse embeddings (CPU, BM25) ─────────────────────────────────
    sparse_results = list(sparse_model.embed(texts))

    for i, chunk in enumerate(chunks):
        # Deterministic UUID from chunk_id so re-indexing is idempotent
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"]))

        sparse = sparse_results[i]

        # Build payload — everything Qdrant will store alongside the vector
        payload = {
            "chunk_id":      chunk["chunk_id"],
            "text":          chunk["text"],
            "word_count":    chunk["word_count"],
            "chunk_index":   chunk["chunk_index"],
            "section":       chunk.get("section", "Unknown"),
            "section_chunk": chunk.get("section_chunk", 0),

            # Paper identity
            "arxiv_id":      chunk["arxiv_id"],
            "title":         chunk["title"],
            "authors":       chunk.get("authors", []),
            "year":          chunk.get("year"),
            "abstract":      chunk.get("abstract", ""),

            # PWC enrichment
            "conference":    chunk.get("conference"),
            "has_code":      chunk.get("has_code", False),
            "github_url":    chunk.get("github_url"),
            "tasks":         chunk.get("tasks", []),

            # Source
            "arxiv_url":     chunk.get("arxiv_url"),
            "pdf_filename":  chunk.get("pdf_filename"),
        }

        points.append(PointStruct(
            id      = point_id,
            vector  = {
                DENSE_VECTOR_NAME: dense_vecs[i].tolist(),
                SPARSE_VECTOR_NAME: SparseVector(
                    indices = sparse.indices.tolist(),
                    values  = sparse.values.tolist(),
                ),
            },
            payload = payload,
        ))

    return points


# ── Main ─────────────────────────────────────────────────────────────
def main(batch_size: int, reindex: bool):
    # Connect
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=30)
    logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")

    # Verify collection exists
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in collections:
        logger.error(f"Collection '{COLLECTION_NAME}' not found.")
        logger.error("Run vectorstore/setup_collection.py first.")
        raise SystemExit(1)

    # Load chunks
    chunks = load_chunks(reindex)
    if not chunks:
        logger.info("Nothing to index — all chunks already indexed.")
        return

    # Load embedding models
    dense_model, sparse_model = load_models()

    # ── Index in batches ──────────────────────────────────────────────
    total_indexed  = 0
    newly_indexed  = []

    with open(INDEXED_IDS_FILE, "a") as id_file:
        for start in tqdm(range(0, len(chunks), batch_size), desc="Indexing batches"):
            batch = chunks[start : start + batch_size]

            try:
                points = build_points(batch, dense_model, sparse_model, batch_size)

                client.upsert(
                    collection_name = COLLECTION_NAME,
                    points          = points,
                    wait            = True,
                )

                # Record indexed IDs for resume support
                for chunk in batch:
                    id_file.write(chunk["chunk_id"] + "\n")

                total_indexed += len(batch)

            except Exception as e:
                logger.warning(f"Batch {start}–{start+batch_size} failed: {e}")
                continue

    # ── Final stats ───────────────────────────────────────────────────
    info           = client.get_collection(COLLECTION_NAME)
    total_in_qdrant = info.points_count

    logger.info("")
    logger.info("═" * 55)
    logger.info("  Indexing Complete ✅")
    logger.info("═" * 55)
    logger.info(f"  Chunks indexed this run : {total_indexed}")
    logger.info(f"  Total points in Qdrant  : {total_in_qdrant}")
    logger.info(f"  Collection              : {COLLECTION_NAME}")
    logger.info("═" * 55)
    logger.info("")
    logger.info("Next: python vectorstore/search.py  (test hybrid search)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed and index chunks into Qdrant")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help=f"Embedding batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--reindex",    action="store_true",
                        help="Re-index all chunks even if already indexed")
    args = parser.parse_args()

    main(batch_size=args.batch_size, reindex=args.reindex)