"""
vectorstore/search.py
─────────────────────────────────────────────────────────────────────
Milestone 2.3 — Hybrid Search Verification

Tests hybrid search (dense + sparse) and payload filtering against
the indexed Qdrant collection.

Usage:
    python vectorstore/search.py
    python vectorstore/search.py --query "attention mechanism in transformers"
    python vectorstore/search.py --query "object detection" --year 2023 --has-code
    python vectorstore/search.py --query "segmentation" --section "Methods"
    python vectorstore/search.py --benchmark   # run latency benchmark
"""

import os
import time
import argparse
from dotenv import load_dotenv
from loguru import logger

import torch
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    NamedVector,
    NamedSparseVector,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
QDRANT_HOST        = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT        = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME    = os.getenv("QDRANT_COLLECTION", "research_papers")
DENSE_MODEL_NAME   = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
SPARSE_MODEL_NAME  = "Qdrant/bm25"
DENSE_VECTOR_NAME  = "dense"
SPARSE_VECTOR_NAME = "sparse"
TOP_K              = 5


# ── Model loader ──────────────────────────────────────────────────────
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dense  = SentenceTransformer(DENSE_MODEL_NAME, device=device)
    sparse = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    return dense, sparse


# ── Build Qdrant filter ───────────────────────────────────────────────
def build_filter(year: int | None, has_code: bool | None, section: str | None) -> Filter | None:
    conditions = []

    if year is not None:
        conditions.append(FieldCondition(
            key="year",
            range=Range(gte=year, lte=year)
        ))
    if has_code is not None:
        conditions.append(FieldCondition(
            key="has_code",
            match=MatchValue(value=has_code)
        ))
    if section is not None:
        conditions.append(FieldCondition(
            key="section",
            match=MatchValue(value=section)
        ))

    return Filter(must=conditions) if conditions else None


# ── Hybrid search ─────────────────────────────────────────────────────
def hybrid_search(
    client:       QdrantClient,
    dense_model:  SentenceTransformer,
    sparse_model: SparseTextEmbedding,
    query:        str,
    top_k:        int = TOP_K,
    qdrant_filter: Filter | None = None,
) -> tuple[list, float]:
    """
    Run hybrid search: dense (semantic) + sparse (BM25 keyword).
    Returns results and latency in ms.
    """
    # Embed query
    dense_vec  = dense_model.encode(query, normalize_embeddings=True).tolist()
    sparse_vec = list(sparse_model.embed([query]))[0]

    t0 = time.perf_counter()

    # ── Dense search ─────────────────────────────────────────────────
    dense_hits = client.search(
        collection_name = COLLECTION_NAME,
        query_vector    = NamedVector(name=DENSE_VECTOR_NAME, vector=dense_vec),
        limit           = top_k * 3,
        query_filter    = qdrant_filter,
        with_payload    = True,
    )

    # ── Sparse search ─────────────────────────────────────────────────
    sparse_hits = client.search(
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

    # ── RRF fusion (manual) ───────────────────────────────────────────
    # Reciprocal Rank Fusion merges dense + sparse rankings
    RRF_K   = 60
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

    latency_ms = (time.perf_counter() - t0) * 1000
    return results, latency_ms


# ── Display results ───────────────────────────────────────────────────
def display_results(results: list, query: str, latency_ms: float, filters: dict):
    print()
    print("═" * 70)
    print(f"  Query   : {query}")
    if filters:
        print(f"  Filters : {filters}")
    print(f"  Latency : {latency_ms:.1f} ms  |  Results: {len(results)}")
    print("═" * 70)

    for i, r in enumerate(results, 1):
        p = r.payload
        conf     = p.get("conference") or "—"
        code     = "✅ code" if p.get("has_code") else "no code"
        year     = p.get("year") or "—"
        section  = p.get("section") or "—"
        score    = getattr(r, "score", 0) or 0
        title    = p.get("title", "")[:65]
        text     = p.get("text", "")[:200].replace("\n", " ")

        print(f"\n  [{i}] score={score:.4f}")
        print(f"       {title}")
        print(f"       [{year}] [{conf}] [{section}] [{code}]")
        print(f"       \"{text}...\"")

    print()


# ── Benchmark ─────────────────────────────────────────────────────────
def run_benchmark(client, dense_model, sparse_model):
    benchmark_queries = [
        ("object detection transformer",          None),
        ("self-supervised visual representation", None),
        ("3D point cloud segmentation",           None),
        ("image generation GAN",                  None),
        ("optical flow estimation",               None),
    ]

    print()
    print("═" * 70)
    print("  Latency Benchmark (10 runs per query)")
    print("═" * 70)

    for query, _ in benchmark_queries:
        latencies = []
        for _ in range(10):
            _, lat = hybrid_search(client, dense_model, sparse_model, query)
            latencies.append(lat)
        avg = sum(latencies) / len(latencies)
        p95 = sorted(latencies)[int(0.95 * len(latencies))]
        print(f"  {query[:45]:<45}  avg={avg:.1f}ms  p95={p95:.1f}ms")

    print()
    info = client.get_collection(COLLECTION_NAME)
    print(f"  Collection: {COLLECTION_NAME}  |  Points: {info.points_count}")
    print("═" * 70)
    print()


# ── Predefined demo queries ───────────────────────────────────────────
DEMO_QUERIES = [
    {
        "query":    "attention mechanism for visual feature extraction",
        "year":     None,
        "has_code": None,
        "section":  None,
    },
    {
        "query":    "training loss function for object detection",
        "year":     None,
        "has_code": None,
        "section":  "Results",
    },
    {
        "query":    "benchmark results on COCO dataset",
        "year":     None,
        "has_code": None,
        "section":  "Results",
    },
]


# ── Main ─────────────────────────────────────────────────────────────
def main(query: str | None, year: int | None, has_code: bool | None,
         section: str | None, benchmark: bool):

    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
    info   = client.get_collection(COLLECTION_NAME)
    logger.info(f"Connected — {info.points_count} points in '{COLLECTION_NAME}'")

    logger.info("Loading embedding models...")
    dense_model, sparse_model = load_models()

    if benchmark:
        run_benchmark(client, dense_model, sparse_model)
        return

    if query:
        # Single query from CLI args
        qdrant_filter = build_filter(year, has_code, section)
        results, latency = hybrid_search(
            client, dense_model, sparse_model, query,
            qdrant_filter=qdrant_filter
        )
        filters = {k: v for k, v in
                   {"year": year, "has_code": has_code, "section": section}.items()
                   if v is not None}
        display_results(results, query, latency, filters)

    else:
        # Run all demo queries to verify everything works
        logger.info("Running demo queries to verify hybrid search...")
        all_passed = True

        for demo in DEMO_QUERIES:
            qdrant_filter = build_filter(
                demo["year"], demo["has_code"], demo["section"]
            )
            results, latency = hybrid_search(
                client, dense_model, sparse_model,
                demo["query"], qdrant_filter=qdrant_filter
            )
            filters = {k: v for k, v in demo.items()
                       if k != "query" and v is not None}
            display_results(results, demo["query"], latency, filters)

            if not results:
                logger.warning(f"No results for: {demo['query']}")
                all_passed = False

        if all_passed:
            print("✅ All demo queries returned results — hybrid search working!")
            print("   Run with --benchmark to measure latency")
        else:
            print("⚠️  Some queries returned no results — check your index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test hybrid search against Qdrant")
    parser.add_argument("--query",     type=str,  default=None,  help="Search query")
    parser.add_argument("--year",      type=int,  default=None,  help="Filter by year")
    parser.add_argument("--has-code",  action="store_true",      help="Filter: papers with code only")
    parser.add_argument("--section",   type=str,  default=None,  help="Filter by section name")
    parser.add_argument("--benchmark", action="store_true",      help="Run latency benchmark")
    args = parser.parse_args()

    main(
        query     = args.query,
        year      = args.year,
        has_code  = args.has_code or None,
        section   = args.section,
        benchmark = args.benchmark,
    )