"""
vectorstore/setup_collection.py
─────────────────────────────────────────────────────────────────────
Milestone 2.1 — Qdrant Collection Setup

Creates the Qdrant collection with:
  - Named dense vector  (sentence-transformers, 384 dims)
  - Named sparse vector (BM25 via FastEmbed)
  - Payload schema with indexes for fast metadata filtering

Run this ONCE before indexing. Safe to re-run — skips if collection
already exists unless --recreate is passed.

Usage:
    python vectorstore/setup_collection.py
    python vectorstore/setup_collection.py --recreate   # wipe and rebuild
"""

import os
import argparse
from dotenv import load_dotenv
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    Distance,
    PayloadSchemaType,
    TextIndexParams,
    TokenizerType,
)

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
QDRANT_HOST       = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT       = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME   = os.getenv("QDRANT_COLLECTION", "research_papers")

# Must match the embedding model used in index.py
DENSE_MODEL       = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DENSE_DIM         = 384    # all-MiniLM-L6-v2 output dimension
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


# ── Payload fields to index for fast filtering ────────────────────────
#
# Qdrant can filter on any payload field, but creating explicit indexes
# makes filtering dramatically faster on large collections.
#
PAYLOAD_INDEXES = [
    # field_name,         schema_type,                  extra_params
    ("year",              PayloadSchemaType.INTEGER,     {}),
    ("has_code",          PayloadSchemaType.BOOL,        {}),
    ("conference",        PayloadSchemaType.KEYWORD,     {}),
    ("section",           PayloadSchemaType.KEYWORD,     {}),
    ("arxiv_id",          PayloadSchemaType.KEYWORD,     {}),
    ("chunk_index",       PayloadSchemaType.INTEGER,     {}),
    # Full-text index on title for keyword title search
    ("title",             PayloadSchemaType.TEXT,        {
        "text_index_params": TextIndexParams(
            type="text",
            tokenizer=TokenizerType.WORD,
            min_token_len=2,
            max_token_len=20,
            lowercase=True,
        )
    }),
]


def get_client() -> QdrantClient:
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=10)
    logger.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    return client


def collection_exists(client: QdrantClient) -> bool:
    collections = [c.name for c in client.get_collections().collections]
    return COLLECTION_NAME in collections


def delete_collection(client: QdrantClient):
    logger.warning(f"Deleting existing collection: {COLLECTION_NAME}")
    client.delete_collection(COLLECTION_NAME)
    logger.info("Deleted.")


def create_collection(client: QdrantClient):
    logger.info(f"Creating collection: {COLLECTION_NAME}")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            # Dense vector — cosine similarity for semantic search
            DENSE_VECTOR_NAME: VectorParams(
                size=DENSE_DIM,
                distance=Distance.COSINE,
                on_disk=False,   # keep in RAM for speed (fits easily for ~50k chunks)
            ),
        },
        sparse_vectors_config={
            # Sparse vector — dot product for BM25 keyword search
            SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(
                    on_disk=False,
                )
            ),
        },
    )
    logger.info(f"Collection '{COLLECTION_NAME}' created with dense + sparse vectors")


def create_payload_indexes(client: QdrantClient):
    logger.info("Creating payload indexes for fast filtering...")

    for field_name, schema_type, extra in PAYLOAD_INDEXES:
        try:
            if schema_type == PayloadSchemaType.TEXT and extra:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=extra["text_index_params"],
                )
            else:
                client.create_payload_index(
                    collection_name=COLLECTION_NAME,
                    field_name=field_name,
                    field_schema=schema_type,
                )
            logger.info(f"  ✅ Index created: {field_name} ({schema_type})")
        except Exception as e:
            logger.warning(f"  ⚠️  Index failed for {field_name}: {e}")

    logger.info("Payload indexes done.")


def print_collection_info(client: QdrantClient):
    info = client.get_collection(COLLECTION_NAME)
    logger.info("")
    logger.info("═" * 55)
    logger.info("  Collection Info")
    logger.info("═" * 55)
    logger.info(f"  Name        : {COLLECTION_NAME}")
    logger.info(f"  Status      : {info.status}")
    logger.info(f"  Points      : {info.points_count}")
    logger.info(f"  Dense dim   : {DENSE_DIM} ({DENSE_VECTOR_NAME})")
    logger.info(f"  Sparse      : {SPARSE_VECTOR_NAME} (BM25)")
    disk_mb = getattr(info, "disk_data_size", 0) or 0
    logger.info(f"  Disk usage  : {disk_mb / 1e6:.1f} MB")
    logger.info("═" * 55)


def main(recreate: bool):
    client = get_client()

    if collection_exists(client):
        if recreate:
            delete_collection(client)
        else:
            logger.info(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")
            logger.info("Pass --recreate to wipe and rebuild.")
            print_collection_info(client)
            return

    create_collection(client)
    create_payload_indexes(client)
    print_collection_info(client)

    logger.info("")
    logger.info("✅ Qdrant collection ready.")
    logger.info("   Next: python vectorstore/index.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create Qdrant collection for DocRAG")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete and recreate the collection if it already exists")
    args = parser.parse_args()
    main(recreate=args.recreate)