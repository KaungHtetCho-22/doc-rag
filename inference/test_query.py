"""
inference/test_query.py
─────────────────────────────────────────────────────────────────────
Quick CLI test for the inference server.
Run after: python inference/serve.py  (in a separate terminal)

Usage:
    python inference/test_query.py
    python inference/test_query.py --query "how does DETR work?"
    python inference/test_query.py --mode local
    python inference/test_query.py --benchmark
"""

import argparse
import time
import requests

BASE_URL = "http://localhost:8000"

TEST_QUERIES = [
    {"query": "How do transformer architectures improve object detection?", "top_k": 5},
    {"query": "What loss functions are used for training segmentation models?",  "top_k": 5},
    {"query": "What are the main benchmarks for evaluating 3D point cloud methods?", "top_k": 5},
]


def check_health():
    try:
        r    = requests.get(f"{BASE_URL}/health", timeout=5)
        data = r.json()
        print(f"  Status  : {data['status']}")
        print(f"  Mode    : {data['mode']}")
        print(f"  Qdrant  : {'✅' if data['qdrant'] else '❌'}")
        print(f"  Groq    : {'✅' if data['groq'] else '❌'}")
        print(f"  Local   : {'✅' if data['local_model'] else '⚠️  (needs GGUF)'}")
        return True
    except Exception as e:
        print(f"  ❌ Server not reachable: {e}")
        print(f"  Run: python inference/serve.py  (in a separate terminal)")
        return False


def run_query(query_params: dict, mode: str | None = None):
    if mode:
        query_params["mode"] = mode

    r    = requests.post(f"{BASE_URL}/query", json=query_params, timeout=60)
    data = r.json()

    # Handle errors gracefully
    if r.status_code != 200:
        print(f"\n  ⚠️  Query failed ({r.status_code}): {data.get('detail', 'unknown error')}")
        print(f"  Query: {query_params['query']}")
        return

    print(f"\n{'═'*65}")
    print(f"  Query   : {data['query']}")
    print(f"  Mode    : {data['mode'].upper()}")
    print(f"  Latency : total={data['latency_ms']}ms | retrieval={data['retrieval_ms']}ms | gen={data['generation_ms']}ms")
    print(f"{'─'*65}")
    print(f"\n  ANSWER:\n  {data['answer'][:600]}")
    print(f"\n  SOURCES ({len(data['sources'])}):")
    for i, src in enumerate(data["sources"], 1):
        conf = src["conference"] or "—"
        code = "✅" if src["has_code"] else "  "
        print(f"  [{i}] {code} [{src['year']}] [{conf}] {src['title'][:55]}")
        print(f"       section: {src['section'][:40]}  score: {src['score']}")


def run_benchmark(mode: str | None = None):
    print(f"\n{'═'*65}")
    print("  Benchmark — 10 queries")
    print(f"{'═'*65}")

    latencies = []
    for _ in range(10):
        payload = {"query": "attention mechanism for feature extraction", "top_k": 5}
        if mode:
            payload["mode"] = mode
        t0 = time.perf_counter()
        requests.post(f"{BASE_URL}/query", json=payload, timeout=60)
        latencies.append((time.perf_counter() - t0) * 1000)

    avg = sum(latencies) / len(latencies)
    p95 = sorted(latencies)[int(0.95 * len(latencies))]
    print(f"  avg={avg:.0f}ms  p95={p95:.0f}ms  min={min(latencies):.0f}ms  max={max(latencies):.0f}ms")

    stats = requests.get(f"{BASE_URL}/stats").json()
    print(f"\n  Server stats:")
    print(f"  Queries served  : {stats['queries_served']}")
    print(f"  Avg latency     : {stats['avg_latency_ms']}ms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query",     default=None)
    parser.add_argument("--mode",      default=None, choices=["groq", "local"])
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    print(f"\n{'═'*65}")
    print("  DocRAG — Inference Test")
    print(f"{'═'*65}")
    if not check_health():
        raise SystemExit(1)

    if args.benchmark:
        run_benchmark(args.mode)
    elif args.query:
        run_query({"query": args.query, "top_k": 5}, args.mode)
    else:
        for q in TEST_QUERIES:
            run_query(q, args.mode)