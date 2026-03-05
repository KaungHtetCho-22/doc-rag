"""
ingestion/collect.py
─────────────────────────────────────────────────────────────────────
Milestone 1.2 — Data Collection

Downloads Computer Vision papers (CVPR/ICCV) from arXiv and enriches
them with metadata from Papers With Code API.

Outputs:
    data/raw/pdfs/          ← downloaded PDF files
    data/raw/metadata.jsonl ← per-paper metadata (title, authors, etc.)

Usage:
    python ingestion/collect.py
    python ingestion/collect.py --max 150 --query "object detection transformer"
"""

import os
import json
import time
import argparse
import hashlib
from pathlib import Path
from datetime import datetime

import arxiv
import requests
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
RAW_DIR      = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
PDF_DIR      = RAW_DIR / "pdfs"
METADATA_FILE = RAW_DIR / "metadata.jsonl"

# CV-focused search queries hitting CVPR/ICCV topics
CV_QUERIES = [
    "object detection transformer CVPR",
    "image segmentation deep learning CVPR",
    "visual representation learning self-supervised CVPR",
    "3D scene understanding point cloud ICCV",
    "generative adversarial network image synthesis CVPR",
    "video understanding temporal CVPR",
    "multi-modal vision language CVPR",
    "neural radiance field NeRF ICCV",
    "optical flow scene estimation ICCV",
    "image classification convolutional CVPR",
]

PWC_API      = "https://paperswithcode.com/api/v1/papers/"
PWC_DELAY    = 0.5   # seconds between PWC requests (be polite)
ARXIV_DELAY  = 3.0   # seconds between arXiv batch requests


# ── Papers With Code enrichment ───────────────────────────────────────
def fetch_pwc_metadata(arxiv_id: str) -> dict:
    """
    Try to fetch extra metadata from Papers With Code for a given arXiv ID.
    Returns empty dict if not found — PWC coverage is ~60% of arXiv papers.
    """
    try:
        # PWC uses arxiv IDs without version suffix
        clean_id = arxiv_id.split("v")[0]
        url = f"{PWC_API}arxiv-{clean_id}/"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            return {
                "pwc_id":          data.get("id"),
                "pwc_url":         data.get("url_abs"),
                "has_code":        bool(data.get("github_link")),
                "github_url":      data.get("github_link"),
                "tasks":           [t.get("name") for t in data.get("tasks", [])],
                "methods":         [m.get("name") for m in data.get("methods", [])],
                "conference":      data.get("proceeding", {}).get("acronym") if data.get("proceeding") else None,
            }
    except Exception as e:
        logger.debug(f"PWC lookup failed for {arxiv_id}: {e}")
    return {}


# ── arXiv collection ─────────────────────────────────────────────────
def collect_papers(queries: list[str], max_total: int) -> list[dict]:
    """
    Search arXiv across multiple CV queries, deduplicate, and return
    a list of paper metadata dicts up to max_total papers.
    """
    client   = arxiv.Client(page_size=50, delay_seconds=ARXIV_DELAY)
    seen_ids = set()
    papers   = []

    # Load already-collected IDs to allow resuming
    if METADATA_FILE.exists():
        with open(METADATA_FILE) as f:
            for line in f:
                entry = json.loads(line)
                seen_ids.add(entry["arxiv_id"])
        logger.info(f"Resuming — {len(seen_ids)} papers already collected")

    per_query = max(max_total // len(queries), 20)

    for query in queries:
        if len(papers) + len(seen_ids) >= max_total:
            break

        logger.info(f"Searching: '{query}' (up to {per_query} results)")

        search = arxiv.Search(
            query       = query,
            max_results = per_query,
            sort_by     = arxiv.SortCriterion.Relevance,
        )

        try:
            for result in client.results(search):
                arxiv_id = result.get_short_id()

                if arxiv_id in seen_ids:
                    continue

                # Only include cs.CV, cs.LG, cs.AI categories
                categories = result.categories or []
                if not any(c in categories for c in ["cs.CV", "cs.LG", "cs.AI", "eess.IV"]):
                    continue

                paper = {
                    "arxiv_id":    arxiv_id,
                    "title":       result.title,
                    "authors":     [a.name for a in result.authors],
                    "abstract":    result.summary.replace("\n", " "),
                    "categories":  categories,
                    "published":   result.published.isoformat() if result.published else None,
                    "updated":     result.updated.isoformat()   if result.updated   else None,
                    "year":        result.published.year         if result.published else None,
                    "pdf_url":     result.pdf_url,
                    "arxiv_url":   result.entry_id,
                    "query_hit":   query,
                    "pdf_path":    None,   # filled after download
                    "pwc":         {},     # filled after PWC enrichment
                }

                papers.append(paper)
                seen_ids.add(arxiv_id)

                if len(papers) + (METADATA_FILE.exists() and sum(1 for _ in open(METADATA_FILE))) >= max_total:
                    break

        except Exception as e:
            logger.warning(f"arXiv search failed for '{query}': {e}")
            continue

    logger.info(f"Found {len(papers)} new papers across all queries")
    return papers


# ── PDF download ──────────────────────────────────────────────────────
def download_pdf(paper: dict) -> str | None:
    """Download PDF for a paper. Returns local path or None on failure."""
    arxiv_id  = paper["arxiv_id"]
    safe_name = arxiv_id.replace("/", "_").replace(".", "_")
    pdf_path  = PDF_DIR / f"{safe_name}.pdf"

    if pdf_path.exists():
        return str(pdf_path)

    try:
        resp = requests.get(paper["pdf_url"], timeout=30, stream=True)
        resp.raise_for_status()

        with open(pdf_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        # Sanity check — real PDFs start with %PDF
        with open(pdf_path, "rb") as f:
            header = f.read(4)
        if header != b"%PDF":
            logger.warning(f"Invalid PDF for {arxiv_id} — deleting")
            pdf_path.unlink()
            return None

        return str(pdf_path)

    except Exception as e:
        logger.warning(f"Download failed for {arxiv_id}: {e}")
        if pdf_path.exists():
            pdf_path.unlink()
        return None


# ── Main ─────────────────────────────────────────────────────────────
def main(max_papers: int, custom_query: str | None):
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    queries = CV_QUERIES.copy()
    if custom_query:
        queries.insert(0, custom_query)

    # ── Step 1: Collect metadata from arXiv ──────────────────────────
    logger.info("═" * 55)
    logger.info("  Step 1/3 — Searching arXiv")
    logger.info("═" * 55)
    papers = collect_papers(queries, max_papers)

    if not papers:
        logger.warning("No new papers found — already at target or arXiv unreachable")
        return

    # ── Step 2: Enrich with Papers With Code ─────────────────────────
    logger.info("")
    logger.info("═" * 55)
    logger.info("  Step 2/3 — Enriching with Papers With Code")
    logger.info("═" * 55)

    for paper in tqdm(papers, desc="PWC lookup"):
        paper["pwc"] = fetch_pwc_metadata(paper["arxiv_id"])
        time.sleep(PWC_DELAY)

    has_code  = sum(1 for p in papers if p["pwc"].get("has_code"))
    has_conf  = sum(1 for p in papers if p["pwc"].get("conference"))
    logger.info(f"PWC enrichment: {has_code}/{len(papers)} have code | {has_conf}/{len(papers)} have conference")

    # ── Step 3: Download PDFs ─────────────────────────────────────────
    logger.info("")
    logger.info("═" * 55)
    logger.info("  Step 3/3 — Downloading PDFs")
    logger.info("═" * 55)

    failed = 0
    for paper in tqdm(papers, desc="Downloading PDFs"):
        path = download_pdf(paper)
        if path:
            paper["pdf_path"] = path
        else:
            failed += 1
        time.sleep(0.5)  # polite delay

    downloaded = len(papers) - failed
    logger.info(f"Downloaded {downloaded}/{len(papers)} PDFs  ({failed} failed)")

    # ── Save metadata ─────────────────────────────────────────────────
    with open(METADATA_FILE, "a") as f:
        for paper in papers:
            f.write(json.dumps(paper) + "\n")

    logger.info("")
    logger.info("═" * 55)
    logger.info("  Collection Complete ✅")
    logger.info("═" * 55)
    logger.info(f"  Papers collected : {len(papers)}")
    logger.info(f"  PDFs downloaded  : {downloaded}")
    logger.info(f"  Metadata saved   : {METADATA_FILE}")
    logger.info(f"  PDF directory    : {PDF_DIR}")

    # ── Print sample ──────────────────────────────────────────────────
    logger.info("")
    logger.info("Sample papers:")
    for p in papers[:3]:
        conf = p["pwc"].get("conference") or "unknown conf"
        code = "✅ code" if p["pwc"].get("has_code") else "no code"
        logger.info(f"  [{p['year']}] [{conf}] [{code}] {p['title'][:70]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect CV papers from arXiv + Papers With Code")
    parser.add_argument("--max",   type=int, default=150,  help="Max papers to collect (default: 150)")
    parser.add_argument("--query", type=str, default=None, help="Optional extra search query")
    args = parser.parse_args()

    logger.info(f"Target: {args.max} papers | Domain: Computer Vision (CVPR/ICCV)")
    main(max_papers=args.max, custom_query=args.query)