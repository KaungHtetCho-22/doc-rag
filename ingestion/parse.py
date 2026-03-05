"""
ingestion/parse.py
─────────────────────────────────────────────────────────────────────
Milestone 1.3 — Docling Parsing

Converts raw PDFs into clean, section-aware text chunks ready for
embedding and storage in Qdrant.

Pipeline per PDF:
    PDF → Docling → structured document → section chunks → JSONL

Outputs:
    data/processed/chunks.jsonl     ← all chunks across all papers
    data/processed/parse_report.json ← success/failure stats

Usage:
    python ingestion/parse.py
    python ingestion/parse.py --workers 2 --max-chunk-tokens 600
"""

import os
import json
import re
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
RAW_DIR        = Path(os.getenv("DATA_RAW_DIR", "data/raw"))
PROCESSED_DIR  = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
PDF_DIR        = RAW_DIR / "pdfs"
METADATA_FILE  = RAW_DIR / "metadata.jsonl"
CHUNKS_FILE    = PROCESSED_DIR / "chunks.jsonl"
REPORT_FILE    = PROCESSED_DIR / "parse_report.json"

# Section headers commonly found in CV/ML papers
KNOWN_SECTIONS = [
    "abstract", "introduction", "related work", "background",
    "method", "methods", "methodology", "approach", "model",
    "architecture", "framework", "proposed method",
    "experiments", "experimental", "experimental setup", "experimental results",
    "results", "evaluation", "analysis", "ablation", "ablation study",
    "discussion", "conclusion", "conclusions", "future work",
    "acknowledgement", "acknowledgements", "references",
]

MAX_CHUNK_TOKENS   = 512    # soft ceiling per chunk (in words, approx)
MIN_CHUNK_TOKENS   = 30     # discard chunks shorter than this
OVERLAP_SENTENCES  = 2      # sentences of overlap between adjacent chunks


# ── Metadata index ────────────────────────────────────────────────────
def load_metadata_index() -> dict:
    """Load metadata.jsonl into a dict keyed by pdf filename."""
    index = {}
    if not METADATA_FILE.exists():
        logger.warning(f"Metadata file not found: {METADATA_FILE}")
        return index
    with open(METADATA_FILE) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("pdf_path"):
                key = Path(entry["pdf_path"]).name
                index[key] = entry
    logger.info(f"Loaded metadata for {len(index)} papers")
    return index


# ── Section detection ─────────────────────────────────────────────────
def detect_section(heading: str) -> str:
    """Normalise a heading string to a canonical section name."""
    h = heading.lower().strip().rstrip(".")
    # strip leading numbers like "1.", "2.1", "III."
    h = re.sub(r"^[\divxlc]+[\.\s]+", "", h).strip()
    for known in KNOWN_SECTIONS:
        if known in h:
            return known.title()
    return heading.strip().title()


# ── Text chunker ──────────────────────────────────────────────────────
def chunk_text(text: str, max_tokens: int, overlap_sentences: int) -> list[str]:
    """
    Split text into chunks that respect sentence boundaries.
    Each chunk is at most max_tokens words long.
    Adjacent chunks share overlap_sentences sentences for context continuity.
    """
    # Split into sentences (simple regex — good enough for paper text)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks    = []
    current   = []
    cur_words = 0

    for sent in sentences:
        words = len(sent.split())
        if cur_words + words > max_tokens and current:
            chunks.append(" ".join(current))
            # keep last N sentences as overlap
            current   = current[-overlap_sentences:] if overlap_sentences else []
            cur_words = sum(len(s.split()) for s in current)
        current.append(sent)
        cur_words += words

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── Docling converter (singleton) ─────────────────────────────────────
def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr             = False   # skip OCR — arXiv PDFs are born-digital
    pipeline_options.do_table_structure = True    # extract table structure
    pipeline_options.generate_picture_images = False

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# ── Parse single PDF ──────────────────────────────────────────────────
def parse_pdf(pdf_path: Path, meta: dict, converter: DocumentConverter, max_chunk_tokens: int) -> list[dict]:
    """
    Parse one PDF with Docling and return a list of chunk dicts.
    Each chunk carries the paper metadata as payload for Qdrant.
    """
    chunks = []

    try:
        result = converter.convert(str(pdf_path))
        doc    = result.document

        # ── Walk document items grouped by section ────────────────────
        current_section  = "Abstract"
        current_texts    = []
        chunk_index      = 0

        def flush_section(section_name: str, texts: list[str]):
            nonlocal chunk_index
            if not texts:
                return
            full_text = " ".join(texts)
            text_chunks = chunk_text(full_text, max_chunk_tokens, OVERLAP_SENTENCES)

            for i, chunk_text_str in enumerate(text_chunks):
                word_count = len(chunk_text_str.split())
                if word_count < MIN_CHUNK_TOKENS:
                    continue

                chunk = {
                    # ── Content ──────────────────────────────────────
                    "chunk_id":      f"{meta['arxiv_id']}__sec{chunk_index:03d}",
                    "text":          chunk_text_str,
                    "word_count":    word_count,
                    "chunk_index":   chunk_index,
                    "section":       section_name,
                    "section_chunk": i,   # chunk number within section

                    # ── Paper identity ────────────────────────────────
                    "arxiv_id":      meta["arxiv_id"],
                    "title":         meta["title"],
                    "authors":       meta["authors"][:5],   # cap at 5
                    "year":          meta.get("year"),
                    "abstract":      meta.get("abstract", "")[:500],

                    # ── PWC enrichment ────────────────────────────────
                    "conference":    meta.get("pwc", {}).get("conference"),
                    "has_code":      meta.get("pwc", {}).get("has_code", False),
                    "github_url":    meta.get("pwc", {}).get("github_url"),
                    "tasks":         meta.get("pwc", {}).get("tasks", []),

                    # ── Source ────────────────────────────────────────
                    "pdf_filename":  pdf_path.name,
                    "arxiv_url":     meta.get("arxiv_url"),
                }
                chunks.append(chunk)
                chunk_index += 1

        # Iterate over document structure
        for item, _ in doc.iterate_items():
            item_type = type(item).__name__

            # Section heading → flush previous section, start new one
            if item_type in ("SectionHeaderItem", "HeadingItem"):
                flush_section(current_section, current_texts)
                current_section = detect_section(item.text if hasattr(item, "text") else str(item))
                current_texts   = []

            # Text content → accumulate
            elif item_type in ("TextItem", "ParagraphItem"):
                text = item.text.strip() if hasattr(item, "text") else ""
                if text:
                    current_texts.append(text)

            # Table → convert to readable text row
            elif item_type == "TableItem":
                try:
                    table_md = item.export_to_markdown()
                    current_texts.append(f"[TABLE]\n{table_md}")
                except Exception:
                    pass

        # Flush final section
        flush_section(current_section, current_texts)

    except Exception as e:
        logger.warning(f"Parse failed for {pdf_path.name}: {e}")
        return []

    return chunks


# ── Main ─────────────────────────────────────────────────────────────
def main(max_chunk_tokens: int, workers: int):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # Load paper metadata
    meta_index = load_metadata_index()

    # Find PDFs to parse
    all_pdfs = sorted(PDF_DIR.glob("*.pdf"))
    if not all_pdfs:
        logger.error(f"No PDFs found in {PDF_DIR} — run collect.py first")
        return

    # Skip already-parsed PDFs
    parsed_ids = set()
    if CHUNKS_FILE.exists():
        with open(CHUNKS_FILE) as f:
            for line in f:
                try:
                    parsed_ids.add(json.loads(line)["pdf_filename"])
                except Exception:
                    pass
        logger.info(f"Resuming — {len(parsed_ids)} PDFs already parsed")

    to_parse = [p for p in all_pdfs if p.name not in parsed_ids]
    logger.info(f"PDFs to parse: {len(to_parse)} / {len(all_pdfs)} total")

    if not to_parse:
        logger.info("All PDFs already parsed — nothing to do")
        return

    # Build Docling converter (one per process is fine for sequential)
    logger.info("Initialising Docling converter...")
    converter = build_converter()

    # ── Parse loop ────────────────────────────────────────────────────
    all_chunks    = []
    success_count = 0
    fail_count    = 0
    section_stats : dict[str, int] = {}

    with open(CHUNKS_FILE, "a") as out_f:
        for pdf_path in tqdm(to_parse, desc="Parsing PDFs"):
            meta = meta_index.get(pdf_path.name, {
                "arxiv_id": pdf_path.stem,
                "title":    pdf_path.stem,
                "authors":  [],
                "year":     None,
                "abstract": "",
                "pwc":      {},
            })

            chunks = parse_pdf(pdf_path, meta, converter, max_chunk_tokens)

            if chunks:
                success_count += 1
                for chunk in chunks:
                    out_f.write(json.dumps(chunk) + "\n")
                    section_stats[chunk["section"]] = section_stats.get(chunk["section"], 0) + 1
                all_chunks.extend(chunks)
            else:
                fail_count += 1

    # ── Report ────────────────────────────────────────────────────────
    total_chunks = len(all_chunks)
    avg_words    = sum(c["word_count"] for c in all_chunks) / max(total_chunks, 1)

    report = {
        "timestamp":      __import__("datetime").datetime.now().isoformat(),
        "pdfs_parsed":    success_count,
        "pdfs_failed":    fail_count,
        "total_chunks":   total_chunks,
        "avg_words":      round(avg_words, 1),
        "section_counts": dict(sorted(section_stats.items(), key=lambda x: -x[1])),
    }

    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("")
    logger.info("═" * 55)
    logger.info("  Parsing Complete ✅")
    logger.info("═" * 55)
    logger.info(f"  PDFs parsed      : {success_count} / {success_count + fail_count}")
    logger.info(f"  Total chunks     : {total_chunks}")
    logger.info(f"  Avg words/chunk  : {avg_words:.0f}")
    logger.info(f"  Chunks file      : {CHUNKS_FILE}")
    logger.info(f"  Report           : {REPORT_FILE}")
    logger.info("")
    logger.info("  Top sections:")
    for section, count in list(report["section_counts"].items())[:8]:
        logger.info(f"    {section:<30} {count} chunks")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse PDFs with Docling into section-aware chunks")
    parser.add_argument("--max-chunk-tokens", type=int, default=512,  help="Max words per chunk (default: 512)")
    parser.add_argument("--workers",          type=int, default=1,    help="Parallel workers (default: 1, Docling is heavy)")
    args = parser.parse_args()

    main(max_chunk_tokens=args.max_chunk_tokens, workers=args.workers)