"""
training/generate_data.py
─────────────────────────────────────────────────────────────────────
Stage 3 — Synthetic Training Data Generation

Reads chunks from Qdrant/chunks.jsonl, sends each to Groq
(llama-3.1-70b) to generate QA pairs, cleans and deduplicates,
then saves as ShareGPT-format JSONL for Unsloth fine-tuning.

Outputs:
    data/training/qa_pairs.jsonl     ← all raw QA pairs
    data/training/train.jsonl        ← 80% split (ShareGPT format)
    data/training/eval.jsonl         ← 20% split (ShareGPT format)
    data/training/gen_report.json    ← generation stats

Usage:
    python training/generate_data.py
    python training/generate_data.py --max-chunks 100   # quick test
    python training/generate_data.py --resume           # skip done chunks
"""

import os
import json
import time
import random
import hashlib
import argparse
from pathlib import Path

from groq import Groq
from tqdm import tqdm
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
GROQ_MODEL        = os.getenv("GROQ_DATAGEN_MODEL", "llama-3.1-70b-versatile")
QA_PER_CHUNK      = 2
GROQ_DELAY        = 1.0     # seconds between API calls (free tier: ~30 req/min)
MAX_RETRIES       = 3
RETRY_DELAY       = 10.0    # seconds to wait on rate limit

PROCESSED_DIR     = Path(os.getenv("DATA_PROCESSED_DIR", "data/processed"))
TRAINING_DIR      = Path(os.getenv("DATA_TRAINING_DIR", "data/training"))
CHUNKS_FILE       = PROCESSED_DIR / "chunks.jsonl"
QA_FILE           = TRAINING_DIR / "qa_pairs.jsonl"
TRAIN_FILE        = TRAINING_DIR / "train.jsonl"
EVAL_FILE         = TRAINING_DIR / "eval.jsonl"
REPORT_FILE       = TRAINING_DIR / "gen_report.json"
DONE_IDS_FILE     = TRAINING_DIR / ".done_chunk_ids"

TRAIN_SPLIT       = 0.8
MIN_ANSWER_WORDS  = 15      # discard trivially short answers
MIN_QUESTION_WORDS = 5


# ── Prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are an expert in computer vision and machine learning research.
Your task is to generate high-quality question-answer pairs from research paper excerpts.
These pairs will be used to train an AI assistant that answers questions about CV/ML papers.

Rules:
- Questions must be specific and answerable from the provided text
- Answers must be grounded in the text — no hallucination
- Questions should vary in type: factual, conceptual, methodological
- Do NOT generate questions about authors, citations, or paper metadata
- Return ONLY valid JSON, no markdown, no preamble"""

USER_PROMPT_TEMPLATE = """Generate exactly {n} question-answer pairs from this research paper excerpt.

Paper: {title} ({year})
Section: {section}

Text:
{text}

Return a JSON array with exactly {n} objects, each with "question" and "answer" keys.
Example format:
[
  {{"question": "What method is proposed for X?", "answer": "The paper proposes..."}},
  {{"question": "How does Y improve over Z?", "answer": "Y improves over Z by..."}}
]"""


# ── Groq generation ───────────────────────────────────────────────────
def generate_qa_pairs(client: Groq, chunk: dict, n: int) -> list[dict]:
    """Call Groq to generate n QA pairs from a chunk. Returns list of dicts."""

    prompt = USER_PROMPT_TEMPLATE.format(
        n       = n,
        title   = chunk.get("title", "Unknown"),
        year    = chunk.get("year", "Unknown"),
        section = chunk.get("section", "Unknown"),
        text    = chunk["text"][:1500],   # cap to avoid token overflow
    )

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model       = GROQ_MODEL,
                messages    = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature = 0.7,
                max_tokens  = 1024,
            )

            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if model adds them
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()

            pairs = json.loads(raw)

            # Validate structure
            if not isinstance(pairs, list):
                return []

            valid = []
            for p in pairs:
                if not isinstance(p, dict):
                    continue
                q = str(p.get("question", "")).strip()
                a = str(p.get("answer", "")).strip()
                if (len(q.split()) >= MIN_QUESTION_WORDS and
                    len(a.split()) >= MIN_ANSWER_WORDS):
                    valid.append({"question": q, "answer": a})

            return valid[:n]

        except json.JSONDecodeError as e:
            logger.debug(f"JSON parse failed (attempt {attempt+1}): {e}")
            time.sleep(2)

        except Exception as e:
            err = str(e).lower()
            if "rate limit" in err or "429" in err:
                logger.warning(f"Rate limited — waiting {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)
            else:
                logger.warning(f"Groq error (attempt {attempt+1}): {e}")
                time.sleep(2)

    return []


# ── ShareGPT formatter ────────────────────────────────────────────────
def to_sharegpt(qa: dict, chunk: dict) -> dict:
    """
    Convert a QA pair to ShareGPT format for Unsloth SFTTrainer.

    ShareGPT format:
    {
      "conversations": [
        {"from": "human",     "value": "<question>"},
        {"from": "assistant", "value": "<answer>"}
      ]
    }
    """
    # Add paper context to the question
    context = (
        f"Based on the paper '{chunk.get('title', 'Unknown')}' "
        f"({chunk.get('year', 'Unknown')}), {chunk.get('section', '')} section:\n\n"
        f"{qa['question']}"
    )
    return {
        "conversations": [
            {"from": "human",     "value": context},
            {"from": "assistant", "value": qa["answer"]},
        ],
        # Metadata for analysis (not used in training)
        "metadata": {
            "arxiv_id":  chunk.get("arxiv_id"),
            "section":   chunk.get("section"),
            "chunk_id":  chunk.get("chunk_id"),
            "year":      chunk.get("year"),
        }
    }


# ── Deduplication ─────────────────────────────────────────────────────
def dedup_pairs(pairs: list[dict]) -> list[dict]:
    """Remove near-duplicate questions using question text hashing."""
    seen   = set()
    unique = []
    for p in pairs:
        # Normalise question for dedup comparison
        key = hashlib.md5(
            p["conversations"][0]["value"].lower()
            .strip()
            .replace("?", "")
            [:100]
            .encode()
        ).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return unique


# ── Train / eval split ────────────────────────────────────────────────
def split_and_save(pairs: list[dict]):
    random.shuffle(pairs)
    split_idx = int(len(pairs) * TRAIN_SPLIT)
    train     = pairs[:split_idx]
    eval_     = pairs[split_idx:]

    with open(TRAIN_FILE, "w") as f:
        for p in train:
            f.write(json.dumps(p) + "\n")

    with open(EVAL_FILE, "w") as f:
        for p in eval_:
            f.write(json.dumps(p) + "\n")

    logger.info(f"Train: {len(train)} pairs → {TRAIN_FILE}")
    logger.info(f"Eval:  {len(eval_)} pairs → {EVAL_FILE}")
    return len(train), len(eval_)


# ── Main ─────────────────────────────────────────────────────────────
def main(max_chunks: int | None, resume: bool):
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        logger.error("GROQ_API_KEY not set in .env")
        raise SystemExit(1)

    client = Groq(api_key=GROQ_API_KEY)
    logger.info(f"Groq client ready | model: {GROQ_MODEL}")

    # Load chunks
    if not CHUNKS_FILE.exists():
        logger.error(f"Chunks file not found: {CHUNKS_FILE}")
        raise SystemExit(1)

    all_chunks = []
    with open(CHUNKS_FILE) as f:
        for line in f:
            all_chunks.append(json.loads(line))

    logger.info(f"Loaded {len(all_chunks)} chunks")

    # Skip already-processed chunks if resuming
    done_ids = set()
    if resume and DONE_IDS_FILE.exists():
        with open(DONE_IDS_FILE) as f:
            done_ids = set(line.strip() for line in f if line.strip())
        logger.info(f"Resuming — {len(done_ids)} chunks already processed")

    chunks_to_process = [
        c for c in all_chunks
        if c["chunk_id"] not in done_ids
        # Skip very short chunks and reference sections (low quality QA)
        and c.get("word_count", 0) >= 50
        and "reference" not in c.get("section", "").lower()
        and "acknowledgement" not in c.get("section", "").lower()
    ]

    if max_chunks:
        chunks_to_process = chunks_to_process[:max_chunks]

    logger.info(f"Chunks to process: {len(chunks_to_process)}")
    logger.info(f"Target QA pairs  : ~{len(chunks_to_process) * QA_PER_CHUNK}")
    logger.info(f"Est. time        : ~{len(chunks_to_process) * GROQ_DELAY / 60:.0f} min")

    # ── Generation loop ───────────────────────────────────────────────
    all_sharegpt  = []
    success_count = 0
    fail_count    = 0
    total_raw     = 0

    with open(QA_FILE, "a") as qa_f, open(DONE_IDS_FILE, "a") as done_f:
        for chunk in tqdm(chunks_to_process, desc="Generating QA pairs"):
            pairs = generate_qa_pairs(client, chunk, QA_PER_CHUNK)

            if pairs:
                success_count += 1
                total_raw     += len(pairs)
                for pair in pairs:
                    sharegpt = to_sharegpt(pair, chunk)
                    qa_f.write(json.dumps(sharegpt) + "\n")
                    all_sharegpt.append(sharegpt)
            else:
                fail_count += 1

            done_f.write(chunk["chunk_id"] + "\n")
            time.sleep(GROQ_DELAY)

    # ── Dedup + split ─────────────────────────────────────────────────
    logger.info(f"Raw pairs generated: {total_raw}")

    # Load all pairs (including from previous runs if resuming)
    all_pairs = []
    with open(QA_FILE) as f:
        for line in f:
            try:
                all_pairs.append(json.loads(line))
            except Exception:
                continue

    before_dedup = len(all_pairs)
    all_pairs    = dedup_pairs(all_pairs)
    after_dedup  = len(all_pairs)

    logger.info(f"After dedup: {after_dedup} pairs ({before_dedup - after_dedup} removed)")

    n_train, n_eval = split_and_save(all_pairs)

    # ── Report ────────────────────────────────────────────────────────
    report = {
        "model":           GROQ_MODEL,
        "chunks_processed": success_count + fail_count,
        "chunks_failed":   fail_count,
        "raw_pairs":       total_raw,
        "after_dedup":     after_dedup,
        "train_pairs":     n_train,
        "eval_pairs":      n_eval,
        "qa_per_chunk":    QA_PER_CHUNK,
    }
    with open(REPORT_FILE, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("")
    logger.info("═" * 55)
    logger.info("  Data Generation Complete ✅")
    logger.info("═" * 55)
    logger.info(f"  Chunks processed : {success_count + fail_count}")
    logger.info(f"  Raw QA pairs     : {total_raw}")
    logger.info(f"  After dedup      : {after_dedup}")
    logger.info(f"  Train split      : {n_train}")
    logger.info(f"  Eval split       : {n_eval}")
    logger.info(f"  Report           : {REPORT_FILE}")
    logger.info("")
    logger.info("Next: python training/finetune.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic QA pairs via Groq")
    parser.add_argument("--max-chunks", type=int, default=None,
                        help="Limit chunks to process (default: all)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed chunks")
    args = parser.parse_args()

    main(max_chunks=args.max_chunks, resume=args.resume)