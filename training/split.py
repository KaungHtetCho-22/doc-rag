"""
training/split_data.py
─────────────────────────────────────────────────────────────────────
Splits qa_pairs.jsonl into train.jsonl and eval.jsonl.

Can be run standalone at any time — useful when:
  - generate_data.py was interrupted and you want to split what you have
  - You want to re-split with a different ratio
  - You added more QA pairs and want to re-split

Outputs:
    data/training/train.jsonl   ← 80% (default)
    data/training/eval.jsonl    ← 20% (default)

Usage:
    python training/split_data.py
    python training/split_data.py --split 0.9   # 90/10 split
    python training/split_data.py --seed 123    # different shuffle
"""

import os
import json
import random
import argparse
from pathlib import Path

from loguru import logger
from dotenv import load_dotenv

load_dotenv()

TRAINING_DIR = Path(os.getenv("DATA_TRAINING_DIR", "data/training"))
QA_FILE      = TRAINING_DIR / "qa_pairs.jsonl"
TRAIN_FILE   = TRAINING_DIR / "train.jsonl"
EVAL_FILE    = TRAINING_DIR / "eval.jsonl"


def main(split: float, seed: int):
    if not QA_FILE.exists():
        logger.error(f"QA pairs file not found: {QA_FILE}")
        logger.error("Run training/generate_data.py first.")
        raise SystemExit(1)

    # Load all pairs
    pairs = []
    skipped = 0
    with open(QA_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                skipped += 1

    if skipped:
        logger.warning(f"Skipped {skipped} malformed lines")

    if not pairs:
        logger.error("No valid QA pairs found in file.")
        raise SystemExit(1)

    logger.info(f"Loaded {len(pairs)} QA pairs from {QA_FILE}")

    # Shuffle
    random.seed(seed)
    random.shuffle(pairs)

    # Split
    split_idx = int(len(pairs) * split)
    train     = pairs[:split_idx]
    eval_     = pairs[split_idx:]

    # Write
    TRAINING_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_FILE, "w") as f:
        for p in train:
            f.write(json.dumps(p) + "\n")

    with open(EVAL_FILE, "w") as f:
        for p in eval_:
            f.write(json.dumps(p) + "\n")

    logger.info("")
    logger.info("═" * 45)
    logger.info("  Split Complete ✅")
    logger.info("═" * 45)
    logger.info(f"  Total pairs : {len(pairs)}")
    logger.info(f"  Train       : {len(train)}  ({split*100:.0f}%)  → {TRAIN_FILE}")
    logger.info(f"  Eval        : {len(eval_)}   ({(1-split)*100:.0f}%)  → {EVAL_FILE}")
    logger.info("")
    logger.info("Next: python training/finetune.py --dry-run")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split QA pairs into train/eval")
    parser.add_argument("--split", type=float, default=0.8,
                        help="Train fraction (default: 0.8)")
    parser.add_argument("--seed",  type=int,   default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    assert 0.5 < args.split < 1.0, "--split must be between 0.5 and 1.0"
    main(split=args.split, seed=args.seed)