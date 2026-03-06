"""
training/finetune.py
─────────────────────────────────────────────────────────────────────
Stage 4 — Fine-tuning with Unsloth + QLoRA

Fine-tunes Llama 3.2 3B Instruct on synthetic CV/ML QA pairs using
QLoRA (4-bit) via Unsloth. Fits in 6GB VRAM (RTX 3060).

Outputs:
    models/adapter/          ← LoRA adapter weights
    models/merged/           ← merged model (full weights)
    models/docrag-3b-q4.gguf ← quantized GGUF for llama.cpp

Tracking:
    MLflow local server — run: mlflow ui

Usage:
    python training/finetune.py
    python training/finetune.py --dry-run     # 10 steps to verify setup
    python training/finetune.py --no-export   # skip GGUF export
"""

import os
import argparse
from pathlib import Path
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
BASE_MODEL     = os.getenv("BASE_MODEL", "unsloth/Llama-3.2-1B-Instruct")
TRAINING_DIR   = Path(os.getenv("DATA_TRAINING_DIR", "data/training"))
MODELS_DIR     = Path("models")
ADAPTER_DIR    = MODELS_DIR / "adapter"
MERGED_DIR     = MODELS_DIR / "merged"
GGUF_PATH      = MODELS_DIR / "docrag-3b-q4.gguf"
TRAIN_FILE     = TRAINING_DIR / "train.jsonl"
EVAL_FILE      = TRAINING_DIR / "eval.jsonl"
MLFLOW_DIR     = Path("mlruns")
MLFLOW_EXP     = "docrag-finetune"

# ── Training hyperparameters ──────────────────────────────────────────
# Tuned for RTX 3060 6GB VRAM
MAX_SEQ_LENGTH = 1024
LORA_RANK      = 16
LORA_ALPHA     = 32
LORA_DROPOUT   = 0.05
BATCH_SIZE     = 1
GRAD_ACCUM     = 8      # effective batch = 1 * 8 = 8
EPOCHS         = 3
LR             = 2e-4
WARMUP_RATIO   = 0.05
LR_SCHEDULER   = "cosine"
WEIGHT_DECAY   = 0.01
SAVE_STEPS     = 50
EVAL_STEPS     = 50
LOGGING_STEPS  = 10


def load_dataset():
    """Load train/eval JSONL files into HuggingFace datasets."""
    from datasets import load_dataset as hf_load_dataset

    if not TRAIN_FILE.exists():
        logger.error(f"Train file not found: {TRAIN_FILE}")
        logger.error("Run training/generate_data.py first.")
        raise SystemExit(1)

    train_ds = hf_load_dataset("json", data_files=str(TRAIN_FILE), split="train")
    eval_ds  = hf_load_dataset("json", data_files=str(EVAL_FILE),  split="train") \
               if EVAL_FILE.exists() else None

    logger.info(f"Train samples : {len(train_ds)}")
    if eval_ds:
        logger.info(f"Eval samples  : {len(eval_ds)}")

    return train_ds, eval_ds


def format_sharegpt(example: dict, tokenizer) -> dict:
    """
    Convert ShareGPT conversation format to a single training string
    using the model's chat template.
    """
    role_map = {"human": "user", "gpt": "assistant", "assistant": "assistant"}
    messages = [
        {"role": role_map.get(turn["from"], turn["from"]), "content": turn["value"]}
        for turn in example["conversations"]
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main(dry_run: bool, no_export: bool):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Reduce CUDA memory fragmentation — critical for 6GB VRAM
    import os as _os
    _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import mlflow
    import torch

    # ── MLflow setup ──────────────────────────────────────────────────
    mlflow.set_tracking_uri(f"file://{Path.cwd() / MLFLOW_DIR}")
    mlflow.set_experiment(MLFLOW_EXP)

    run_name = f"docrag-llama3.2-3b-lora-r{LORA_RANK}" + ("-dryrun" if dry_run else "")

    with mlflow.start_run(run_name=run_name) as run:
        logger.info(f"MLflow run: {run.info.run_id}")
        logger.info(f"MLflow UI:  mlflow ui --port 5000")

        # Log hyperparameters
        mlflow.log_params({
            "base_model":  BASE_MODEL,
            "lora_rank":   LORA_RANK,
            "lora_alpha":  LORA_ALPHA,
            "batch_size":  BATCH_SIZE,
            "grad_accum":  GRAD_ACCUM,
            "epochs":      EPOCHS,
            "lr":          LR,
            "max_seq_len": MAX_SEQ_LENGTH,
            "dry_run":     dry_run,
        })

        # ── 1. Load model with Unsloth ────────────────────────────────
        logger.info("═" * 55)
        logger.info("  Step 1/5 — Loading model with Unsloth")
        logger.info("═" * 55)

        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name     = BASE_MODEL,
            max_seq_length = MAX_SEQ_LENGTH,
            dtype          = None,        # auto: bf16 on Ampere+, fp16 on older
            load_in_4bit   = True,        # QLoRA — critical for 6GB VRAM
        )

        vram_loaded = torch.cuda.memory_allocated() / 1e9
        logger.info(f"Base model loaded: {BASE_MODEL}")
        logger.info(f"VRAM used: {vram_loaded:.2f} GB")
        mlflow.log_metric("vram_after_load_gb", vram_loaded)

        # ── 2. Attach LoRA adapters ───────────────────────────────────
        logger.info("")
        logger.info("═" * 55)
        logger.info("  Step 2/5 — Attaching LoRA adapters")
        logger.info("═" * 55)

        model = FastLanguageModel.get_peft_model(
            model,
            r              = LORA_RANK,
            target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_alpha     = LORA_ALPHA,
            lora_dropout   = LORA_DROPOUT,
            bias           = "none",
            use_gradient_checkpointing = "unsloth",  # saves ~30% VRAM
            random_state   = 42,
            use_rslora     = False,
        )

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        pct       = 100 * trainable / total
        logger.info(f"Trainable params : {trainable:,} ({pct:.2f}%)")
        logger.info(f"Total params     : {total:,}")
        mlflow.log_params({"trainable_params": trainable, "trainable_pct": round(pct, 2)})

        # ── 3. Prepare dataset ────────────────────────────────────────
        logger.info("")
        logger.info("═" * 55)
        logger.info("  Step 3/5 — Preparing dataset")
        logger.info("═" * 55)

        train_ds, eval_ds = load_dataset()

        train_ds = train_ds.map(
            lambda x: format_sharegpt(x, tokenizer),
            remove_columns=train_ds.column_names,
        )
        if eval_ds:
            eval_ds = eval_ds.map(
                lambda x: format_sharegpt(x, tokenizer),
                remove_columns=eval_ds.column_names,
            )

        mlflow.log_params({
            "train_samples": len(train_ds),
            "eval_samples":  len(eval_ds) if eval_ds else 0,
        })
        logger.info(f"Dataset formatted. Sample:\n{train_ds[0]['text'][:300]}...")

        # ── 4. Train ──────────────────────────────────────────────────
        logger.info("")
        logger.info("═" * 55)
        logger.info("  Step 4/5 — Training")
        logger.info("═" * 55)

        from trl import SFTTrainer, SFTConfig

        training_args = SFTConfig(
            output_dir              = str(ADAPTER_DIR),
            num_train_epochs        = 1 if dry_run else EPOCHS,
            max_steps               = 10 if dry_run else -1,
            per_device_train_batch_size  = BATCH_SIZE,
            per_device_eval_batch_size   = BATCH_SIZE,
            gradient_accumulation_steps  = GRAD_ACCUM,
            learning_rate           = LR,
            lr_scheduler_type       = LR_SCHEDULER,
            warmup_ratio            = WARMUP_RATIO,
            weight_decay            = WEIGHT_DECAY,
            fp16                    = not torch.cuda.is_bf16_supported(),
            bf16                    = torch.cuda.is_bf16_supported(),
            optim                   = "adamw_8bit",
            logging_steps           = LOGGING_STEPS,
            save_steps              = SAVE_STEPS,
            eval_steps              = EVAL_STEPS if eval_ds else None,
            eval_strategy           = "steps" if eval_ds else "no",
            save_strategy           = "steps",
            load_best_model_at_end  = True if eval_ds else False,
            metric_for_best_model   = "eval_loss" if eval_ds else None,
            report_to               = "none",   # we handle logging manually
            max_seq_length          = MAX_SEQ_LENGTH,
            dataset_text_field      = "text",
            packing                 = False,
            seed                    = 42,
        )

        trainer = SFTTrainer(
            model         = model,
            tokenizer     = tokenizer,
            train_dataset = train_ds,
            eval_dataset  = eval_ds,
            args          = training_args,
        )

        # Hook: log each loss step to MLflow
        from transformers import TrainerCallback

        class MLflowCallback(TrainerCallback):
            def on_log(self, args, state, control, logs=None, **kwargs):
                if logs:
                    step = state.global_step
                    for k, v in logs.items():
                        if isinstance(v, (int, float)):
                            mlflow.log_metric(k, v, step=step)

        trainer.add_callback(MLflowCallback())

        vram_before = torch.cuda.memory_allocated() / 1e9
        logger.info(f"VRAM before training: {vram_before:.2f} GB")
        mlflow.log_metric("vram_before_train_gb", vram_before)

        # Resume from latest checkpoint if available, else start fresh
        import glob
        checkpoints = sorted(glob.glob(str(ADAPTER_DIR / "checkpoint-*")))
        resume = checkpoints[-1] if checkpoints else None
        if resume:
            logger.info(f"Resuming from checkpoint: {resume}")
        trainer_stats = trainer.train(resume_from_checkpoint=resume)

        train_loss = trainer_stats.training_loss
        runtime    = trainer_stats.metrics.get("train_runtime", 0)

        mlflow.log_metrics({
            "final_train_loss": train_loss,
            "runtime_min":      runtime / 60,
        })

        logger.info(f"Training loss : {train_loss:.4f}")
        logger.info(f"Runtime       : {runtime / 60:.1f} min")

        # Save adapter
        ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(ADAPTER_DIR))
        tokenizer.save_pretrained(str(ADAPTER_DIR))
        logger.info(f"LoRA adapter saved → {ADAPTER_DIR}")

        # Log adapter as MLflow artifact
        mlflow.log_artifacts(str(ADAPTER_DIR), artifact_path="adapter")

        # ── 5. Export GGUF ────────────────────────────────────────────
        if not no_export and not dry_run:
            logger.info("")
            logger.info("═" * 55)
            logger.info("  Step 5/5 — Exporting to GGUF")
            logger.info("═" * 55)

            MERGED_DIR.mkdir(parents=True, exist_ok=True)

            logger.info("Merging LoRA into base model...")
            model.save_pretrained_merged(
                str(MERGED_DIR),
                tokenizer,
                save_method = "merged_16bit",
            )
            logger.info(f"Merged model saved → {MERGED_DIR}")

            logger.info("Exporting to GGUF Q4_K_M...")
            model.save_pretrained_gguf(
                str(MODELS_DIR / "docrag-3b"),
                tokenizer,
                quantization_method = "q4_k_m",
            )

            gguf_files = list(MODELS_DIR.glob("docrag-3b*.gguf"))
            if gguf_files:
                gguf_files[0].rename(GGUF_PATH)
                logger.info(f"GGUF saved → {GGUF_PATH}")
                mlflow.log_artifact(str(GGUF_PATH), artifact_path="gguf")
        else:
            logger.info("Skipping GGUF export (--no-export or --dry-run)")

        logger.info("")
        logger.info("═" * 55)
        logger.info("  Fine-tuning Complete ✅")
        logger.info("═" * 55)
        logger.info(f"  Adapter   : {ADAPTER_DIR}")
        if not no_export and not dry_run:
            logger.info(f"  GGUF      : {GGUF_PATH}")
        logger.info(f"  MLflow ID : {run.info.run_id}")
        logger.info(f"  View UI   : mlflow ui --port 5000")
        logger.info("")
        logger.info("Next: python inference/serve.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Llama 3.2 3B with Unsloth QLoRA")
    parser.add_argument("--dry-run",   action="store_true",
                        help="Run only 10 steps to verify setup")
    parser.add_argument("--no-export", action="store_true",
                        help="Skip GGUF export after training")
    args = parser.parse_args()

    main(dry_run=args.dry_run, no_export=args.no_export)