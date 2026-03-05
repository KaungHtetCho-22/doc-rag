"""
evaluation/run_ragas.py
─────────────────────────────────────────────────────────────────────
Stage 6 — RAG Quality Evaluation with Ragas

Evaluates the RAG pipeline across 4 metrics:
  - Faithfulness       : is the answer grounded in the retrieved context?
  - Answer Relevancy   : does the answer address the question?
  - Context Recall     : did retrieval find the relevant chunks?
  - Context Precision  : are retrieved chunks actually useful?

Runs against the live inference server (serve.py must be running).
Logs all results to MLflow for tracking across model versions.

Outputs:
    evaluation/results/ragas_scores.json   ← metric scores
    evaluation/results/ragas_detail.jsonl  ← per-question breakdown
    evaluation/results/ragas_report.md     ← human-readable report

Usage:
    # Server must be running first:
    python inference/serve.py &

    python evaluation/run_ragas.py
    python evaluation/run_ragas.py --mode local   # test local GGUF model
    python evaluation/run_ragas.py --n 20         # quick run with 20 questions
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime

import requests
import mlflow
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────
EVAL_DIR       = Path("evaluation/results")
SCORES_FILE    = EVAL_DIR / "ragas_scores.json"
DETAIL_FILE    = EVAL_DIR / "ragas_detail.jsonl"
REPORT_FILE    = EVAL_DIR / "ragas_report.md"
MLFLOW_DIR     = Path("mlruns")
MLFLOW_EXP     = "docrag-evaluation"
SERVER_URL     = "http://localhost:8000"

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
GROQ_JUDGE_MODEL = "llama-3.3-70b-versatile"   # stronger model for evaluation

# ── Evaluation questions ──────────────────────────────────────────────
# Diverse set covering different CV/ML topics and query types
EVAL_QUESTIONS = [
    # Conceptual
    "How do transformer architectures improve object detection compared to CNNs?",
    "What is the role of self-attention in visual feature extraction?",
    "How does contrastive learning work for visual representation learning?",
    "What are the main advantages of anchor-free object detection methods?",
    "How do vision transformers handle variable image sizes?",

    # Methodological
    "What loss functions are commonly used for training object detection models?",
    "How is non-maximum suppression used in object detection pipelines?",
    "What is the difference between instance segmentation and semantic segmentation?",
    "How do feature pyramid networks improve multi-scale object detection?",
    "What techniques are used to reduce computational cost in transformer models?",

    # Results/Benchmarks
    "What datasets are commonly used to evaluate object detection performance?",
    "How is mean average precision calculated for object detection evaluation?",
    "What are the main challenges in 3D point cloud object detection?",
    "How do self-supervised methods compare to supervised methods on detection tasks?",
    "What metrics are used to evaluate video object detection?",

    # Domain-specific
    "How are transformer decoders used in end-to-end object detection?",
    "What is deformable attention and why is it useful for detection?",
    "How does knowledge distillation improve small object detection models?",
    "What role does data augmentation play in training detection models?",
    "How are 2D and 3D detection approaches combined in autonomous driving?",
]


# ── Query the RAG server ──────────────────────────────────────────────
def query_server(question: str, mode: str) -> dict | None:
    try:
        r = requests.post(
            f"{SERVER_URL}/query",
            json    = {"query": question, "top_k": 5, "mode": mode},
            timeout = 60,
        )
        if r.status_code == 200:
            return r.json()
        logger.warning(f"Server returned {r.status_code} for: {question[:50]}")
        return None
    except Exception as e:
        logger.warning(f"Query failed: {e}")
        return None


# ── LLM-as-judge scoring ──────────────────────────────────────────────
def score_with_llm(question: str, answer: str, context: str) -> dict:
    """
    Use Groq (strong model) as judge to score faithfulness and relevancy.
    Returns scores between 0.0 and 1.0.
    """
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""Evaluate this RAG system response. Score each metric from 0.0 to 1.0.

Question: {question}

Retrieved Context:
{context[:2000]}

Generated Answer:
{answer[:1000]}

Score these metrics:
1. faithfulness: Is the answer fully supported by the context? (1.0 = fully grounded, 0.0 = hallucinated)
2. answer_relevancy: Does the answer actually address the question? (1.0 = perfectly relevant, 0.0 = off-topic)
3. context_precision: Are the retrieved chunks useful for answering? (1.0 = all chunks relevant, 0.0 = none relevant)

Return ONLY valid JSON, no explanation:
{{"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}}"""

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model       = GROQ_JUDGE_MODEL,
                messages    = [{"role": "user", "content": prompt}],
                temperature = 0.0,
                max_tokens  = 100,
            )
            raw = response.choices[0].message.content.strip()
            # Strip markdown fences
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            scores = json.loads(raw)
            # Clamp all values to [0, 1]
            return {k: max(0.0, min(1.0, float(v))) for k, v in scores.items()}
        except Exception as e:
            logger.debug(f"Judge attempt {attempt+1} failed: {e}")
            time.sleep(2)

    # Fallback if judge fails
    return {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}


def score_context_recall(question: str, context: str) -> float:
    """
    Simple heuristic for context recall:
    checks if key terms from the question appear in the retrieved context.
    """
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    prompt = f"""Does the retrieved context contain information needed to answer this question?

Question: {question}

Context:
{context[:2000]}

Reply with ONLY a JSON object: {{"context_recall": 0.0}}
Where 1.0 = context fully covers what's needed, 0.0 = context is missing key information."""

    try:
        response = client.chat.completions.create(
            model       = GROQ_JUDGE_MODEL,
            messages    = [{"role": "user", "content": prompt}],
            temperature = 0.0,
            max_tokens  = 50,
        )
        raw = response.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        result = json.loads(raw)
        return max(0.0, min(1.0, float(result.get("context_recall", 0.0))))
    except Exception:
        return 0.0


# ── Report generation ─────────────────────────────────────────────────
def generate_report(scores: dict, details: list, mode: str, n_questions: int) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def bar(score: float, width: int = 20) -> str:
        filled = int(score * width)
        return "█" * filled + "░" * (width - filled)

    lines = [
        "# DocRAG Evaluation Report",
        f"",
        f"**Date:** {timestamp}  ",
        f"**Mode:** {mode.upper()}  ",
        f"**Questions:** {n_questions}  ",
        f"",
        "## Ragas Scores",
        "",
        "| Metric | Score | Bar | Target |",
        "|--------|-------|-----|--------|",
        f"| Faithfulness     | {scores['faithfulness']:.3f} | {bar(scores['faithfulness'])} | > 0.80 |",
        f"| Answer Relevancy | {scores['answer_relevancy']:.3f} | {bar(scores['answer_relevancy'])} | > 0.75 |",
        f"| Context Recall   | {scores['context_recall']:.3f} | {bar(scores['context_recall'])} | > 0.70 |",
        f"| Context Precision| {scores['context_precision']:.3f} | {bar(scores['context_precision'])} | > 0.70 |",
        f"",
        "## Target Assessment",
        "",
    ]

    targets = {
        "Faithfulness":      (scores["faithfulness"],      0.80),
        "Answer Relevancy":  (scores["answer_relevancy"],  0.75),
        "Context Recall":    (scores["context_recall"],    0.70),
        "Context Precision": (scores["context_precision"], 0.70),
    }

    all_pass = True
    for metric, (score, target) in targets.items():
        status = "✅ PASS" if score >= target else "❌ FAIL"
        if score < target:
            all_pass = False
        lines.append(f"- {metric}: {score:.3f} {status} (target: {target})")

    lines += [
        "",
        f"**Overall: {'✅ All targets met' if all_pass else '⚠️ Some targets missed'}**",
        "",
        "## Per-Question Breakdown",
        "",
        "| # | Question | Faith | Relev | Recall | Prec |",
        "|---|----------|-------|-------|--------|------|",
    ]

    for i, d in enumerate(details, 1):
        q_short = d["question"][:45] + "..." if len(d["question"]) > 45 else d["question"]
        lines.append(
            f"| {i} | {q_short} "
            f"| {d['faithfulness']:.2f} "
            f"| {d['answer_relevancy']:.2f} "
            f"| {d['context_recall']:.2f} "
            f"| {d['context_precision']:.2f} |"
        )

    lines += ["", "## Latency", ""]
    latencies = [d["latency_ms"] for d in details if d.get("latency_ms")]
    if latencies:
        lines.append(f"- Avg: {sum(latencies)/len(latencies):.0f}ms")
        lines.append(f"- P95: {sorted(latencies)[int(0.95*len(latencies))]:.0f}ms")
        lines.append(f"- Min: {min(latencies):.0f}ms")
        lines.append(f"- Max: {max(latencies):.0f}ms")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────
def main(mode: str, n: int):
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # Check server
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        logger.info(f"Server connected — mode: {r.json()['mode']}")
    except Exception:
        logger.error("Inference server not reachable. Run: python inference/serve.py")
        raise SystemExit(1)

    if not GROQ_API_KEY or GROQ_API_KEY == "your_groq_api_key_here":
        logger.error("GROQ_API_KEY required for LLM-as-judge evaluation")
        raise SystemExit(1)

    questions = EVAL_QUESTIONS[:n]
    logger.info(f"Evaluating {len(questions)} questions | mode: {mode}")

    # MLflow tracking
    mlflow.set_tracking_uri(f"file://{Path.cwd() / MLFLOW_DIR}")
    mlflow.set_experiment("docrag-evaluation")

    with mlflow.start_run(run_name=f"ragas-eval-{mode}-{datetime.now().strftime('%m%d-%H%M')}"):
        mlflow.log_params({"mode": mode, "n_questions": len(questions), "judge_model": GROQ_JUDGE_MODEL})

        details    = []
        all_faith  = []
        all_relev  = []
        all_recall = []
        all_prec   = []

        for i, question in enumerate(questions, 1):
            logger.info(f"[{i}/{len(questions)}] {question[:60]}")

            # Query RAG
            result = query_server(question, mode)
            if not result:
                logger.warning(f"Skipping question {i} — server error")
                continue

            answer  = result["answer"]
            sources = result["sources"]
            context = "\n\n".join(
                f"[{s['title'][:50]}]\n{s['text']}" for s in sources
            )

            # Score
            scores        = score_with_llm(question, answer, context)
            recall        = score_context_recall(question, context)
            time.sleep(1.5)  # respect Groq rate limit

            faith = scores.get("faithfulness", 0.0)
            relev = scores.get("answer_relevancy", 0.0)
            prec  = scores.get("context_precision", 0.0)

            all_faith.append(faith)
            all_relev.append(relev)
            all_recall.append(recall)
            all_prec.append(prec)

            detail = {
                "question":          question,
                "answer":            answer[:500],
                "faithfulness":      faith,
                "answer_relevancy":  relev,
                "context_recall":    recall,
                "context_precision": prec,
                "latency_ms":        result.get("latency_ms"),
                "mode":              mode,
            }
            details.append(detail)

            # Log per-step to MLflow
            mlflow.log_metrics({
                "faithfulness":      faith,
                "answer_relevancy":  relev,
                "context_recall":    recall,
                "context_precision": prec,
            }, step=i)

            logger.info(
                f"  faith={faith:.2f} relev={relev:.2f} "
                f"recall={recall:.2f} prec={prec:.2f}"
            )

        if not details:
            logger.error("No questions evaluated successfully")
            return

        # Aggregate scores
        def avg(lst): return sum(lst) / len(lst) if lst else 0.0

        final_scores = {
            "faithfulness":      avg(all_faith),
            "answer_relevancy":  avg(all_relev),
            "context_recall":    avg(all_recall),
            "context_precision": avg(all_prec),
            "mode":              mode,
            "n_questions":       len(details),
            "timestamp":         datetime.now().isoformat(),
        }

        # Log final aggregates to MLflow
        mlflow.log_metrics({
            "avg_faithfulness":      final_scores["faithfulness"],
            "avg_answer_relevancy":  final_scores["answer_relevancy"],
            "avg_context_recall":    final_scores["context_recall"],
            "avg_context_precision": final_scores["context_precision"],
        })

        # Save outputs
        with open(SCORES_FILE, "w") as f:
            json.dump(final_scores, f, indent=2)

        with open(DETAIL_FILE, "w") as f:
            for d in details:
                f.write(json.dumps(d) + "\n")

        report = generate_report(final_scores, details, mode, len(details))
        with open(REPORT_FILE, "w") as f:
            f.write(report)

        mlflow.log_artifact(str(SCORES_FILE))
        mlflow.log_artifact(str(REPORT_FILE))

        # Print summary
        logger.info("")
        logger.info("═" * 55)
        logger.info("  Ragas Evaluation Complete ✅")
        logger.info("═" * 55)
        logger.info(f"  Faithfulness      : {final_scores['faithfulness']:.3f}  (target > 0.80)")
        logger.info(f"  Answer Relevancy  : {final_scores['answer_relevancy']:.3f}  (target > 0.75)")
        logger.info(f"  Context Recall    : {final_scores['context_recall']:.3f}  (target > 0.70)")
        logger.info(f"  Context Precision : {final_scores['context_precision']:.3f}  (target > 0.70)")
        logger.info(f"  Report            : {REPORT_FILE}")
        logger.info(f"  MLflow UI         : mlflow ui --port 5000")
        logger.info("═" * 55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline with Ragas")
    parser.add_argument("--mode", default="groq", choices=["groq", "local"],
                        help="Inference mode to evaluate (default: groq)")
    parser.add_argument("--n",    type=int, default=20,
                        help="Number of questions to evaluate (default: 20, max: 20)")
    args = parser.parse_args()

    main(mode=args.mode, n=min(args.n, len(EVAL_QUESTIONS)))