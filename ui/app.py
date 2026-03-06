"""
ui/app.py
─────────────────────────────────────────────────────────────────────
Stage 7 — Gradio Demo Interface

A polished research assistant UI that showcases all 6 technologies:
  - Docling  : live PDF upload + parsing
  - Qdrant   : hybrid search with metadata filters
  - Groq     : cloud inference mode
  - llama.cpp: local inference mode
  - Unsloth  : fine-tuned model (shown in local mode)
  - RAG      : sources + citations displayed per answer

Usage:
    # inference server must be running first:
    python inference/serve.py

    python ui/app.py
    python ui/app.py --port 7860 --share   # public gradio link
"""

import os
import json
import requests
import tempfile
import argparse
from pathlib import Path
from dotenv import load_dotenv

import gradio as gr

load_dotenv()

SERVER_URL = "http://localhost:8000"


# ── Server helpers ────────────────────────────────────────────────────
def get_server_health() -> dict:
    try:
        return requests.get(f"{SERVER_URL}/health", timeout=3).json()
    except Exception:
        return {}


def get_server_stats() -> dict:
    try:
        return requests.get(f"{SERVER_URL}/stats", timeout=3).json()
    except Exception:
        return {}


def switch_mode(mode: str) -> str:
    try:
        r = requests.post(f"{SERVER_URL}/mode/{mode}", timeout=5)
        if r.status_code == 200:
            return f"✅ Switched to **{mode.upper()}** mode"
        return f"⚠️ {r.json().get('detail', 'Switch failed')}"
    except Exception as e:
        return f"❌ Server error: {e}"


# ── Query handler ─────────────────────────────────────────────────────
def run_query(
    question: str,
    mode: str,
    year_filter: str,
    has_code: bool,
    section_filter: str,
    top_k: int,
) -> tuple[str, str, str]:
    """
    Returns: (answer_md, sources_md, stats_md)
    """
    if not question.strip():
        return "Please enter a question.", "", ""

    payload = {
        "query":   question.strip(),
        "top_k":   top_k,
        "mode":    mode.lower(),
    }

    if year_filter and year_filter != "Any":
        try:
            payload["year"] = int(year_filter)
        except ValueError:
            pass

    if has_code:
        payload["has_code"] = True

    if section_filter and section_filter != "Any":
        payload["section"] = section_filter

    try:
        r = requests.post(f"{SERVER_URL}/query", json=payload, timeout=60)
    except Exception as e:
        return f"❌ Could not reach server: {e}\n\nMake sure `python inference/serve.py` is running.", "", ""

    if r.status_code != 200:
        detail = r.json().get("detail", "Unknown error")
        return f"⚠️ {detail}", "", ""

    data = r.json()

    # ── Format answer ─────────────────────────────────────────────────
    answer_md = f"{data['answer']}\n"

    # ── Format sources ────────────────────────────────────────────────
    sources_md = ""
    for i, src in enumerate(data["sources"], 1):
        conf     = src.get("conference") or "—"
        year     = src.get("year") or "—"
        code     = "✅ Code available" if src.get("has_code") else "No code"
        section  = src.get("section", "")[:50]
        title    = src.get("title", "")[:70]
        text     = src.get("text", "")[:250]
        score    = src.get("score", 0)
        arxiv_id = src.get("arxiv_id", "")
        url      = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "#"

        sources_md += f"""**[{i}] [{title}]({url})**
`{year}` · `{conf}` · `{section}` · {code} · score: `{score:.4f}`

> {text}...

---
"""

    # ── Format stats ──────────────────────────────────────────────────
    stats_md = (
        f"**Mode:** `{data['mode'].upper()}`  \n"
        f"**Total:** `{data['latency_ms']}ms`  \n"
        f"**Retrieval:** `{data['retrieval_ms']}ms`  \n"
        f"**Generation:** `{data['generation_ms']}ms`  \n"
        f"**Sources:** `{len(data['sources'])}`"
    )

    return answer_md, sources_md, stats_md


# ── PDF upload + live Docling parse ───────────────────────────────────
def parse_uploaded_pdf(pdf_file) -> str:
    """Parse an uploaded PDF using Docling and return chunk preview."""
    if pdf_file is None:
        return "Upload a PDF to see Docling parsing in action."

    try:
        from docling.document_converter import DocumentConverter
        converter = DocumentConverter()
        result    = converter.convert(pdf_file.name)
        doc       = result.document

        # Extract first few text items
        texts = []
        for item, _ in doc.iterate_items():
            item_type = type(item).__name__
            if item_type in ("TextItem", "ParagraphItem") and hasattr(item, "text"):
                if item.text.strip():
                    texts.append(item.text.strip())
            if len(texts) >= 10:
                break

        if not texts:
            return "⚠️ Could not extract text from this PDF."

        preview = "\n\n".join(texts[:6])
        return (
            f"✅ **Docling parsed successfully**\n\n"
            f"**Extracted {len(texts)}+ text blocks. Preview:**\n\n"
            f"---\n\n{preview[:1500]}\n\n---\n\n"
            f"*In the full pipeline, this text is chunked by section, embedded, and indexed into Qdrant.*"
        )

    except Exception as e:
        return f"❌ Parse error: {e}"


# ── Status panel ──────────────────────────────────────────────────────
def refresh_status() -> str:
    health = get_server_health()
    stats  = get_server_stats()

    if not health:
        return (
            "❌ **Server offline**\n\n"
            "Run in a terminal:\n```\npython inference/serve.py\n```"
        )

    mode      = health.get("mode", "?").upper()
    qdrant    = "✅" if health.get("qdrant")      else "❌"
    groq      = "✅" if health.get("groq")        else "❌"
    local     = "✅" if health.get("local_model") else "⚠️ needs GGUF"
    points    = stats.get("total_points", "?")
    queries   = stats.get("queries_served", 0)
    avg_lat   = stats.get("avg_latency_ms", 0)

    return (
        f"### System Status\n\n"
        f"| Component | Status |\n"
        f"|-----------|--------|\n"
        f"| Qdrant    | {qdrant} `{points} chunks` |\n"
        f"| Groq API  | {groq} |\n"
        f"| Local GGUF| {local} |\n"
        f"| Mode      | `{mode}` |\n\n"
        f"**Queries served:** {queries}  \n"
        f"**Avg latency:** {avg_lat}ms"
    )


# ── Build Gradio UI ───────────────────────────────────────────────────
def build_ui() -> gr.Blocks:
    health = get_server_health()
    server_up = bool(health)

    with gr.Blocks(
        title = "DocRAG — CV Research Intelligence",
        theme = gr.themes.Base(
            primary_hue   = gr.themes.colors.slate,
            secondary_hue = gr.themes.colors.zinc,
            neutral_hue   = gr.themes.colors.zinc,
            font          = [gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
        ),
        css = """
        .gradio-container { max-width: 1200px; margin: 0 auto; }
        footer { display: none !important; }
        """,
    ) as demo:

        gr.Markdown("""
# ◈ DocRAG Intelligence Platform
### Computer Vision Research Assistant · Hybrid RAG · Fine-tuned LLM

> Ask questions across **2,141 indexed chunks** from 76 CV/ML papers (CVPR · ICCV · NeurIPS)
""")

        if not server_up:
            gr.Markdown("""
> ⚠️ **Inference server not detected.** Start it with:
> ```
> python inference/serve.py
> ```
""")

        with gr.Tabs():

            # ── Tab 1: Query ─────────────────────────────────────────
            with gr.TabItem("🔍  Query"):
                with gr.Row():
                    with gr.Column(scale=3):
                        question_box = gr.Textbox(
                            label       = "Question",
                            placeholder = "e.g. How does DETR use transformers for object detection?",
                            lines       = 3,
                        )

                        with gr.Row():
                            mode_radio = gr.Radio(
                                choices = ["groq", "local"],
                                value   = "groq",
                                label   = "Inference Mode",
                                info    = "groq = cloud (fast) · local = fine-tuned GGUF (offline)",
                            )
                            top_k_slider = gr.Slider(
                                minimum = 1, maximum = 10, value = 5, step = 1,
                                label   = "Sources to retrieve",
                            )

                        with gr.Accordion("🔧 Metadata Filters", open=False):
                            with gr.Row():
                                year_dd = gr.Dropdown(
                                    choices = ["Any", "2019", "2020", "2021", "2022", "2023", "2024", "2025"],
                                    value   = "Any",
                                    label   = "Year",
                                )
                                section_dd = gr.Dropdown(
                                    choices = ["Any", "Abstract", "Introduction", "Methods",
                                               "Results", "Conclusion", "Related Work"],
                                    value   = "Any",
                                    label   = "Section",
                                )
                                has_code_cb = gr.Checkbox(label="Has code only", value=False)

                        query_btn = gr.Button("Ask ▶", variant="primary", elem_classes=["query-btn"])

                    with gr.Column(scale=1):
                        stats_box = gr.Markdown(
                            value = refresh_status(),
                            label = "Status",
                        )
                        refresh_btn = gr.Button("↻ Refresh", size="sm")

                with gr.Row():
                    with gr.Column(scale=2):
                        answer_box = gr.Markdown(
                            label = "Answer",
                            value = "*Answer will appear here...*",
                            elem_classes = ["answer-box"],
                        )
                    with gr.Column(scale=1):
                        latency_box = gr.Markdown(label="Latency", value="")

                sources_box = gr.Markdown(label="Sources", value="")

                # Example questions
                gr.Examples(
                    examples = [
                        ["How do transformer architectures improve object detection?"],
                        ["What is contrastive learning and how is it used in CV?"],
                        ["How does NeRF represent 3D scenes?"],
                        ["What are the main challenges in video object detection?"],
                        ["How does self-supervised learning reduce labeling requirements?"],
                    ],
                    inputs  = [question_box],
                    label   = "Example questions",
                )

                # Wire events
                query_btn.click(
                    fn      = run_query,
                    inputs  = [question_box, mode_radio, year_dd, has_code_cb, section_dd, top_k_slider],
                    outputs = [answer_box, sources_box, latency_box],
                )
                question_box.submit(
                    fn      = run_query,
                    inputs  = [question_box, mode_radio, year_dd, has_code_cb, section_dd, top_k_slider],
                    outputs = [answer_box, sources_box, latency_box],
                )
                refresh_btn.click(fn=refresh_status, outputs=stats_box)

            # ── Tab 2: Live Document Parsing ─────────────────────────
            with gr.TabItem("📄  Live Parsing (Docling)"):
                gr.Markdown("""
### Upload any PDF — watch Docling parse it in real time

This demonstrates the ingestion pipeline: PDF → Docling → structured text chunks.
In the full pipeline, these chunks are embedded and indexed into Qdrant.
""")
                pdf_upload = gr.File(
                    label      = "Upload PDF",
                    file_types = [".pdf"],
                )
                parse_output = gr.Markdown(value="*Upload a PDF to see Docling parsing...*")
                pdf_upload.change(fn=parse_uploaded_pdf, inputs=pdf_upload, outputs=parse_output)

            # ── Tab 3: Architecture ──────────────────────────────────
            with gr.TabItem("🏗️  Architecture"):
                gr.Markdown("""
### System Architecture

```
 PDF / arXiv API
      │
      ▼
 [ Docling ]          Document parsing — extracts sections, tables, text
      │                preserving structure from complex PDFs
      ▼
 [ Qdrant ]           Vector store — 2,141 chunks indexed with:
      │                  • Dense vectors  (sentence-transformers 384d)
      │                  • Sparse vectors (BM25 via FastEmbed)
      │                  • Payload metadata (year, section, has_code...)
      │
      ▼
 User Query ──────────────────────────────────────┐
      │                                           │
      ▼                                           │
 Hybrid Search                             Metadata Filter
 (RRF fusion of dense + sparse)            year / section / has_code
      │
      ▼
 Context Assembly
 (top-k chunks with source labels)
      │
      ├──── Cloud Mode ──── [ Groq API ]        llama-3.1-8b-instant
      │                                         ~1000ms · ultra-fast
      │
      └──── Local Mode ──── [ llama.cpp ]       Fine-tuned GGUF (Q4_K_M)
                                                QLoRA via Unsloth on RTX 3060
                                                Fully offline · private
```

### RAG Techniques Used

| Technique | Implementation |
|-----------|---------------|
| **Hybrid Search** | Dense (semantic) + Sparse (BM25) with RRF fusion |
| **Section-aware chunking** | Docling preserves paper section boundaries |
| **Metadata filtering** | Qdrant payload indexes on year, section, has_code |
| **Chunk overlap** | 2-sentence overlap between adjacent chunks |
| **Domain fine-tuning** | Unsloth QLoRA on synthetic CV/ML QA pairs |

### Tech Stack

| Layer | Tool | Role |
|-------|------|------|
| Ingestion | **Docling** | PDF → structured Markdown |
| Vector Store | **Qdrant** | Hybrid search + filtering |
| Data Generation | **Groq** | Synthetic QA pair generation |
| Fine-tuning | **Unsloth** | QLoRA on RTX 3060 6GB |
| Local Inference | **llama.cpp** | Offline GGUF serving |
| Cloud Inference | **Groq** | Fast API inference |
""")

            # ── Tab 4: Evaluation ────────────────────────────────────
            with gr.TabItem("📊  Evaluation"):
                gr.Markdown("""
### Ragas Evaluation

Run the evaluation harness to measure RAG quality:

```bash
python evaluation/run_ragas.py --n 20
```

Results are saved to `evaluation/results/ragas_report.md` and logged to MLflow.

```bash
mlflow ui --port 5000
```
""")
                scores_path = Path("evaluation/results/ragas_scores.json")
                if scores_path.exists():
                    with open(scores_path) as f:
                        scores = json.load(f)

                    def score_bar(v):
                        filled = int(v * 20)
                        return "█" * filled + "░" * (20 - filled)

                    gr.Markdown(f"""
### Latest Scores  ·  mode: `{scores.get('mode', '?').upper()}`  ·  n={scores.get('n_questions', '?')}

| Metric | Score | Bar | Target |
|--------|-------|-----|--------|
| Faithfulness | `{scores.get('faithfulness', 0):.3f}` | `{score_bar(scores.get('faithfulness',0))}` | > 0.80 |
| Answer Relevancy | `{scores.get('answer_relevancy', 0):.3f}` | `{score_bar(scores.get('answer_relevancy',0))}` | > 0.75 |
| Context Recall | `{scores.get('context_recall', 0):.3f}` | `{score_bar(scores.get('context_recall',0))}` | > 0.70 |
| Context Precision | `{scores.get('context_precision', 0):.3f}` | `{score_bar(scores.get('context_precision',0))}` | > 0.70 |
""")
                else:
                    gr.Markdown("*No evaluation results yet. Run `python evaluation/run_ragas.py` first.*")

    return demo


# ── Entrypoint ────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DocRAG Gradio UI")
    parser.add_argument("--port",  type=int,  default=7860)
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--host",  default="0.0.0.0")
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(
        server_name = args.host,
        server_port = args.port,
        share       = args.share,
    )