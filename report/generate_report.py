# -*- coding: utf-8 -*-
"""Generate the project report PDF (max 2 pages)."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.units import cm

PINK   = colors.HexColor("#e8829a")
LAVEN  = colors.HexColor("#c9b8e8")
GOLD   = colors.HexColor("#d4a843")
TEXT   = colors.HexColor("#4a3f55")
LIGHT  = colors.HexColor("#f9f0f5")

def build_report(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm,  bottomMargin=2*cm,
    )
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "title", parent=styles["Title"],
        fontSize=22, textColor=PINK, spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    h1 = ParagraphStyle(
        "h1", parent=styles["Heading1"],
        fontSize=13, textColor=LAVEN, spaceBefore=10, spaceAfter=4,
        fontName="Helvetica-Bold",
    )
    body = ParagraphStyle(
        "body", parent=styles["Normal"],
        fontSize=10, textColor=TEXT, spaceAfter=6, leading=15,
    )
    small = ParagraphStyle(
        "small", parent=body, fontSize=9, textColor=colors.HexColor("#7a6a8a"),
    )

    story = []

    # Title
    story.append(Paragraph("DocMind — Multi-Modal RAG QA System", title_style))
    story.append(Paragraph("Technical Report", small))
    story.append(HRFlowable(width="100%", color=PINK, thickness=1.5, spaceAfter=8))

    # 1. Architecture
    story.append(Paragraph("1. System Architecture", h1))
    story.append(Paragraph(
        "DocMind follows a five-stage pipeline: (1) Multi-modal Ingestion, "
        "(2) Semantic Chunking, (3) Embedding Generation, (4) Vector Retrieval, "
        "and (5) LLM-based Answer Synthesis with Citations.", body))

    arch_data = [
        ["Stage", "Module", "Technology"],
        ["Ingestion",  "ingestion/pdf_extractor.py",  "pdfplumber + pypdf"],
        ["Chunking",   "processing/chunker.py",        "Custom semantic splitter"],
        ["Embeddings", "retrieval/vector_store.py",    "sentence-transformers / TF-IDF"],
        ["Retrieval",  "retrieval/vector_store.py",    "FAISS (L2) / cosine fallback"],
        ["QA",         "qa/qa_engine.py",              "OpenAI-compatible LLM"],
    ]
    arch_table = Table(arch_data, colWidths=[3.5*cm, 6.5*cm, 5.5*cm])
    arch_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), PINK),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.4, LAVEN),
        ("ROUNDEDCORNERS", [4]),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING",    (0, 0), (-1, -1), 5),
    ]))
    story.append(arch_table)
    story.append(Spacer(1, 10))

    # 2. Design Decisions
    story.append(Paragraph("2. Design Decisions", h1))
    decisions = [
        ("<b>Generic LLM layer:</b> utils/llm_client.py abstracts the provider — "
         "swap base_url to use Groq, Together, Ollama, etc."),
        ("<b>Graceful degradation:</b> If FAISS is unavailable, the system falls "
         "back to TF-IDF cosine retrieval automatically."),
        ("<b>UTF-8 enforcement:</b> All I/O is explicitly configured to UTF-8 "
         "to prevent encoding errors with Arabic, emojis, and special characters."),
        ("<b>Error isolation:</b> Every pipeline stage is wrapped in try/except; "
         "user-facing messages are clean and actionable."),
        ("<b>Generic UI:</b> No hardcoded topics — works with any uploaded PDF."),
    ]
    for d in decisions:
        story.append(Paragraph(f"&#x2022; {d}", body))

    # 3. Evaluation
    story.append(Paragraph("3. Evaluation", h1))
    story.append(Paragraph(
        "Five representative queries were designed to test all system components:", body))

    eval_data = [
        ["#", "Query",                                      "Expected"],
        ["1", "Main objective of the document?",            "Synthesized intro summary + citation"],
        ["2", "Key findings / conclusions?",                "Conclusion bullets + citation"],
        ["3", "Tables in document?",                        "Table content description + citation"],
        ["4", "What methodology is described?",             "Method synthesis + citation"],
        ["5", "Recommendations / future work?",             "Recommendations list + citation"],
    ]
    eval_table = Table(eval_data, colWidths=[0.7*cm, 7.5*cm, 7.3*cm])
    eval_table.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), LAVEN),
        ("TEXTCOLOR",    (0, 0), (-1, 0), colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",     (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, colors.white]),
        ("GRID",         (0, 0), (-1, -1), 0.4, LAVEN),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
    ]))
    story.append(eval_table)
    story.append(Spacer(1, 10))

    # 4. Bug Fixes
    story.append(Paragraph("4. Critical Bug Fixes Applied", h1))
    fixes = [
        "UTF-8 encoding enforced on stdout/stderr and all file I/O",
        "API 401 errors caught and shown as 'Invalid API key. Please verify and try again.'",
        "All LLM, retrieval, and parsing calls wrapped in try/except",
        "No stack traces exposed in UI — clean error messages only",
        "Removed all hardcoded topics (HDFS, YARN, etc.) from the UI",
    ]
    for f in fixes:
        story.append(Paragraph(f"&#x2713; {f}", body))

    story.append(Spacer(1, 8))
    story.append(HRFlowable(width="100%", color=LAVEN, thickness=0.8))
    story.append(Paragraph(
        "DocMind RAG System — Production-ready submission | Built with Python, Streamlit, FAISS",
        small,
    ))

    doc.build(story)
    print(f"Report saved to: {output_path}")


if __name__ == "__main__":
    build_report("report/report.pdf")
