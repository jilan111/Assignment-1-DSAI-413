# DocMind — Multi-Modal RAG QA System

Upload any PDF. Ask anything. Get answers with page citations.

DocMind is a Retrieval-Augmented Generation (RAG) system that extracts text, tables, and image captions from PDF documents and answers natural-language queries with traceable source citations. It runs on any OpenAI-compatible LLM backend (OpenAI, Groq, Together, Ollama).

![Python](https://img.shields.io/badge/python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![FAISS](https://img.shields.io/badge/vector_store-FAISS-green)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## Demo

> _Add a screenshot or 30-second GIF of the Streamlit UI here (`docs/demo.gif`)._

---

## Features

- **Multi-modal extraction** — text, tables, and image captions parsed from PDFs via pdfplumber
- **Semantic + structural chunking** with configurable overlap
- **Vector retrieval** — FAISS embeddings (sentence-transformers), with TF-IDF fallback when GPU/embeddings are unavailable
- **Cited answers** — every response includes the source page numbers
- **Pluggable LLM** — works with any OpenAI-compatible API
- **Built-in evaluation** harness with sample queries

---

## Architecture

```
PDF ──▶ Ingestion ──▶ Chunking ──▶ Embeddings ──▶ FAISS Index
                                                       │
                                                       ▼
User Query ──▶ Retrieval ──▶ LLM (with context) ──▶ Cited Answer
```

---

## Quick start

```bash
git clone https://github.com/jilan111/DocMind.git
cd DocMind
pip install -r requirements.txt

# Configure your LLM
export OPENAI_API_KEY=sk-...
export OPENAI_BASE_URL=https://api.openai.com/v1   # or Groq, Ollama, etc.

streamlit run app.py
```

Lightweight install (TF-IDF fallback, no GPU embeddings):

```bash
pip install -r requirements-lite.txt
```

---

## Project structure

```
multimodal_rag/
├── app.py                    # Streamlit entry point
├── ingestion/
│   └── pdf_extractor.py      # Multi-modal PDF extraction
├── processing/
│   └── chunker.py            # Semantic & structural chunking
├── retrieval/
│   └── vector_store.py       # FAISS / TF-IDF
├── qa/
│   └── qa_engine.py          # LLM QA with citations
├── utils/
│   └── llm_client.py         # LLM abstraction
├── evaluation/
│   └── eval_queries.py       # Sample queries + evaluator
├── data/                     # Test PDFs
└── report/                   # Technical documentation
```

---

## Tech stack

| Layer | Tool |
|-------|------|
| UI | Streamlit |
| PDF parsing | pdfplumber |
| Embeddings | sentence-transformers |
| Vector store | FAISS (TF-IDF fallback) |
| LLM | OpenAI-compatible API |
| Language | Python 3.10+ |

---

## Roadmap

- [ ] Async ingestion for large PDFs
- [ ] Hybrid (BM25 + dense) retrieval
- [ ] Re-ranker stage
- [ ] Persistent index across sessions
- [ ] Dockerfile + one-command deploy

---

## License

MIT — see [LICENSE](LICENSE).

---

## Author

Built by **Jilan Ismail** — [GitHub](https://github.com/jilan111) · [LinkedIn](https://www.linkedin.com/in/jilan-ismail-596b2b2b2/)
