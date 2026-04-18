# ✦ DocMind — Multi-Modal RAG QA System

> Upload any PDF. Ask anything. Get cited answers.

A production-ready, multi-modal Retrieval-Augmented Generation (RAG) system with a beautiful feminine UI built on Streamlit. Works with **any** OpenAI-compatible LLM API.

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│  INGESTION  (ingestion/pdf_extractor.py)                │
│  Text + Tables + Image captions via pdfplumber          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  CHUNKING  (processing/chunker.py)                      │
│  Semantic + structural splitting with overlap           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  EMBEDDINGS + VECTOR STORE  (retrieval/vector_store.py) │
│  sentence-transformers → FAISS  (TF-IDF fallback)       │
└────────────────────────┬────────────────────────────────┘
                         │  top-k retrieval
                         ▼
┌─────────────────────────────────────────────────────────┐
│  QA GENERATION  (qa/qa_engine.py)                       │
│  LLM synthesizes answer + (Source: Page N) citations    │
└─────────────────────────────────────────────────────────┘
```

---

## ⚡ Quick Start

### 1. Clone / extract the project
```bash
cd multimodal_rag
```

### 2. Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your API key

**Option A — Environment variable (recommended):**
```bash
cp .env.example .env
# Edit .env and set LLM_API_KEY=sk-your-key-here
```

**Option B — UI sidebar:**  
Enter the key directly in the app sidebar. No restart needed.

### 5. Run the app
```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## 🔑 API Configuration

The system supports **any OpenAI-compatible API**. Change the base URL in the sidebar:

| Provider | Base URL |
|----------|----------|
| OpenAI   | `https://api.openai.com/v1` |
| Groq     | `https://api.groq.com/openai/v1` |
| Together | `https://api.together.xyz/v1` |
| Ollama   | `http://localhost:11434/v1` |

---

## 📄 Usage

1. Open the app in your browser
2. Enter your **API key** in the left sidebar and click **Save API Settings**
3. Click **Browse files** to upload a PDF (drag & drop supported)
4. Click **⚡ Process Document** — the system indexes the document
5. Type any question in the chat box and press **Send ✦**
6. Answers appear with **gold citation badges** showing the source page

---

## 💬 Example Queries

```
"What is the main objective of this document?"
"Summarize the key findings."
"Are there any tables? What do they contain?"
"What methodology is described?"
"List any recommendations or future work."
```

---

## 📁 Project Structure

```
multimodal_rag/
├── app.py                    # Streamlit application (entry point)
├── requirements.txt
├── .env.example
├── README.md
│
├── ingestion/
│   └── pdf_extractor.py      # Multi-modal PDF extraction
│
├── processing/
│   └── chunker.py            # Semantic + structural chunking
│
├── retrieval/
│   └── vector_store.py       # FAISS / TF-IDF vector store
│
├── qa/
│   └── qa_engine.py          # LLM QA with citations
│
├── utils/
│   └── llm_client.py         # Generic LLM abstraction layer
│
├── evaluation/
│   └── eval_queries.py       # 5 example queries + evaluator
│
├── report/
│   ├── report.pdf            # Technical report (2 pages)
│   └── generate_report.py
│
├── demo/
│   └── demo_script_ar.txt    # Egyptian Arabic demo script
│
└── data/                     # Place test PDFs here
```

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| `Invalid API key` | Double-check your key in the sidebar. Make sure no extra spaces. |
| `API unavailable` | Check your internet connection or provider status. |
| Slow processing | Large PDFs take more time. Use the top-k slider to reduce retrieval. |
| `No relevant content found` | Try rephrasing your question, or reduce top-k to 2–3. |
| Encoding errors | The app enforces UTF-8 everywhere. Restart with `streamlit run app.py`. |
| FAISS not installing | The system auto-falls back to TF-IDF. No action needed. |

---

## 🧪 Running Evaluations

```python
# In a Python shell after processing a document:
from evaluation.eval_queries import run_evaluation
results = run_evaluation(vector_store, llm_client, verbose=True)
```

---

## 📜 License

For academic submission purposes only.
