# -*- coding: utf-8 -*-
"""
Multi-Modal RAG QA System — Streamlit Application
Feminine design: blush, lavender, gold accents, glassmorphism panels.
"""

import sys
import os
import logging
import streamlit as st

# Ensure UTF-8 across all I/O
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass  # Ignore if pipes are broken or reconfigure not available

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind ✦ RAG Assistant",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — feminine glassmorphism design ────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700&family=Playfair+Display:wght@400;600&display=swap');

    :root {
        --blush:    #f9c4d2;
        --lavender: #c9b8e8;
        --beige:    #f5ede0;
        --ivory:    #fdf8f3;
        --gold:     #d4a843;
        --rose:     #e8829a;
        --text:     #4a3f55;
    }

    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif;
        color: var(--text);
        background: linear-gradient(135deg, #fdf0f5 0%, #efe8f8 50%, #f5ede0 100%);
    }

    /* Hide default Streamlit header */
    header[data-testid="stHeader"] { display: none; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f9d0de 0%, #e8d8f5 100%);
        border-right: 1px solid var(--blush);
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.6);
        border: 2px dashed var(--lavender);
        border-radius: 16px;
        padding: 12px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--rose), var(--lavender));
        color: white !important;
        border: none;
        border-radius: 24px;
        padding: 10px 28px;
        font-family: 'Nunito', sans-serif;
        font-weight: 600;
        font-size: 15px;
        cursor: pointer;
        transition: transform 0.15s, box-shadow 0.15s;
        box-shadow: 0 4px 14px rgba(232,130,154,0.35);
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(232,130,154,0.5);
    }

    /* Text input */
    .stTextInput > div > div > input,
    .stTextArea textarea {
        background: rgba(255,255,255,0.75);
        border: 1.5px solid var(--blush);
        border-radius: 12px;
        font-family: 'Nunito', sans-serif;
        color: var(--text);
    }

    /* Chat bubbles */
    .chat-user {
        background: linear-gradient(135deg, var(--rose), #d4669c);
        color: white;
        border-radius: 20px 20px 4px 20px;
        padding: 14px 18px;
        margin: 8px 0 8px 20%;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 3px 12px rgba(232,130,154,0.3);
    }
    .chat-assistant {
        background: rgba(255,255,255,0.85);
        border: 1px solid var(--lavender);
        border-radius: 20px 20px 20px 4px;
        padding: 14px 18px;
        margin: 8px 20% 8px 0;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 3px 12px rgba(201,184,232,0.25);
        backdrop-filter: blur(8px);
    }
    .citation-badge {
        display: inline-block;
        background: linear-gradient(135deg, var(--gold), #e8b84b);
        color: white;
        border-radius: 10px;
        padding: 2px 10px;
        font-size: 12px;
        font-weight: 600;
        margin: 2px;
    }
    .hero-title {
        font-family: 'Playfair Display', serif;
        font-size: 2.4rem;
        background: linear-gradient(135deg, var(--rose), var(--lavender));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        margin-bottom: 0;
    }
    .hero-subtitle {
        text-align: center;
        color: #9c8aad;
        font-size: 1rem;
        margin-top: 4px;
    }
    .glass-card {
        background: rgba(255,255,255,0.65);
        border: 1px solid rgba(201,184,232,0.4);
        border-radius: 20px;
        padding: 20px 24px;
        backdrop-filter: blur(10px);
        box-shadow: 0 6px 24px rgba(180,150,210,0.12);
        margin-bottom: 16px;
    }
    .status-ok  { color: #6abf69; font-weight: 600; }
    .status-err { color: #e87a6f; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
defaults = {
    "messages": [],
    "vector_store": None,
    "llm_client": None,
    "doc_loaded": False,
    "doc_name": "",
    "chunk_count": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Auto-initialize LLM client ──────────────────────────────────────────────────
if st.session_state["llm_client"] is None:
    from utils.llm_client import LLMClient
    st.session_state["llm_client"] = LLMClient()  # Uses Groq hardcoded credentials


# ── Helper: display alert ─────────────────────────────────────────────────────
def show_error(msg: str):
    st.markdown(
        f'<div class="glass-card"><span class="status-err">⚠️ {msg}</span></div>',
        unsafe_allow_html=True,
    )


def show_success(msg: str):
    st.markdown(
        f'<div class="glass-card"><span class="status-ok">✓ {msg}</span></div>',
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✦ DocMind")
    st.markdown("*Your intelligent document assistant*")
    st.divider()

    # File upload
    st.markdown("### 📄 Upload Document")
    uploaded = st.file_uploader(
        "Drop your PDF here",
        type=["pdf"],
        accept_multiple_files=False,
    )

    top_k = st.slider("Retrieved chunks (top-k)", 2, 10, 5)

    if uploaded and st.button("⚡ Process Document"):
        with st.spinner("Extracting and indexing document…"):
            try:
                from ingestion.pdf_extractor import extract_from_pdf
                from processing.chunker import chunk_documents
                from retrieval.vector_store import VectorStore

                raw_chunks = extract_from_pdf(uploaded.read(), uploaded.name)
                chunks = chunk_documents(raw_chunks)
                vs = VectorStore()
                vs.build(chunks)

                st.session_state["vector_store"] = vs
                st.session_state["doc_loaded"] = True
                st.session_state["doc_name"] = uploaded.name
                st.session_state["chunk_count"] = len(chunks)
                st.session_state["messages"] = []

                show_success(f"Indexed {len(chunks)} chunks — ready to answer!")
            except Exception as e:
                logger.error("Processing error: %s", e)
                show_error(str(e))

    if st.session_state["doc_loaded"]:
        st.markdown(
            f'<div class="glass-card">📂 <b>{st.session_state["doc_name"]}</b><br>'
            f'<small>{st.session_state["chunk_count"]} chunks indexed</small></div>',
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("🗑️ Clear Chat"):
        st.session_state["messages"] = []
        st.rerun()


# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">✦ DocMind RAG Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Upload any document · Ask anything · Get cited answers</p>', unsafe_allow_html=True)
st.markdown("")

if not st.session_state["doc_loaded"]:
    st.markdown(
        """
        <div class="glass-card" style="text-align:center; padding:40px;">
            <div style="font-size:3rem;">📄</div>
            <h3 style="font-family:'Playfair Display',serif; color:#c97a99;">
                Upload a PDF to get started
            </h3>
            <p style="color:#9c8aad;">
                Use the sidebar to configure your API key, upload a document,<br>
                then ask any question about its content.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    # Chat history
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state["messages"]:
            if msg["role"] == "user":
                st.markdown(
                    f'<div class="chat-user">🙋 {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                citations_html = "".join(
                    f'<span class="citation-badge">{c}</span>'
                    for c in msg.get("citations", [])
                )
                st.markdown(
                    f'<div class="chat-assistant">✦ {msg["content"]}'
                    f'{"<br><br>" + citations_html if citations_html else ""}'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # Input
    st.markdown("")
    query = st.text_input(
        "Query",
        placeholder="Ask anything about your document…",
        key="query_input",
        label_visibility="collapsed",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        send = st.button("Send ✦")

    if send and query.strip():
        st.session_state["messages"].append({"role": "user", "content": query.strip()})

        with st.spinner("Thinking…"):
            try:
                from qa.qa_engine import answer_query

                vs = st.session_state["vector_store"]
                llm = st.session_state["llm_client"]

                retrieved = vs.search(query.strip(), top_k=top_k)
                answer, citations = answer_query(query.strip(), retrieved, llm)

                st.session_state["messages"].append(
                    {"role": "assistant", "content": answer, "citations": citations}
                )

            except ValueError as e:
                # Auth / key error
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": str(e),
                        "citations": [],
                    }
                )
            except RuntimeError as e:
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": str(e),
                        "citations": [],
                    }
                )
            except Exception as e:
                logger.error("Unexpected error: %s", e)
                st.session_state["messages"].append(
                    {
                        "role": "assistant",
                        "content": "An unexpected error occurred. Please try again.",
                        "citations": [],
                    }
                )

        st.rerun()
