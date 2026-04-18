# -*- coding: utf-8 -*-
"""
Embedding + FAISS vector store + top-k retrieval.
Falls back to TF-IDF BM25-style retrieval if sentence-transformers unavailable.
"""

import logging
import numpy as np
from typing import List, Tuple
from ingestion.pdf_extractor import Chunk

logger = logging.getLogger(__name__)

_EMBED_MODEL = None
_USE_TFIDF = False


def _get_embedder():
    global _EMBED_MODEL, _USE_TFIDF
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        _USE_TFIDF = False
        logger.info("Using SentenceTransformer for embeddings.")
    except ImportError:
        logger.warning("sentence-transformers not available; using TF-IDF fallback.")
        _USE_TFIDF = True
        _EMBED_MODEL = "tfidf"
    return _EMBED_MODEL


class VectorStore:
    """FAISS-backed vector store with fallback to cosine TF-IDF."""

    def __init__(self):
        self.chunks: List[Chunk] = []
        self.index = None
        self._tfidf_matrix = None
        self._vectorizer = None

    def build(self, chunks: List[Chunk]):
        self.chunks = chunks
        texts = [c.content for c in chunks]
        embedder = _get_embedder()

        if _USE_TFIDF or embedder == "tfidf":
            self._build_tfidf(texts)
        else:
            self._build_faiss(embedder, texts)

    def _build_faiss(self, embedder, texts: List[str]):
        try:
            import faiss
            embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            embeddings = embeddings.astype(np.float32)
            dim = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings)
            self._embeddings = embeddings
            logger.info("FAISS index built with %d vectors (dim=%d)", len(texts), dim)
        except ImportError:
            logger.warning("FAISS not available; switching to TF-IDF.")
            self._build_tfidf(texts)

    def _build_tfidf(self, texts: List[str]):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self._tfidf_matrix = self._vectorizer.fit_transform(texts)
        logger.info("TF-IDF index built with %d documents", len(texts))

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not self.chunks:
            return []

        embedder = _get_embedder()

        if self._vectorizer is not None:
            return self._search_tfidf(query, top_k)
        elif self.index is not None:
            return self._search_faiss(embedder, query, top_k)
        else:
            return [(c, 1.0) for c in self.chunks[:top_k]]

    def _search_faiss(self, embedder, query: str, top_k: int):
        import faiss
        q_emb = embedder.encode([query], convert_to_numpy=True).astype(np.float32)
        distances, indices = self.index.search(q_emb, min(top_k, len(self.chunks)))
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.chunks):
                results.append((self.chunks[idx], float(dist)))
        return results

    def _search_tfidf(self, query: str, top_k: int):
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._tfidf_matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices if scores[i] > 0]
