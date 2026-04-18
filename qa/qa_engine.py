# -*- coding: utf-8 -*-
"""
QA generation: synthesizes answers from retrieved chunks using the LLM.
Includes source citations per chunk.
"""

import logging
from typing import List, Tuple
from ingestion.pdf_extractor import Chunk
from utils.llm_client import LLMClient

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a precise and helpful document assistant.
Answer the user's question based ONLY on the provided context excerpts.
Synthesize a clear, concise answer — do not copy text verbatim.
After your answer, list the sources used in format: (Source: Page X).
If the context does not contain enough information, say so honestly.
"""


def answer_query(
    query: str,
    retrieved: List[Tuple[Chunk, float]],
    llm_client: LLMClient,
) -> Tuple[str, List[str]]:
    """
    Generate an answer from retrieved chunks.
    Returns (answer_text, list_of_citations).
    """
    if not retrieved:
        return "No relevant content found in the document for this query.", []

    # Build context block
    context_parts = []
    citations = []
    for chunk, score in retrieved:
        citation = chunk.citation()
        context_parts.append(f"[{citation}]\n{chunk.content}")
        if citation not in citations:
            citations.append(citation)

    context = "\n\n---\n\n".join(context_parts)

    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        answer = llm_client.chat(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
        )
        return answer, citations

    except ValueError as e:
        # Auth error — bubble up with clean message
        raise
    except RuntimeError as e:
        raise
    except Exception as e:
        logger.error("QA generation error: %s", e)
        raise RuntimeError(f"Answer generation failed: {e}")
