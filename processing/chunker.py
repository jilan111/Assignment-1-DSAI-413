# -*- coding: utf-8 -*-
"""
Smart chunking: semantic + structural splitting of raw chunks.
"""

import re
import logging
from typing import List
from ingestion.pdf_extractor import Chunk

logger = logging.getLogger(__name__)

MAX_CHUNK_CHARS = 800
OVERLAP_CHARS = 100


def split_text(text: str, max_chars: int = MAX_CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> List[str]:
    """Split long text into overlapping windows, respecting sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # Start new chunk with overlap from end of previous
            current = (current[-overlap:] + " " + sentence).strip() if current else sentence

    if current:
        chunks.append(current)

    return chunks or [text]


def chunk_documents(raw_chunks: List[Chunk]) -> List[Chunk]:
    """
    Apply smart chunking:
    - Tables and images are kept as single chunks (already compact)
    - Text chunks are split if too long
    """
    result: List[Chunk] = []
    new_id = 0

    for chunk in raw_chunks:
        if chunk.chunk_type in ("table", "image"):
            chunk.chunk_id = new_id
            result.append(chunk)
            new_id += 1
            continue

        # Text: split if needed
        sub_texts = split_text(chunk.content)
        for sub in sub_texts:
            result.append(
                Chunk(
                    content=sub,
                    page=chunk.page,
                    chunk_type=chunk.chunk_type,
                    chunk_id=new_id,
                    metadata=chunk.metadata,
                )
            )
            new_id += 1

    logger.info("Chunking produced %d chunks", len(result))
    return result
