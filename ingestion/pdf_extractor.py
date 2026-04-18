# -*- coding: utf-8 -*-
"""
Multi-modal ingestion: extracts text, tables, and image captions from PDFs.
"""

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single content chunk from the document."""
    content: str
    page: int
    chunk_type: str          # "text" | "table" | "image"
    chunk_id: int = 0
    metadata: dict = field(default_factory=dict)

    def citation(self) -> str:
        return f"(Source: Page {self.page})"


def extract_from_pdf(file_bytes: bytes, filename: str = "document.pdf") -> List[Chunk]:
    """
    Full multi-modal extraction pipeline:
    1. Text  — via pdfplumber
    2. Tables — via pdfplumber
    3. Images — basic captioning via PIL (if available)
    """
    chunks: List[Chunk] = []
    chunk_id = 0

    try:
        import pdfplumber

        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                # ── Text ────────────────────────────────────────────────────
                try:
                    text = page.extract_text() or ""
                    text = text.strip()
                    if text:
                        chunks.append(
                            Chunk(
                                content=text,
                                page=page_num,
                                chunk_type="text",
                                chunk_id=chunk_id,
                            )
                        )
                        chunk_id += 1
                except Exception as e:
                    logger.warning("Text extraction failed on page %d: %s", page_num, e)

                # ── Tables ──────────────────────────────────────────────────
                try:
                    tables = page.extract_tables() or []
                    for t_idx, table in enumerate(tables):
                        if not table:
                            continue
                        rows = []
                        for row in table:
                            cleaned = [str(cell).strip() if cell is not None else "" for cell in row]
                            rows.append(" | ".join(cleaned))
                        table_text = "\n".join(rows)
                        if table_text.strip():
                            chunks.append(
                                Chunk(
                                    content=f"[TABLE]\n{table_text}",
                                    page=page_num,
                                    chunk_type="table",
                                    chunk_id=chunk_id,
                                    metadata={"table_index": t_idx},
                                )
                            )
                            chunk_id += 1
                except Exception as e:
                    logger.warning("Table extraction failed on page %d: %s", page_num, e)

                # ── Images ──────────────────────────────────────────────────
                try:
                    images = page.images or []
                    for img_idx, img_info in enumerate(images):
                        caption = (
                            f"[IMAGE on page {page_num}] "
                            f"Dimensions: {img_info.get('width', '?')}×{img_info.get('height', '?')} px"
                        )
                        chunks.append(
                            Chunk(
                                content=caption,
                                page=page_num,
                                chunk_type="image",
                                chunk_id=chunk_id,
                                metadata={"image_index": img_idx},
                            )
                        )
                        chunk_id += 1
                except Exception as e:
                    logger.warning("Image extraction failed on page %d: %s", page_num, e)

    except ImportError:
        # Fallback: pypdf text only
        logger.warning("pdfplumber not available; falling back to pypdf text extraction.")
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_num, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    chunks.append(
                        Chunk(content=text, page=page_num, chunk_type="text", chunk_id=chunk_id)
                    )
                    chunk_id += 1
        except Exception as e:
            logger.error("PDF extraction completely failed: %s", e)
            raise RuntimeError(f"Could not extract content from PDF: {e}")

    except Exception as e:
        logger.error("Ingestion error: %s", e)
        raise RuntimeError(f"Document ingestion failed: {e}")

    logger.info("Extracted %d chunks from '%s'", len(chunks), filename)
    return chunks
