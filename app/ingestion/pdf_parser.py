"""
PDF loading and chunking service.

Pages are joined before splitting so sentence-boundary chunks aren't broken
at page edges (a paragraph spanning pages stays intact).
"""

import logging
from pathlib import Path

from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader

from app.config.settings import get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


def load_and_chunk_pdf(path: str) -> list[str]:
    """
    Load a PDF and return a list of text chunks.

    Returns [] if the PDF has no extractable text (e.g. scanned/image-only).
    """
    settings = get_settings()
    splitter = SentenceSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    resolved = Path(path).resolve()
    logger.info("Loading PDF: %s", resolved)

    try:
        docs = PDFReader().load_data(file=resolved)
    except Exception as exc:
        logger.error("PDFReader failed for '%s': %s", resolved, exc)
        raise

    texts = [
        d.text
        for d in docs
        if getattr(d, "text", None) and d.text.strip()
    ]

    if not texts:
        logger.warning("No extractable text found in '%s' (image-only PDF?).", resolved)
        return []

    full_text = "\n\n".join(texts)
    chunks = splitter.split_text(full_text)

    logger.info(
        "Chunked '%s' -> %d page(s), %d chunk(s) (chunk_size=%d, overlap=%d).",
        resolved.name,
        len(docs),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
