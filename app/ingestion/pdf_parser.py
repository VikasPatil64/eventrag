"""
PDF loading and chunking service.

FIX B-11: The original code split each PDF *page* individually with a loop:
              for t in texts:           # t = one page
                  chunks.extend(splitter.split_text(t))
          This loses sentences that span page boundaries (e.g. a paragraph
          that starts on page 3 and ends on page 4 becomes two tiny,
          context-free fragments).  We now join all pages first, then chunk
          the full document once.
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
    Load a PDF from *path* and return a list of text chunks.

    Returns an empty list if the PDF contains no extractable text
    (e.g. a scanned/image-only PDF) instead of crashing.
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
        logger.warning("No extractable text found in '%s' (image-only PDF?)", resolved)
        return []

    # FIX B-11: join all pages FIRST, then split once across the full document.
    full_text = "\n\n".join(texts)
    chunks = splitter.split_text(full_text)

    logger.info(
        "Chunked '%s' → %d page(s), %d chunk(s) "
        "(chunk_size=%d, overlap=%d)",
        resolved.name,
        len(docs),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
