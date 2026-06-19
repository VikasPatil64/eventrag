"""
Structured logging configuration — call configure_logging() once at startup.
"""

import logging
import sys


def configure_logging(level: int = logging.INFO) -> None:
    """
    Call once at application startup (in main.py lifespan).

    Format:  2026-06-18 14:00:00,123 | INFO     | app.ingestion.embedder | Embedded 12 texts
    """
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,  # override any root-logger config set by uvicorn/inngest
    )
    # Quieten noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("qdrant_client").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("google").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — use `logger = get_logger(__name__)` in every module."""
    return logging.getLogger(name)
