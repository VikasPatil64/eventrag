"""
Structured logging configuration.

FIX B-15: The original project had zero application-level logging.
          Every module now calls `get_logger(__name__)` for consistent,
          labelled output that is useful in both local and Docker environments.
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


def get_logger(name: str) -> logging.Logger:
    """Convenience wrapper — use `logger = get_logger(__name__)` in every module."""
    return logging.getLogger(name)
