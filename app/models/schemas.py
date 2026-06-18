"""
Pydantic v2 models shared across the ingestion and retrieval pipelines.

FIX B-06: source_id: str = None → Optional[str] = None (Pydantic v2 requires
          Optional for fields that can be None).
FIX B-07: RAQQueryResult typo renamed to RAGQueryResult; added proper fields.
FIX B-18: rag_query_pdf_ai now returns RAGQueryResult not RAGSearchResult.
"""

from typing import Optional

import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    """Output of the load-and-chunk step."""

    chunks: list[str]
    # FIX B-06: was `str = None` — Pydantic v2 raises a validation error for that.
    source_id: Optional[str] = None


class RAGUpsertResult(pydantic.BaseModel):
    """Output of the embed-and-upsert step."""

    ingested: int
    source_id: Optional[str] = None


class RAGSearchResult(pydantic.BaseModel):
    """Output of the embed-and-search step."""

    contexts: list[str]
    sources: list[str]


class RAGQueryResult(pydantic.BaseModel):
    """Final output of the rag_query_pdf_ai function.

    FIX B-07: was named RAQQueryResult (typo). Now correctly RAGQueryResult.
    """

    answer: str
    sources: list[str]
    num_contexts: int
