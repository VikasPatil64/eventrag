"""
Shared data models used across the ingestion and retrieval pipeline.

These models are passed between different steps of the RAG workflow
and keep the data structure consistent throughout the application.
"""

from typing import Optional

import pydantic


class RAGChunkAndSrc(pydantic.BaseModel):
    """
    Represents the output of the PDF parsing step.

    Stores the text chunks extracted from a document along with
    an optional source identifier.
    """

    chunks: list[str]
    source_id: Optional[str] = None


class RAGUpsertResult(pydantic.BaseModel):
    """
    Stores metadata about the vector database insertion process.
    """

    ingested: int
    source_id: Optional[str] = None


class RAGSearchResult(pydantic.BaseModel):
    """
    Represents the chunks retrieved from semantic search
    together with their corresponding source identifiers.
    """

    contexts: list[str]
    sources: list[str]


class RAGQueryResult(pydantic.BaseModel):
    """
    Final response returned to the user after retrieval
    and answer generation.
    """

    answer: str
    sources: list[str]
    num_contexts: int