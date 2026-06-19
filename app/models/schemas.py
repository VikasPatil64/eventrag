"""
<<<<<<< HEAD
Shared data models used across the ingestion and retrieval pipeline.

These models are passed between different steps of the RAG workflow
and keep the data structure consistent throughout the application.
=======
Pydantic v2 models shared across the ingestion and retrieval pipelines.
>>>>>>> 4bb0fcb (Add provider-agnostic RAG with Gemini and local embeddings)
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
<<<<<<< HEAD
    """
    Final response returned to the user after retrieval
    and answer generation.
    """
=======
    """Final output of the rag_query_pdf_ai function."""
>>>>>>> 4bb0fcb (Add provider-agnostic RAG with Gemini and local embeddings)

    answer: str
    sources: list[str]
    num_contexts: int