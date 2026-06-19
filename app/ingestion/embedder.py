"""
Embedding service — dispatches to local (sentence-transformers) or OpenAI
based on the EMBED_PROVIDER setting.
"""

from app.config.settings import get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# OpenAI client singleton — only instantiated when EMBED_PROVIDER=openai.
_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        settings = get_settings()
        _openai_client = OpenAI(api_key=settings.openai_api_key)
        logger.debug("OpenAI embed client initialised (model=%s).", settings.openai_embed_model)
    return _openai_client


def _embed_openai(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    client = _get_openai_client()
    logger.info(
        "Embedding %d text(s) with OpenAI %s (dim=%d).",
        len(texts),
        settings.openai_embed_model,
        settings.openai_embed_dim,
    )
    try:
        response = client.embeddings.create(
            model=settings.openai_embed_model,
            input=texts,
            dimensions=settings.openai_embed_dim,
        )
    except Exception as exc:
        logger.error("OpenAI embedding failed: %s", exc)
        raise
    embeddings = [item.embedding for item in response.data]
    logger.info("OpenAI embedding complete — %d vector(s).", len(embeddings))
    return embeddings


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using the configured provider.

    Returns [] immediately for empty input — avoids unnecessary API calls.
    """
    if not texts:
        return []

    settings = get_settings()

    if settings.embed_provider == "local":
        from app.ingestion.local_embedder import embed_texts as _local_embed
        return _local_embed(texts)

    return _embed_openai(texts)
