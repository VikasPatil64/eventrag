"""
OpenAI embedding service.

FIX B-04: embed_texts([]) previously called the OpenAI API with an empty
          input list → API returned a 400 error → Inngest step died.
          Now returns [] immediately.

FIX B-15: Added structured logging for every embedding call.

NOTE on dimensions (B-02 background):
  text-embedding-3-large returns 1536 dims by default.
  Settings.openai_embed_dim is set to 1536 to match.
  If you later want 3072 dims you must BOTH:
    1. Set OPENAI_EMBED_DIM=3072 in .env
    2. Pass dimensions=settings.openai_embed_dim to the API call (done here).
  We always pass `dimensions=` now so the collection dim and API output
  stay in sync regardless of which value is configured.
"""

from openai import OpenAI

from app.config.settings import get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Module-level singleton — instantiated once, reused across all calls.
# The original code created a new OpenAI() client on every import of
# data_loader.py which also blocked on network if the API was unreachable.
_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        settings = get_settings()
        _client = OpenAI(api_key=settings.openai_api_key)
        logger.debug("OpenAI client initialised (model=%s)", settings.openai_embed_model)
    return _client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of strings using the configured OpenAI embedding model.

    Returns an empty list when *texts* is empty (FIX B-04).
    Raises on API errors so the Inngest step can surface them cleanly.
    """
    # FIX B-04: guard against empty input — was crashing the Inngest step.
    if not texts:
        logger.debug("embed_texts() called with empty list — returning []")
        return []

    settings = get_settings()
    client = _get_client()

    logger.info(
        "Embedding %d text(s) with %s (dim=%d)",
        len(texts),
        settings.openai_embed_model,
        settings.openai_embed_dim,
    )

    try:
        response = client.embeddings.create(
            model=settings.openai_embed_model,
            input=texts,
            # Always pass dimensions= so the actual output matches
            # whatever openai_embed_dim is configured to.
            # This was the root of B-02: without this the API silently
            # returned 1536-dim vectors into a 3072-dim collection.
            dimensions=settings.openai_embed_dim,
        )
    except Exception as exc:
        logger.error("OpenAI embedding failed: %s", exc)
        raise

    embeddings = [item.embedding for item in response.data]
    logger.info("Embedding complete — received %d vector(s)", len(embeddings))
    return embeddings
