"""
Local embedding using sentence-transformers/all-MiniLM-L6-v2.

The model is loaded once and reused — first call triggers a ~90MB download
to the HuggingFace cache (~/.cache/huggingface).
"""

from app.config.settings import get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        settings = get_settings()
        logger.info("Loading local embed model: %s", settings.local_embed_model)
        _model = SentenceTransformer(settings.local_embed_model)
        logger.info("Local embed model loaded (dim=%d).", settings.local_embed_dim)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings using the local sentence-transformers model."""
    if not texts:
        return []

    model = _get_model()
    logger.info("Embedding %d text(s) with local model.", len(texts))
    vectors = model.encode(texts, show_progress_bar=False).tolist()
    logger.info("Local embedding complete — %d vector(s).", len(vectors))
    return vectors
