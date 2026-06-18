"""
Qdrant vector store wrapper.

FIX B-05: QdrantStorage is now a module-level singleton via get_vector_store().
          The original code called QdrantStorage() on every upsert and search,
          opening a new TCP connection each time — connection pool exhaustion
          under any concurrency.

FIX B-08: URL, collection, and dimensions are read from Settings (env-
          configurable) instead of being hardcoded.

FIX B-12: search() now accepts and applies a score_threshold so semantically
          irrelevant chunks are not returned to the LLM (prevents hallucination
          from off-topic context).

ADDED:    _ensure_collection() checks that an *existing* collection has the
          correct vector dimensions; if not, it recreates it with the right
          dimensions.  This prevents silent upsert failures when you change
          the embed model.

ADDED:    Payload index on source_id for fast per-document filtered search
          (needed for V2 multi-document support).
"""

from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from app.config.settings import Settings, get_settings
from app.utils.logging_config import get_logger

logger = get_logger(__name__)


class QdrantStorage:
    """Thin, logging-instrumented wrapper around QdrantClient."""

    def __init__(self, settings: Settings) -> None:
        self.collection = settings.qdrant_collection
        self.dim = settings.openai_embed_dim
        self.score_threshold = settings.qdrant_score_threshold

        logger.info("Connecting to Qdrant at %s", settings.qdrant_url)
        self.client = QdrantClient(url=settings.qdrant_url, timeout=30)
        self._ensure_collection()

    # ── Collection management ───────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        """Create the collection if absent; recreate it if the dim changed."""
        if self.client.collection_exists(self.collection):
            info = self.client.get_collection(self.collection)
            existing_dim = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing_dim != self.dim:
                logger.warning(
                    "Collection '%s' has dim=%d but settings require dim=%d. "
                    "Recreating — all existing data will be lost.",
                    self.collection,
                    existing_dim,
                    self.dim,
                )
                self.client.delete_collection(self.collection)
                self._create_collection()
            else:
                logger.debug(
                    "Collection '%s' exists with correct dim=%d — reusing.",
                    self.collection,
                    self.dim,
                )
        else:
            self._create_collection()

    def _create_collection(self) -> None:
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE),
        )
        # Index source_id so filtered search is O(log n) not O(n).
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="source_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        logger.info(
            "Created collection '%s' (dim=%d, metric=COSINE, "
            "payload_index=source_id)",
            self.collection,
            self.dim,
        )

    # ── Write ───────────────────────────────────────────────────────────────

    def upsert(
        self,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict],
    ) -> None:
        if not ids:
            logger.warning("upsert() called with empty ids — skipping.")
            return

        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info(
            "Upserted %d point(s) into collection '%s'.",
            len(points),
            self.collection,
        )

    # ── Read ────────────────────────────────────────────────────────────────

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        score_threshold: float | None = None,
    ) -> dict:
        """
        Return the top-k most similar chunks above *score_threshold*.

        FIX B-12: score_threshold prevents irrelevant chunks from reaching
        the LLM.  Defaults to the value in Settings (configurable via env).
        """
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            with_payload=True,
            limit=top_k,
            score_threshold=threshold,
        )

        contexts: list[str] = []
        sources: set[str] = set()

        for r in results:
            payload: dict = getattr(r, "payload", None) or {}
            text: str = payload.get("text", "")
            source: str = payload.get("source_id", "")
            if text:
                contexts.append(text)
                sources.add(source)

        logger.info(
            "Search returned %d context(s) (top_k=%d, threshold=%.2f).",
            len(contexts),
            top_k,
            threshold,
        )
        return {"contexts": contexts, "sources": list(sources)}


# ── Singleton accessor ──────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_vector_store() -> QdrantStorage:
    """
    Return the singleton QdrantStorage instance.

    FIX B-05: The original code called QdrantStorage() on every upsert and
    search request, creating a new QdrantClient (TCP connection) each time.
    lru_cache ensures the client is created exactly once.
    """
    return QdrantStorage(settings=get_settings())
