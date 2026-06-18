"""
Centralised application settings loaded from environment / .env file.

FIX B-01: Single source of truth for inngest_app_id — eliminates the
          "rag-app" vs "rag_app" mismatch between main.py and streamlit_app.py.
FIX B-02: openai_embed_dim=1536 matches the *actual* default output of
          text-embedding-3-large (the old code claimed 3072 but never passed
          dimensions= to the API, producing 1536-dim vectors into a
          3072-dim Qdrant collection → rejection on every upsert).
FIX B-08: qdrant_url is now env-configurable instead of hardcoded.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",          # silently ignore unknown keys in .env
    )

    # ── Inngest ────────────────────────────────────────────────────────────
    # MUST be identical in both main.py and streamlit_app.py.
    inngest_app_id: str = "rag-app"
    inngest_dev_server_url: str = "http://127.0.0.1:8288"

    # ── OpenAI ─────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    # text-embedding-3-large default output = 1536 dims (NOT 3072).
    # If you ever want 3072 you must ALSO pass dimensions=3072 to the API.
    openai_embed_model: str = "text-embedding-3-large"
    openai_embed_dim: int = 1536
    openai_llm_model: str = "gpt-4o-mini"

    # ── Qdrant ─────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docs"
    # Minimum cosine similarity to include a result (0.0 = no filter).
    # 0.3 is a safe starting point — raise to 0.5 for stricter retrieval.
    qdrant_score_threshold: float = 0.3

    # ── Upload ─────────────────────────────────────────────────────────────
    upload_dir: str = "uploads"

    # ── Chunking ───────────────────────────────────────────────────────────
    chunk_size: int = 1024
    chunk_overlap: int = 200
    default_top_k: int = 5


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
