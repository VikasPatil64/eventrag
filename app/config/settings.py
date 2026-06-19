"""
Centralised application settings loaded from environment / .env file.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Inngest ─────────────────────────────────────────────────────────────
    inngest_app_id: str = "rag-app"
    inngest_dev_server_url: str = "http://127.0.0.1:8288"

    # ── Provider selection ──────────────────────────────────────────────────
    # embed_provider: "openai" | "local"
    # llm_provider:   "openai" | "gemini"
    embed_provider: str = "local"
    llm_provider: str = "gemini"

    # ── OpenAI ──────────────────────────────────────────────────────────────
    openai_api_key: str = ""
    openai_embed_model: str = "text-embedding-3-large"
    openai_embed_dim: int = 1536
    openai_llm_model: str = "gpt-4o-mini"

    # ── Local embeddings (sentence-transformers) ────────────────────────────
    local_embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    local_embed_dim: int = 384

    # ── Gemini ──────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_llm_model: str = "gemini-2.5-flash"

    # ── Qdrant ──────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "docs"
    qdrant_score_threshold: float = 0.3

    # ── Upload ──────────────────────────────────────────────────────────────
    upload_dir: str = "uploads"

    # ── Chunking ────────────────────────────────────────────────────────────
    chunk_size: int = 1024
    chunk_overlap: int = 200
    default_top_k: int = 5

    @property
    def active_embed_dim(self) -> int:
        """Return the embedding dimension for the active provider."""
        return self.local_embed_dim if self.embed_provider == "local" else self.openai_embed_dim

    @property
    def active_embed_model(self) -> str:
        """Return the embedding model name for the active provider."""
        return self.local_embed_model if self.embed_provider == "local" else self.openai_embed_model

    @property
    def active_llm_model(self) -> str:
        """Return the LLM model name for the active provider."""
        return self.gemini_llm_model if self.llm_provider == "gemini" else self.openai_llm_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the singleton Settings instance (cached after first call)."""
    return Settings()
