"""12-factor configuration: everything comes from environment variables.

Reads `.env` in development; in Docker/Kubernetes the same variables
arrive from the container environment (ConfigMap/Secret). Nothing is
hardcoded, so one image runs everywhere.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Service
    api_key: str = "dev-key"
    instance_name: str = "api-local"

    # Anthropic — the SDK reads ANTHROPIC_API_KEY itself; we only pick the model.
    # (validation_alias bypasses the RAG_ prefix for the key, standard name wins)
    anthropic_api_key: str = Field(default="", validation_alias="ANTHROPIC_API_KEY")
    anthropic_model: str = "claude-opus-5"

    # Retrieval
    fetch_k: int = 20  # candidates from hybrid search
    top_k: int = 5  # kept after reranking

    # Local models (fastembed)
    dense_model: str = "BAAI/bge-small-en-v1.5"
    sparse_model: str = "Qdrant/bm25"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"

    # Where uploaded files are kept
    uploads_dir: str = "./data/uploads"

    # Vector store: empty URL = embedded local-file mode
    qdrant_url: str = ""
    qdrant_path: str = "./data/qdrant"
    collection: str = "documents"


@lru_cache
def get_settings() -> Settings:
    return Settings()
