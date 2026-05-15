"""Application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings."""

    app_name: str = "Production RAG System"
    app_env: str = "development"
    debug: bool = False

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    frontend_port: int = 8501

    chroma_path: Path = Path("chroma_store")
    upload_dir: Path = Path("data/uploads")
    log_level: str = "INFO"

    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    llm_provider: str = "openrouter"
    llm_model: str = "openai/gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    llm_timeout_seconds: int = 60
    llm_max_retries: int = 3
    llm_retry_backoff_seconds: float = 1.5

    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1/chat/completions"
    openrouter_primary_model: str = "openai/gpt-4o-mini"
    openrouter_fallback_models: List[str] = Field(
        default_factory=lambda: [
            "anthropic/claude-3-haiku",
            "meta-llama/llama-3-70b-instruct",
        ]
    )
    openrouter_http_referer: str = "http://localhost:8000"
    openrouter_app_title: str = "Production RAG System"

    chunk_size: int = 600
    chunk_overlap: int = 100
    retrieval_top_k: int = 5
    hybrid_alpha: float = 0.65
    max_upload_size_mb: int = 25
    allowed_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".txt", ".md", ".markdown", ".docx"]
    )
    api_rate_limit: str = "30/minute"
    enable_reranking: bool = True
    enable_query_rewrite: bool = True
    semantic_cache_ttl_seconds: int = 3600
    max_context_characters: int = 12000

    openai_compatible_headers_json: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    @field_validator("openrouter_fallback_models", mode="before")
    @classmethod
    def parse_fallback_models(cls, value):
        """Accept comma-separated fallback models from env."""

        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings."""

    settings = Settings()
    settings.chroma_path.mkdir(parents=True, exist_ok=True)
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    return settings
