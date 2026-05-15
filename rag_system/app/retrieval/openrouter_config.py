"""OpenRouter configuration helpers."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import get_settings


@dataclass(frozen=True)
class OpenRouterConfig:
    """Resolved OpenRouter runtime configuration."""

    api_key: str
    endpoint: str
    primary_model: str
    fallback_models: list[str]
    timeout_seconds: int
    max_retries: int
    retry_backoff_seconds: float
    http_referer: str
    app_title: str

    @property
    def model_candidates(self) -> list[str]:
        """Return model attempts in priority order."""

        return [self.primary_model, *[model for model in self.fallback_models if model != self.primary_model]]


def get_openrouter_config() -> OpenRouterConfig:
    """Build OpenRouter config from application settings."""

    settings = get_settings()
    primary_model = settings.openrouter_primary_model or settings.llm_model
    return OpenRouterConfig(
        api_key=settings.openrouter_api_key or settings.llm_api_key,
        endpoint=settings.openrouter_base_url or settings.llm_base_url,
        primary_model=primary_model,
        fallback_models=settings.openrouter_fallback_models,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        retry_backoff_seconds=settings.llm_retry_backoff_seconds,
        http_referer=settings.openrouter_http_referer,
        app_title=settings.openrouter_app_title,
    )
