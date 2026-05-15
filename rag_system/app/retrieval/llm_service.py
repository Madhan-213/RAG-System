"""OpenRouter-backed LLM service."""

from __future__ import annotations

import asyncio
import json
from functools import lru_cache
from typing import AsyncIterator

import httpx

from app.prompting.templates import SYSTEM_PROMPT
from app.retrieval.openrouter_config import OpenRouterConfig, get_openrouter_config


class LLMServiceError(Exception):
    """Base LLM service error."""


class InvalidAPIKeyError(LLMServiceError):
    """Raised when the OpenRouter key is invalid or missing."""


class RateLimitError(LLMServiceError):
    """Raised when rate limiting persists after retries."""


class OpenRouterLLMService:
    """Reusable OpenRouter chat completion client."""

    def __init__(self, config: OpenRouterConfig | None = None) -> None:
        self.config = config or get_openrouter_config()

    async def answer(self, messages: list[dict[str, str]], model: str | None = None) -> str:
        """Generate a non-streaming response with fallback model support."""

        payload = await self._post_with_fallback(messages=messages, model=model, stream=False)
        try:
            return payload["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMServiceError("OpenRouter returned an unexpected response format.") from exc

    async def stream_answer(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield streaming response chunks from OpenRouter."""

        last_error: Exception | None = None
        for candidate in self._model_candidates(model):
            try:
                async for token in self._stream_once(messages=messages, model=candidate):
                    yield token
                return
            except InvalidAPIKeyError:
                raise
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            raise last_error
        raise LLMServiceError("Unable to stream a response from OpenRouter.")

    async def rewrite_query(self, question: str, chat_history: str = "") -> str:
        """Rewrite a conversational question into a standalone search query."""

        prompt = (
            "Rewrite the question into a concise standalone search query. "
            "Return only the rewritten query.\n\n"
            f"Conversation:\n{chat_history}\n\nQuestion:\n{question}"
        )
        response = await self.answer(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ]
        )
        return response.strip().strip('"') or question

    async def _post_with_fallback(
        self,
        messages: list[dict[str, str]],
        model: str | None,
        stream: bool,
    ) -> dict:
        last_error: Exception | None = None
        for candidate in self._model_candidates(model):
            try:
                return await self._post_once(messages=messages, model=candidate, stream=stream)
            except InvalidAPIKeyError:
                raise
            except Exception as exc:
                last_error = exc
                continue

        if last_error:
            raise last_error
        raise LLMServiceError("Unable to get a response from OpenRouter.")

    async def _post_once(
        self,
        messages: list[dict[str, str]],
        model: str,
        stream: bool,
    ) -> dict:
        if not self.config.api_key:
            raise InvalidAPIKeyError(
                "OpenRouter API key is missing. Set OPENROUTER_API_KEY in your .env file."
            )

        headers = self._headers()
        payload = {"model": model, "messages": messages, "temperature": 0.1, "stream": stream}
        timeout = httpx.Timeout(self.config.timeout_seconds)

        for attempt in range(1, self.config.max_retries + 1):
            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(self.config.endpoint, headers=headers, json=payload)

            if response.status_code in {401, 403}:
                raise InvalidAPIKeyError(
                    "OpenRouter rejected the API key. Verify OPENROUTER_API_KEY and account access."
                )
            if response.status_code == 429:
                if attempt == self.config.max_retries:
                    raise RateLimitError("OpenRouter rate limit exceeded after retry attempts.")
                await asyncio.sleep(self.config.retry_backoff_seconds * attempt)
                continue
            if 500 <= response.status_code < 600:
                if attempt == self.config.max_retries:
                    raise LLMServiceError(
                        f"OpenRouter server error {response.status_code}: {response.text}"
                    )
                await asyncio.sleep(self.config.retry_backoff_seconds * attempt)
                continue
            if response.status_code >= 400:
                raise LLMServiceError(
                    f"OpenRouter request failed with status {response.status_code}: {response.text}"
                )

            return response.json()

        raise LLMServiceError("OpenRouter request failed unexpectedly.")

    async def _stream_once(
        self,
        messages: list[dict[str, str]],
        model: str,
    ) -> AsyncIterator[str]:
        if not self.config.api_key:
            raise InvalidAPIKeyError(
                "OpenRouter API key is missing. Set OPENROUTER_API_KEY in your .env file."
            )

        headers = self._headers()
        payload = {"model": model, "messages": messages, "temperature": 0.1, "stream": True}
        timeout = httpx.Timeout(self.config.timeout_seconds)

        for attempt in range(1, self.config.max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=timeout) as client:
                    async with client.stream(
                        "POST",
                        self.config.endpoint,
                        headers=headers,
                        json=payload,
                    ) as response:
                        if response.status_code in {401, 403}:
                            raise InvalidAPIKeyError(
                                "OpenRouter rejected the API key. Verify OPENROUTER_API_KEY and account access."
                            )
                        if response.status_code == 429:
                            raise RateLimitError("OpenRouter rate limit exceeded.")
                        if response.status_code >= 400:
                            raise LLMServiceError(
                                f"OpenRouter stream failed with status {response.status_code}: {await response.aread()}"
                            )

                        async for line in response.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                return
                            try:
                                payload = json.loads(data)
                                delta = payload["choices"][0]["delta"].get("content")
                            except (KeyError, IndexError, TypeError, json.JSONDecodeError):
                                delta = None
                            if delta:
                                yield delta
                        return
            except InvalidAPIKeyError:
                raise
            except RateLimitError:
                if attempt == self.config.max_retries:
                    raise
                await asyncio.sleep(self.config.retry_backoff_seconds * attempt)
            except Exception as exc:
                if attempt == self.config.max_retries:
                    raise LLMServiceError(f"OpenRouter streaming failed: {exc}") from exc
                await asyncio.sleep(self.config.retry_backoff_seconds * attempt)

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": self.config.http_referer,
            "X-Title": self.config.app_title,
        }

    def _model_candidates(self, model: str | None) -> list[str]:
        if model:
            return [model, *[item for item in self.config.fallback_models if item != model]]
        return self.config.model_candidates


@lru_cache(maxsize=1)
def get_llm_service() -> OpenRouterLLMService:
    """Return cached OpenRouter LLM service."""

    return OpenRouterLLMService()
