"""Semantic cache helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from app.config import get_settings
from app.schemas import QueryResponse
from app.utils.helpers import text_hash


class SemanticCache:
    """Tiny in-memory cache keyed by normalized query and filters."""

    def __init__(self) -> None:
        self.ttl_seconds = get_settings().semantic_cache_ttl_seconds
        self._store: Dict[str, tuple[datetime, QueryResponse]] = {}

    def build_key(self, question: str, filters: dict) -> str:
        return text_hash(f"{question.lower().strip()}::{sorted(filters.items())}")

    def get(self, question: str, filters: dict) -> Optional[QueryResponse]:
        key = self.build_key(question, filters)
        payload = self._store.get(key)
        if not payload:
            return None
        timestamp, response = payload
        if datetime.now(timezone.utc) - timestamp > timedelta(seconds=self.ttl_seconds):
            self._store.pop(key, None)
            return None
        return response

    def set(self, question: str, filters: dict, response: QueryResponse) -> None:
        key = self.build_key(question, filters)
        self._store[key] = (datetime.now(timezone.utc), response)
