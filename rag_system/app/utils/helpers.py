"""General helper utilities."""

from __future__ import annotations

import hashlib
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""

    return datetime.now(timezone.utc)


def safe_document_id(filename: str) -> str:
    """Build a stable document id with a random suffix."""

    stem = re.sub(r"[^a-zA-Z0-9_-]+", "-", Path(filename).stem).strip("-").lower()
    return f"{stem or 'document'}-{uuid.uuid4().hex[:8]}"


def chunk_id(document_id: str, index: int) -> str:
    """Return deterministic chunk id for a document."""

    return f"{document_id}-chunk-{index:04d}"


def text_hash(value: str) -> str:
    """Hash text for caching and deduplication."""

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def normalize_whitespace(text: str) -> str:
    """Normalize repeated whitespace while preserving paragraph intent."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
