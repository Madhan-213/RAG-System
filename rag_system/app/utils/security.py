"""Security and validation helpers."""

from __future__ import annotations

from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.config import get_settings


def validate_upload(upload: UploadFile) -> None:
    """Validate file extension and announced content length."""

    settings = get_settings()
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in settings.allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type: {suffix or 'unknown'}",
        )


def validate_file_size(size_bytes: int) -> None:
    """Validate file size limit."""

    settings = get_settings()
    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds {settings.max_upload_size_mb} MB limit.",
        )


def safe_prompt_text(value: str) -> str:
    """Basic prompt sanitation for control characters."""

    return value.replace("\x00", " ").strip()
