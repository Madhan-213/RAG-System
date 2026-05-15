"""Shared API dependencies."""

from functools import lru_cache

from app.ingestion.service import DocumentIngestionService
from app.retrieval.service import RetrievalService


@lru_cache(maxsize=1)
def get_ingestion_service() -> DocumentIngestionService:
    """Return cached ingestion service."""

    return DocumentIngestionService()


@lru_cache(maxsize=1)
def get_retrieval_service() -> RetrievalService:
    """Return cached retrieval service."""

    return RetrievalService()
