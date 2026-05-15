"""Embedding and reranking services."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, List

from sentence_transformers import CrossEncoder, SentenceTransformer

from app.config import get_settings


class EmbeddingService:
    """Reusable sentence-transformer embedding service."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model = SentenceTransformer(model_name or settings.embedding_model_name)

    def embed_documents(self, texts: Iterable[str]) -> List[List[float]]:
        """Batch embed documents."""

        return self.model.encode(
            list(texts),
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a single query."""

        return self.embed_documents([text])[0]


class RerankerService:
    """Optional cross-encoder reranker."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model = CrossEncoder(model_name or settings.reranker_model_name)

    def rerank(self, query: str, passages: List[str]) -> List[float]:
        """Return relevance scores for passages."""

        if not passages:
            return []
        return list(self.model.predict([(query, passage) for passage in passages]).tolist())


@lru_cache(maxsize=1)
def get_embedding_service() -> EmbeddingService:
    """Return cached embedding service."""

    return EmbeddingService()


@lru_cache(maxsize=1)
def get_reranker_service() -> RerankerService:
    """Return cached reranker service."""

    return RerankerService()
