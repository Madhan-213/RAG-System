"""Hybrid retrieval helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from rank_bm25 import BM25Okapi

from app.schemas import DocumentChunk


class HybridRetriever:
    """Combine BM25 and dense retrieval scores."""

    @staticmethod
    def bm25_search(query: str, chunks: List[DocumentChunk], top_k: int) -> List[DocumentChunk]:
        """Return lexical matches using BM25."""

        if not chunks:
            return []
        tokenized_corpus = [chunk.text.lower().split() for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(query.lower().split())
        ranked = sorted(
            zip(chunks, scores),
            key=lambda item: item[1],
            reverse=True,
        )[:top_k]
        results: List[DocumentChunk] = []
        for chunk, score in ranked:
            results.append(chunk.model_copy(update={"score": float(score)}))
        return results

    @staticmethod
    def combine_results(
        vector_results: List[DocumentChunk],
        bm25_results: List[DocumentChunk],
        alpha: float,
        top_k: int,
    ) -> List[DocumentChunk]:
        """Fuse dense and lexical scores."""

        aggregated: Dict[str, Dict[str, float | DocumentChunk]] = defaultdict(dict)
        for result in vector_results:
            aggregated[result.id]["chunk"] = result
            aggregated[result.id]["vector"] = float(result.score or 0.0)
            aggregated[result.id].setdefault("bm25", 0.0)
        for rank, result in enumerate(bm25_results, start=1):
            aggregated[result.id]["chunk"] = result
            aggregated[result.id].setdefault("vector", 0.0)
            aggregated[result.id]["bm25"] = 1.0 / rank

        fused: List[DocumentChunk] = []
        for payload in aggregated.values():
            score = alpha * float(payload.get("vector", 0.0)) + (1 - alpha) * float(payload.get("bm25", 0.0))
            chunk = payload["chunk"].model_copy(update={"score": score})
            fused.append(chunk)

        return sorted(fused, key=lambda item: item.score or 0.0, reverse=True)[:top_k]
