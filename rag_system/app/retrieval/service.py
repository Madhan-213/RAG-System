"""End-to-end retrieval orchestration."""

from __future__ import annotations

from typing import List

from app.config import get_settings
from app.embeddings.service import get_embedding_service, get_reranker_service
from app.prompting.templates import build_grounded_messages
from app.retrieval.cache import SemanticCache
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.llm_service import get_llm_service
from app.retrieval.memory import ConversationMemory
from app.retrieval.vector_store import ChromaVectorStore
from app.schemas import DocumentChunk, QueryRequest, QueryResponse, SourceCitation
from app.utils.security import safe_prompt_text


class RetrievalService:
    """Coordinates query rewriting, retrieval, reranking, and answering."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.vector_store = ChromaVectorStore()
        self.embedding_service = get_embedding_service()
        self.reranker = get_reranker_service() if self.settings.enable_reranking else None
        self.llm = get_llm_service()
        self.memory = ConversationMemory()
        self.cache = SemanticCache()

    async def retrieve(self, request: QueryRequest) -> tuple[str | None, List[DocumentChunk], str]:
        """Retrieve relevant chunks for a query."""

        conversation_id = self.memory.ensure_conversation(request.conversation_id)
        history = self.memory.render_history(conversation_id)
        rewritten_query = None
        search_query = request.question
        if self.settings.enable_query_rewrite and history:
            rewritten_query = await self.llm.rewrite_query(request.question, history)
            search_query = rewritten_query or request.question

        query_embedding = self.embedding_service.embed_query(search_query)
        vector_results = self.vector_store.similarity_search(
            query_embedding=query_embedding,
            top_k=max(request.top_k * 2, request.top_k),
            filters=request.filters,
        )

        results = vector_results
        if request.use_hybrid:
            lexical_pool = self.vector_store.get_all_chunks(request.filters)
            bm25_results = HybridRetriever.bm25_search(search_query, lexical_pool, request.top_k * 2)
            results = HybridRetriever.combine_results(
                vector_results=vector_results,
                bm25_results=bm25_results,
                alpha=self.settings.hybrid_alpha,
                top_k=request.top_k * 2,
            )

        if request.use_reranking and self.reranker and results:
            rerank_scores = self.reranker.rerank(search_query, [chunk.text for chunk in results])
            reranked = []
            for chunk, score in zip(results, rerank_scores):
                reranked.append(chunk.model_copy(update={"score": float(score)}))
            results = sorted(reranked, key=lambda item: item.score or 0.0, reverse=True)

        return rewritten_query, results[: request.top_k], conversation_id

    async def answer_query(self, request: QueryRequest) -> QueryResponse:
        """Return a grounded answer with citations."""

        cached = self.cache.get(request.question, request.filters)
        if cached and not request.conversation_id:
            return cached

        rewritten_query, chunks, conversation_id = await self.retrieve(request)
        if not chunks:
            response = QueryResponse(
                answer="I don't have information about that in the provided documents.",
                confidence=0.0,
                rewritten_query=rewritten_query,
                retrieved_chunks=[],
                citations=[],
                conversation_id=conversation_id,
            )
            self.cache.set(request.question, request.filters, response)
            return response

        trimmed_chunks = self._trim_context(chunks)
        messages = build_grounded_messages(safe_prompt_text(request.question), trimmed_chunks)
        answer = await self.llm.answer(messages)
        confidence = sum(chunk.score or 0.0 for chunk in trimmed_chunks) / max(len(trimmed_chunks), 1)
        citations = [
            SourceCitation(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                filename=chunk.metadata.get("filename", "unknown"),
                page_number=chunk.metadata.get("page_number"),
                score=chunk.score,
                snippet=chunk.text[:240],
            )
            for chunk in trimmed_chunks
        ]

        response = QueryResponse(
            answer=answer,
            confidence=round(float(confidence), 4),
            rewritten_query=rewritten_query,
            citations=citations,
            retrieved_chunks=trimmed_chunks,
            conversation_id=conversation_id,
        )
        self.memory.add_turn(conversation_id, "user", request.question)
        self.memory.add_turn(conversation_id, "assistant", answer)
        self.cache.set(request.question, request.filters, response)
        return response

    async def stream_query(self, request: QueryRequest):
        """Yield answer tokens for a streaming response."""

        rewritten_query, chunks, conversation_id = await self.retrieve(request)
        if not chunks:
            yield "I don't have information about that in the provided documents."
            return

        trimmed_chunks = self._trim_context(chunks)
        messages = build_grounded_messages(safe_prompt_text(request.question), trimmed_chunks)
        answer_parts: List[str] = []
        async for token in self.llm.stream_answer(messages):
            answer_parts.append(token)
            yield token

        answer = "".join(answer_parts)
        self.memory.add_turn(conversation_id, "user", request.question)
        self.memory.add_turn(conversation_id, "assistant", answer)

    def _trim_context(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        trimmed: List[DocumentChunk] = []
        budget = 0
        for chunk in chunks:
            if budget + len(chunk.text) > self.settings.max_context_characters and trimmed:
                break
            trimmed.append(chunk)
            budget += len(chunk.text)
        return trimmed
