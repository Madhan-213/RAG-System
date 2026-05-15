"""Prompt building utilities."""

from __future__ import annotations

from typing import Iterable, List

from app.schemas import DocumentChunk

SYSTEM_PROMPT = """You are a knowledgeable assistant. Answer ONLY using the provided context.
If the answer is unavailable, say:
'I don't have information about that in the provided documents.'"""


def build_context_block(chunks: Iterable[DocumentChunk]) -> str:
    """Build a formatted context block from retrieved chunks."""

    formatted: List[str] = []
    for index, chunk in enumerate(chunks, start=1):
        filename = chunk.metadata.get("filename", "unknown")
        page = chunk.metadata.get("page_number")
        location = f"{filename}, page {page}" if page else filename
        formatted.append(f"[Source {index} | {location}]\n{chunk.text}")
    return "\n\n".join(formatted)


def build_grounded_messages(question: str, chunks: Iterable[DocumentChunk]) -> list[dict[str, str]]:
    """Return OpenAI-compatible chat messages."""

    context = build_context_block(chunks)
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Provide a concise, grounded answer and cite the relevant source numbers."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
