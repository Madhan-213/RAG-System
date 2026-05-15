"""Recursive chunking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass
class Chunk:
    """Chunked text plus source metadata."""

    text: str
    page_number: int | None = None


class RecursiveTextChunker:
    """A lightweight recursive chunker using token-like word counts."""

    def __init__(self, chunk_size: int = 600, chunk_overlap: int = 100) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators: Sequence[str] = ("\n\n", "\n", ". ", " ", "")

    def split_text(self, text: str) -> List[str]:
        """Split text recursively into semantically coherent chunks."""

        return [chunk for chunk in self._split(text, 0) if chunk.strip()]

    def split_pages(self, pages: Iterable[Chunk]) -> List[Chunk]:
        """Split extracted pages while preserving source page numbers."""

        chunks: List[Chunk] = []
        for page in pages:
            for text in self.split_text(page.text):
                chunks.append(Chunk(text=text, page_number=page.page_number))
        return chunks

    def _split(self, text: str, separator_index: int) -> List[str]:
        if self._token_length(text) <= self.chunk_size or separator_index >= len(self.separators):
            return self._merge_with_overlap([text])

        separator = self.separators[separator_index]
        parts = list(text) if separator == "" else text.split(separator)
        if len(parts) == 1:
            return self._split(text, separator_index + 1)

        sub_chunks: List[str] = []
        buffer = ""
        joiner = "" if separator == "" else separator
        for part in parts:
            candidate = f"{buffer}{joiner if buffer else ''}{part}".strip()
            if candidate and self._token_length(candidate) > self.chunk_size:
                if buffer:
                    sub_chunks.extend(self._split(buffer, separator_index + 1))
                buffer = part
            else:
                buffer = candidate

        if buffer:
            sub_chunks.extend(self._split(buffer, separator_index + 1))

        return self._merge_with_overlap(sub_chunks)

    def _merge_with_overlap(self, chunks: List[str]) -> List[str]:
        merged: List[str] = []
        for chunk in chunks:
            chunk = chunk.strip()
            if not chunk:
                continue
            if not merged:
                merged.append(chunk)
                continue
            if self._token_length(chunk) <= self.chunk_size and self._token_length(merged[-1]) + self._token_length(chunk) < self.chunk_size:
                merged[-1] = f"{merged[-1]}\n{chunk}".strip()
            else:
                overlap_tokens = merged[-1].split()[-self.chunk_overlap :] if self.chunk_overlap else []
                prefix = " ".join(overlap_tokens)
                merged.append(f"{prefix} {chunk}".strip() if prefix else chunk)
        return merged

    @staticmethod
    def _token_length(text: str) -> int:
        return len(text.split())
