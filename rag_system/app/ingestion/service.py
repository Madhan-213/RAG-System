"""Document ingestion service."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from app.chunking.recursive_chunker import Chunk, RecursiveTextChunker
from app.config import get_settings
from app.ingestion.loaders import get_loader
from app.schemas import StoredDocument
from app.utils.helpers import safe_document_id, utc_now


class DocumentIngestionService:
    """Convert raw files into chunkable document structures."""

    def __init__(self, chunker: RecursiveTextChunker | None = None) -> None:
        settings = get_settings()
        self.chunker = chunker or RecursiveTextChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )

    def ingest(self, file_path: Path) -> Tuple[StoredDocument, List[Chunk]]:
        """Load a file and split it into chunks."""

        loader = get_loader(file_path)
        pages = loader.load(file_path)
        document_id = safe_document_id(file_path.name)
        stored_document = StoredDocument(
            id=document_id,
            filename=file_path.name,
            file_type=file_path.suffix.lower(),
            upload_date=utc_now(),
            metadata={"path": str(file_path)},
        )
        chunks = self.chunker.split_pages(
            Chunk(text=page.text, page_number=page.page_number) for page in pages
        )
        stored_document.chunk_count = len(chunks)
        return stored_document, chunks
