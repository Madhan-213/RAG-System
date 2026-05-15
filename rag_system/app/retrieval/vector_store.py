"""Chroma-backed vector store service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.schemas import DocumentChunk, StoredDocument
from app.utils.helpers import chunk_id


class ChromaVectorStore:
    """Persistent Chroma storage with a lightweight document catalog."""

    def __init__(self, collection_name: str = "rag_documents") -> None:
        settings = get_settings()
        self.catalog_path = settings.chroma_path / "documents.json"
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_documents(
        self,
        document: StoredDocument,
        chunks: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        chunk_metadata: Sequence[Dict[str, Any]],
    ) -> List[str]:
        """Store embedded chunks and persist document metadata."""

        ids = [chunk_id(document.id, index) for index, _ in enumerate(chunks)]
        metadatas = []
        for item in chunk_metadata:
            payload = {
                "document_id": document.id,
                "filename": document.filename,
                **item,
            }
            metadatas.append(payload)

        self.collection.add(
            ids=ids,
            documents=list(chunks),
            embeddings=list(embeddings),
            metadatas=metadatas,
        )
        self._upsert_document(document)
        return ids

    def similarity_search(
        self,
        query_embedding: Sequence[float],
        top_k: int = 5,
        filters: Dict[str, Any] | None = None,
    ) -> List[DocumentChunk]:
        """Run vector similarity search."""

        result = self.collection.query(
            query_embeddings=[list(query_embedding)],
            n_results=top_k,
            where=filters or None,
        )
        return self._to_chunks(result)

    def metadata_filtering(self, filters: Dict[str, Any], limit: int = 20) -> List[DocumentChunk]:
        """Get chunks matching metadata filters."""

        result = self.collection.get(where=filters, limit=limit)
        return self._to_chunks(result, from_get=True)

    def delete_documents(self, document_id: str) -> None:
        """Delete all chunks belonging to a document."""

        self.collection.delete(where={"document_id": document_id})
        catalog = self._load_catalog()
        catalog = [doc for doc in catalog if doc["id"] != document_id]
        self.catalog_path.write_text(json.dumps(catalog, indent=2), encoding="utf-8")

    def list_documents(self) -> List[StoredDocument]:
        """Return stored documents."""

        return [StoredDocument.model_validate(item) for item in self._load_catalog()]

    def get_all_chunks(self, filters: Dict[str, Any] | None = None) -> List[DocumentChunk]:
        """Return all chunks for lexical retrieval."""

        result = self.collection.get(where=filters or None, include=["documents", "metadatas"])
        return self._to_chunks(result, from_get=True)

    def _to_chunks(self, result: Dict[str, Any], from_get: bool = False) -> List[DocumentChunk]:
        docs = result.get("documents", [[]] if not from_get else [])
        metas = result.get("metadatas", [[]] if not from_get else [])
        ids = result.get("ids", [[]] if not from_get else [])
        distances = result.get("distances", [[]] if not from_get else [])

        if from_get:
            documents = docs or []
            metadatas = metas or []
            identifiers = ids or []
            distance_values: List[float | None] = [None] * len(identifiers)
        else:
            documents = docs[0] if docs else []
            metadatas = metas[0] if metas else []
            identifiers = ids[0] if ids else []
            distance_values = distances[0] if distances else []

        chunks: List[DocumentChunk] = []
        for index, text in enumerate(documents):
            metadata = metadatas[index] or {}
            distance = distance_values[index] if index < len(distance_values) else None
            score = None if distance is None else max(0.0, 1.0 - float(distance))
            chunks.append(
                DocumentChunk(
                    id=identifiers[index],
                    document_id=metadata.get("document_id", ""),
                    text=text,
                    score=score,
                    metadata=metadata,
                )
            )
        return chunks

    def _load_catalog(self) -> List[Dict[str, Any]]:
        if not self.catalog_path.exists():
            return []
        return json.loads(self.catalog_path.read_text(encoding="utf-8"))

    def _upsert_document(self, document: StoredDocument) -> None:
        catalog = self._load_catalog()
        filtered = [item for item in catalog if item["id"] != document.id]
        filtered.append(document.model_dump(mode="json"))
        self.catalog_path.write_text(json.dumps(filtered, indent=2), encoding="utf-8")
