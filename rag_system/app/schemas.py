"""Pydantic request and response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Single chunk stored and returned by the system."""

    id: str
    document_id: str
    text: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StoredDocument(BaseModel):
    """Document metadata visible through the API."""

    id: str
    filename: str
    file_type: str
    upload_date: datetime
    chunk_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(BaseModel):
    """User query payload."""

    question: str
    top_k: int = Field(default=5, ge=1, le=20)
    use_hybrid: bool = True
    use_reranking: bool = True
    filters: Dict[str, Any] = Field(default_factory=dict)
    conversation_id: Optional[str] = None
    stream: bool = False


class SourceCitation(BaseModel):
    """Source details returned to the frontend."""

    chunk_id: str
    document_id: str
    filename: str
    page_number: Optional[int] = None
    score: Optional[float] = None
    snippet: str


class QueryResponse(BaseModel):
    """Grounded answer payload."""

    answer: str
    confidence: float
    rewritten_query: Optional[str] = None
    citations: List[SourceCitation] = Field(default_factory=list)
    retrieved_chunks: List[DocumentChunk] = Field(default_factory=list)
    conversation_id: str


class UploadResponse(BaseModel):
    """Upload result payload."""

    document: StoredDocument
    chunk_ids: List[str]


class EvaluationSample(BaseModel):
    """Sample used for evaluation metrics."""

    question: str
    answer: str
    ground_truth: str
    retrieved_contexts: List[str]
    reference_contexts: List[str]


class EvaluationResult(BaseModel):
    """Aggregate metric scores."""

    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
