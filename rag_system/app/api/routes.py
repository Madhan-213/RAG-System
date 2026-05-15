"""FastAPI routes for the RAG system."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_ingestion_service, get_retrieval_service
from app.api.limiter import limiter
from app.embeddings.service import get_embedding_service
from app.ingestion.service import DocumentIngestionService
from app.retrieval.service import RetrievalService
from app.schemas import QueryRequest, QueryResponse, UploadResponse
from app.utils.helpers import utc_now
from app.utils.security import validate_file_size, validate_upload

router = APIRouter()


@router.post("/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    file: UploadFile = File(...),
    ingestion_service: DocumentIngestionService = Depends(get_ingestion_service),
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
) -> UploadResponse:
    """Upload, ingest, embed, and index a document."""

    validate_upload(file)
    raw_bytes = await file.read()
    validate_file_size(len(raw_bytes))

    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    from app.config import get_settings

    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()
    destination = settings.upload_dir / f"{utc_now().strftime('%Y%m%d%H%M%S')}_{Path(file.filename or 'file').name}"
    destination.write_bytes(raw_bytes)

    document, chunks = ingestion_service.ingest(destination)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No readable text was extracted from the document.",
        )

    chunk_texts = [chunk.text for chunk in chunks]
    chunk_metadata = [
        {
            "page_number": chunk.page_number,
            "file_type": suffix,
            "upload_date": document.upload_date.isoformat(),
        }
        for chunk in chunks
    ]
    embeddings = get_embedding_service().embed_documents(chunk_texts)
    chunk_ids = retrieval_service.vector_store.add_documents(
        document=document,
        chunks=chunk_texts,
        embeddings=embeddings,
        chunk_metadata=chunk_metadata,
    )
    return UploadResponse(document=document, chunk_ids=chunk_ids)


@router.post("/query", response_model=QueryResponse)
@limiter.limit("30/minute")
async def query_documents(
    request: Request,
    query_request: QueryRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Answer a grounded question from indexed documents."""

    if query_request.stream:
        stream = retrieval_service.stream_query(query_request)
        return StreamingResponse(stream, media_type="text/plain")
    return await retrieval_service.answer_query(query_request)


@router.get("/documents")
@limiter.limit("30/minute")
async def list_documents(
    request: Request,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """List indexed documents."""

    return retrieval_service.vector_store.list_documents()


@router.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
@limiter.limit("10/minute")
async def delete_document(
    request: Request,
    document_id: str,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    """Delete a document and its chunks."""

    retrieval_service.vector_store.delete_documents(document_id)
