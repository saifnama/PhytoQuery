from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Request, Response, BackgroundTasks
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import shutil
import uuid
import asyncio
from datetime import datetime, timezone
from backend.schemas.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    UploadJobStatus,
    IndexedFileInfo,
)
from backend.services.rag_engine import RAGService, rag_service
from backend.services.rag_engine import RAGLLMTimeoutError, RAGProviderAuthError
from backend.core.session import attach_session_cookie, get_or_set_session_id
from backend.core.rag_storage import get_user_upload_file_path
from backend.core.upload_jobs import UploadJobStore
from backend.core.user_locks import user_lock_manager
import logging


# Dependency provider
def get_rag_service() -> RAGService:
    return rag_service


router = APIRouter(prefix="/api/chat", tags=["Chat"])
logger = logging.getLogger(__name__)

job_store = UploadJobStore()


async def _process_upload_job(job_id: str, saved_paths: List[str], parser_type: str, user_id: str):
    """Background task: process and index uploaded PDFs without blocking the HTTP response."""
    service = get_rag_service()
    try:
        async with user_lock_manager.lock(user_id):
            # Run CPU-bound embedding generation in a thread pool so the event loop stays responsive
            indexed_files, _ = await asyncio.to_thread(
                service.process_and_index_pdfs_with_texts,
                saved_paths,
                parser_type=parser_type,
                user_id=user_id,
            )

            job_store.update(job_id, {
                "status": "completed",
                "message": f"Successfully indexed {len(indexed_files)} files using {parser_type}",
                "files": indexed_files,
                "summaries": None,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            })
        logger.info(f"Upload job {job_id} completed: {len(indexed_files)} files indexed")
    except Exception as e:
        logger.error(f"Upload job {job_id} failed: {e}")
        job_store.update(job_id, {
            "status": "failed",
            "message": f"Indexing failed: {str(e)}",
            "error": str(e),
            "completed_at": datetime.now(timezone.utc).isoformat(),
        })


@router.post("/upload/json", response_model=UploadResponse)
async def upload_pdfs_json(
    request: Request,
    response: Response,
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    service: RAGService = Depends(get_rag_service),
    parser_type: Optional[str] = Form("pymupdf"),
):
    """Upload PDFs for indexing. Returns immediately; processing continues in background.

    Args:
        parser_type: "pymupdf" for fast extraction, "docling" for detailed
    """
    user_id = get_or_set_session_id(request, response)

    # Validate parser_type
    if parser_type not in ("pymupdf", "docling"):
        parser_type = "docling"

    saved_paths = []
    async with user_lock_manager.lock(user_id):
        for file in files:
            if not file.filename.lower().endswith(".pdf"):
                continue

            file_path = get_user_upload_file_path(user_id, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            saved_paths.append(str(file_path))

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded")

    job_id = str(uuid.uuid4())
    filenames = [os.path.basename(p) for p in saved_paths]
    job_store.create({
        "job_id": job_id,
        "user_id": user_id,
        "status": "processing",
        "message": f"Processing {len(saved_paths)} file(s) with {parser_type}...",
        "files": filenames,
        "parser_type": parser_type,
        "summaries": None,
        "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
    })

    background_tasks.add_task(_process_upload_job, job_id, saved_paths, parser_type, user_id)
    logger.info(f"Upload job {job_id} queued for user {user_id}: {len(saved_paths)} file(s)")

    return UploadResponse(
        status="processing",
        message=f"Processing {len(saved_paths)} file(s) with {parser_type}. Poll /api/chat/upload/status/{job_id} for updates.",
        files=filenames,
        summaries=None,
        job_id=job_id,
    )


@router.get("/upload/status/{job_id}", response_model=UploadJobStatus)
async def get_upload_status(request: Request, response: Response, job_id: str):
    """Get the status of an async upload job."""
    user_id = get_or_set_session_id(request, response)
    job = job_store.get(job_id)
    if not job or job.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Job not found")
    return UploadJobStatus(**job)


@router.get("/upload/jobs")
async def list_upload_jobs(request: Request, response: Response):
    """List active upload jobs for the current user only."""
    user_id = get_or_set_session_id(request, response)
    return job_store.list_for_user(user_id)


@router.get("/files/json", response_model=List[IndexedFileInfo])
async def list_indexed_files(
    request: Request,
    response: Response,
    service: RAGService = Depends(get_rag_service),
):
    """List all documents currently indexed in the RAG vector store for the user."""
    user_id = get_or_set_session_id(request, response)
    return service.list_indexed_files(user_id)


@router.get("/files/{filename}/content")
async def get_uploaded_file_content(
    request: Request,
    response: Response,
    filename: str,
):
    """Return the uploaded PDF file for inline viewing for the current user."""
    user_id = get_or_set_session_id(request, response)
    safe_filename = os.path.basename(filename)
    if safe_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = get_user_upload_file_path(user_id, safe_filename)
    if not file_path.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    file_response = FileResponse(
        str(file_path),
        media_type="application/pdf",
        filename=safe_filename,
        content_disposition_type="inline",
    )
    attach_session_cookie(file_response, request, user_id)
    return file_response


@router.delete("/files/{filename}")
async def delete_source(
    request: Request,
    response: Response,
    filename: str,
    service: RAGService = Depends(get_rag_service),
):
    """Remove a source completely: chunks from ChromaDB."""
    user_id = get_or_set_session_id(request, response)
    async with user_lock_manager.lock(user_id):
        success = service.delete_source(filename, user_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete '{filename}'")
    return {
        "status": "success",
        "message": f"Deleted '{filename}' and all its embeddings.",
    }


@router.post("/reset")
async def reset_rag_data(
    request: Request,
    response: Response,
    service: RAGService = Depends(get_rag_service),
):
    """Permanently delete all indexed chunks for the user."""
    user_id = get_or_set_session_id(request, response)
    async with user_lock_manager.lock(user_id):
        success = service.reset_rag(user_id)
        if success:
            job_store.delete_user_jobs(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset RAG data.")
    return {
        "status": "success",
        "message": "All chat history and sources permanently deleted for user.",
    }


@router.post("/cleanup")
async def cleanup_user_data(
    request: Request,
    response: Response,
    service: RAGService = Depends(get_rag_service),
):
    """Clean up all data for a user when they close their browser.
    Deletes: ChromaDB, uploads, and all user files."""
    user_id = get_or_set_session_id(request, response)
    async with user_lock_manager.lock(user_id):
        success = service.cleanup_user(user_id)
        if success:
            job_store.delete_user_jobs(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cleanup user data.")
    return {"status": "success", "message": "All user data cleaned up."}


@router.post("/query/json", response_model=QueryResponse)
async def query_rag_json(
    http_request: Request,
    response: Response,
    payload: QueryRequest,
    service: RAGService = Depends(get_rag_service),
):
    user_id = get_or_set_session_id(http_request, response)
    try:
        # Convert chat_history from Pydantic models to dicts
        history = None
        if payload.chat_history:
            history = [{"role": m.role, "content": m.content} for m in payload.chat_history]

        result = await service.query(
            payload.query,
            filter_files=payload.selected_files,
            user_id=user_id,
            chat_history=history,
        )
        return QueryResponse(**result)
    except RAGProviderAuthError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except RAGLLMTimeoutError as e:
        raise HTTPException(status_code=504, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
