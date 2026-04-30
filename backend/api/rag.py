from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Request, Response
from fastapi.responses import FileResponse
from typing import List, Optional
import os
import shutil
from backend.schemas.schemas import (
    QueryRequest,
    QueryResponse,
    UploadResponse,
    IndexedFileInfo,
)
from backend.services.rag_engine import RAGService, rag_service
from backend.services.rag_engine import RAGProviderAuthError
from backend.core.session import attach_session_cookie, get_or_set_session_id
import logging


# Dependency provider
def get_rag_service() -> RAGService:
    return rag_service


router = APIRouter(prefix="/api/chat", tags=["Chat"])
logger = logging.getLogger(__name__)

UPLOAD_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "uploads",
)
os.makedirs(UPLOAD_DIR, exist_ok=True)
# --- JSON Endpoints ---


@router.post("/upload/json", response_model=UploadResponse)
async def upload_pdfs_json(
    request: Request,
    response: Response,
    files: List[UploadFile] = File(...),
    service: RAGService = Depends(get_rag_service),
    parser_type: Optional[str] = Form("pymupdf"),
):
    """Upload PDFs for indexing for a specific user.

    Args:
        parser_type: "pymupdf" for fast extraction, "docling" for detailed
    """
    user_id = get_or_set_session_id(request, response)

    # Validate parser_type
    if parser_type not in ("pymupdf", "docling"):
        parser_type = "docling"

    saved_paths = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        # User-specific upload directory
        user_upload_dir = os.path.join(UPLOAD_DIR, user_id)
        os.makedirs(user_upload_dir, exist_ok=True)

        file_path = os.path.join(user_upload_dir, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(file_path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded")

    try:
        indexed_files = service.process_and_index_pdfs(
            saved_paths, parser_type=parser_type, user_id=user_id
        )

        # Generate summaries for each uploaded file
        summaries = {}
        for path in saved_paths:
            filename = os.path.basename(path)
            extracted_text = getattr(service, '_last_extracted_text', '')
            if extracted_text:
                try:
                    summary = await service.summarize_document(extracted_text, filename)
                    if summary:
                        summaries[filename] = summary
                except Exception as e:
                    logger.warning(f"Summary generation failed for {filename}: {e}")

        return UploadResponse(
            status="success",
            message=f"Successfully indexed {len(indexed_files)} files using {parser_type}",
            files=indexed_files,
            summaries=summaries if summaries else None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


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

    file_path = os.path.join(UPLOAD_DIR, user_id, safe_filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    file_response = FileResponse(
        file_path,
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
    success = service.reset_rag(user_id)
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
    success = service.cleanup_user(user_id)
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
