from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, Header
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


def get_user_id(x_user_id: Optional[str] = Header(None)) -> str:
    """Extract user ID from header, default to 'default' if not provided."""
    if not x_user_id or not x_user_id.strip():
        return "default"
    return x_user_id.strip()


# --- JSON Endpoints ---


@router.post("/upload/json", response_model=UploadResponse)
async def upload_pdfs_json(
    files: List[UploadFile] = File(...),
    service: RAGService = Depends(get_rag_service),
    parser_type: Optional[str] = Form("pymupdf"),
    x_user_id: Optional[str] = Header(None),
):
    """Upload PDFs for indexing for a specific user.

    Args:
        parser_type: "pymupdf" for fast extraction, "docling" for detailed
    """
    user_id = get_user_id(x_user_id)

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
    service: RAGService = Depends(get_rag_service),
    x_user_id: Optional[str] = Header(None),
):
    """List all documents currently indexed in the RAG vector store for the user."""
    user_id = get_user_id(x_user_id)
    return service.list_indexed_files(user_id)


@router.get("/files/{filename}/content")
async def get_uploaded_file_content(
    filename: str,
    user_id: Optional[str] = None,
    x_user_id: Optional[str] = Header(None),
):
    """Return the uploaded PDF file for inline viewing for the current user."""
    user_id = get_user_id(x_user_id or user_id)
    safe_filename = os.path.basename(filename)
    if safe_filename != filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = os.path.join(UPLOAD_DIR, user_id, safe_filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        file_path,
        media_type="application/pdf",
        filename=safe_filename,
        content_disposition_type="inline",
    )


@router.delete("/files/{filename}")
async def delete_source(
    filename: str,
    service: RAGService = Depends(get_rag_service),
    x_user_id: Optional[str] = Header(None),
):
    """Remove a source completely: chunks from ChromaDB."""
    user_id = get_user_id(x_user_id)
    success = service.delete_source(filename, user_id)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete '{filename}'")
    return {
        "status": "success",
        "message": f"Deleted '{filename}' and all its embeddings.",
    }


@router.post("/reset")
async def reset_rag_data(
    service: RAGService = Depends(get_rag_service),
    x_user_id: Optional[str] = Header(None),
):
    """Permanently delete all indexed chunks for the user."""
    user_id = get_user_id(x_user_id)
    success = service.reset_rag(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset RAG data.")
    return {
        "status": "success",
        "message": "All chat history and sources permanently deleted for user.",
    }


@router.post("/cleanup")
async def cleanup_user_data(
    service: RAGService = Depends(get_rag_service),
    x_user_id: Optional[str] = Header(None),
):
    """Clean up all data for a user when they close their browser.
    Deletes: ChromaDB, uploads, and all user files."""
    user_id = get_user_id(x_user_id)
    success = service.cleanup_user(user_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to cleanup user data.")
    return {"status": "success", "message": "All user data cleaned up."}


@router.post("/query/json", response_model=QueryResponse)
async def query_rag_json(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service),
    x_user_id: Optional[str] = Header(None),
):
    user_id = get_user_id(x_user_id)
    try:
        # Convert chat_history from Pydantic models to dicts
        history = None
        if request.chat_history:
            history = [{"role": m.role, "content": m.content} for m in request.chat_history]

        result = await service.query(
            request.query,
            filter_files=request.selected_files,
            user_id=user_id,
            chat_history=history,
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
