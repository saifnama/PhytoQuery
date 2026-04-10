from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import List, Optional
import os
import shutil
from backend.schemas.schemas import QueryRequest, QueryResponse, UploadResponse, IndexedFileInfo
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

# --- JSON Endpoints ---


@router.post("/upload/json", response_model=UploadResponse)
async def upload_pdfs_json(
    files: List[UploadFile] = File(...),
    service: RAGService = Depends(get_rag_service),
    parser_type: Optional[str] = Form("pymupdf"),
):
    """Upload PDFs for indexing.

    Args:
        parser_type: "pymupdf" for fast extraction, "docling" for detailed (default: "docling")
    """
    # Validate parser_type
    if parser_type not in ("pymupdf", "docling"):
        parser_type = "docling"

    saved_paths = []
    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        saved_paths.append(file_path)

    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid PDF files uploaded")

    try:
        indexed_files = service.process_and_index_pdfs(
            saved_paths, parser_type=parser_type
        )
        return UploadResponse(
            status="success",
            message=f"Successfully indexed {len(indexed_files)} files using {parser_type}",
            files=indexed_files,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.get("/files/json", response_model=List[IndexedFileInfo])
async def list_indexed_files(service: RAGService = Depends(get_rag_service)):
    """List all documents currently indexed in the RAG vector store."""
    return service.list_indexed_files()


@router.delete("/files/{filename}")
async def delete_source(filename: str, service: RAGService = Depends(get_rag_service)):
    """Remove a source completely: chunks from ChromaDB + PDF from disk."""
    success = service.delete_source(filename)
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to delete '{filename}'")
    return {"status": "success", "message": f"Deleted '{filename}' and all its embeddings."}


@router.post("/reset")
async def reset_rag_data(service: RAGService = Depends(get_rag_service)):
    """Permanently delete all indexed chunks and uploaded PDFs."""
    success = service.reset_rag()
    if not success:
        raise HTTPException(status_code=500, detail="Failed to reset RAG data.")
    return {"status": "success", "message": "All chat history and sources permanently deleted."}

@router.post("/query/json", response_model=QueryResponse)
async def query_rag_json(
    request: QueryRequest, service: RAGService = Depends(get_rag_service)
):
    try:
        result = await service.query(
            request.query, filter_files=request.selected_files
        )
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

