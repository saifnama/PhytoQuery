from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import List
import os
import shutil
from backend.schemas.schemas import QueryRequest, QueryResponse, UploadResponse
from backend.services.rag_engine import RAGService, rag_service
import logging


# Dependency provider
def get_rag_service() -> RAGService:
    return rag_service


router = APIRouter(prefix="/rag", tags=["RAG"])
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
    files: List[UploadFile] = File(...), service: RAGService = Depends(get_rag_service)
):
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
        indexed_files = service.process_and_index_pdfs(saved_paths)
        return UploadResponse(
            status="success",
            message=f"Successfully indexed {len(indexed_files)} files",
            files=indexed_files,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/query/json", response_model=QueryResponse)
async def query_rag_json(
    request: QueryRequest, service: RAGService = Depends(get_rag_service)
):
    try:
        result = await service.query(request.query)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")
