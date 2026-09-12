"""Shared FastAPI dependencies (Phase 3). Single home for Depends providers."""
from typing import Any

from backend.src.db.session import get_db

__all__ = ["get_db", "get_ner_service", "get_rag_service"]


def get_ner_service():
    from backend.src.ner.service import ner_service

    return ner_service


def get_rag_service() -> Any:
    from backend.src.chat.service import get_rag_service as _get_rag_service

    return _get_rag_service()
