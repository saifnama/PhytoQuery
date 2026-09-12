"""Chat engine derived constants (split ex services/rag_engine.py)."""
import logging

from backend.src.settings import RAG_TEMPERATURE, _safe_float, _safe_int

logger = logging.getLogger(__name__)

LLM_TEMPERATURE = RAG_TEMPERATURE


RAG_QUERY_TIMEOUT_SECONDS = _safe_float("RAG_QUERY_TIMEOUT_SECONDS", 45.0)


RAG_SUMMARY_TIMEOUT_SECONDS = _safe_float("RAG_SUMMARY_TIMEOUT_SECONDS", 20.0)


RAG_RERANK_CANDIDATE_K = _safe_int("RAG_RERANK_CANDIDATE_K", 40)


RAG_RERANK_BATCH_SIZE = _safe_int("RAG_RERANK_BATCH_SIZE", 8)
