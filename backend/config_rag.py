"""
RAG Configuration
=================
Retrieval-Augmented Generation settings and providers.
"""

import os

# =============================================================================
# OPENROUTER (Primary for RAG)
# =============================================================================

RAG_OPENROUTER_API_KEY = os.getenv("RAG_OPENROUTER_API_KEY", "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").strip()
RAG_OPENROUTER_MODEL = os.getenv(
    "RAG_OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
)

# =============================================================================
# OLLAMA (Fallback for RAG)
# =============================================================================

RAG_OLLAMA_URL = os.getenv("RAG_OLLAMA_URL", "")
RAG_OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "llama3.1:8b")

# =============================================================================
# RAG SETTINGS
# =============================================================================

RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.1"))
RAG_CONTEXT_WINDOW = int(os.getenv("RAG_CONTEXT_WINDOW", "8192"))
# Primary embedding model (Qwen3-Embedding-4B recommended)
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-4B")
# Fallback embedding model if primary fails to load (e.g., OOM)
RAG_FALLBACK_EMBEDDING_MODEL = os.getenv("RAG_FALLBACK_EMBEDDING_MODEL", "BAAI/bge-m3")
# MRL output dimension for Qwen3 (None = full 2560 dims; e.g., 1024 or 512 for speed/memory)
RAG_EMBEDDING_DIM = os.getenv("RAG_EMBEDDING_DIM")
RAG_EMBEDDING_DIM = int(RAG_EMBEDDING_DIM) if RAG_EMBEDDING_DIM else None
# Query instruction for instruction-aware embedding models (Qwen3)
RAG_EMBEDDING_INSTRUCTION = os.getenv("RAG_EMBEDDING_INSTRUCTION", "")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.85"))


def get_rag_provider() -> dict:
    """RAG: OpenRouter first (better quality), then Ollama as fallback."""
    has_valid_openrouter_key = bool(RAG_OPENROUTER_API_KEY) and RAG_OPENROUTER_API_KEY not in {
        "sk-",
        "sk",
    }
    if has_valid_openrouter_key:
        return {
            "provider": "openrouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "model": RAG_OPENROUTER_MODEL,
            "api_key": RAG_OPENROUTER_API_KEY,
        }
    elif RAG_OLLAMA_URL:
        return {
            "provider": "ollama",
            "url": f"{RAG_OLLAMA_URL}/api/chat",
            "model": RAG_OLLAMA_MODEL,
        }
    else:
        return {
            "provider": "unconfigured",
            "url": "",
            "model": "",
            "api_key": "",
        }
