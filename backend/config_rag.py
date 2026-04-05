"""
RAG Configuration
==================
Retrieval-Augmented Generation settings and providers.
"""

import os

# =============================================================================
# OPENROUTER (Primary for RAG)
# =============================================================================

RAG_OPENROUTER_API_KEY = os.getenv("RAG_OPENROUTER_API_KEY", "")
RAG_OPENROUTER_MODEL = os.getenv("RAG_OPENROUTER_MODEL", "stepfun/step-3.5-flash:free")

# =============================================================================
# OLLAMA (Fallback for RAG)
# =============================================================================

RAG_OLLAMA_URL = os.getenv("RAG_OLLAMA_URL", "https://p.trycloudflare.com")
RAG_OLLAMA_MODEL = os.getenv("RAG_OLLAMA_MODEL", "llama3.1:8b")

# =============================================================================
# RAG SETTINGS
# =============================================================================

RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", "0.1"))
RAG_CONTEXT_WINDOW = int(os.getenv("RAG_CONTEXT_WINDOW", "8192"))
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "10"))
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.85"))


def get_rag_provider() -> dict:
    """RAG: OpenRouter first (better quality), then Ollama as fallback."""
    if RAG_OPENROUTER_API_KEY:
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
        raise ValueError(
            "No RAG provider configured. Set RAG_OPENROUTER_API_KEY or RAG_OLLAMA_URL"
        )
