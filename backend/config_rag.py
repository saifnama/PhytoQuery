"""
RAG Configuration
=================
Retrieval-Augmented Generation settings and providers.
"""

import os

# =============================================================================
# OPENROUTER (Primary for RAG)
# =============================================================================

RAG_OPENROUTER_API_KEY = os.getenv("RAG_OPENROUTER_API_KEY", "").strip()
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
# Accelerator override: "cuda" | "mps" | "cpu".  When unset the backend auto-detects
# the best device (cuda > mps > cpu).
RAG_DEVICE = os.getenv("RAG_DEVICE", "")
# Enterprise: shard embedding/reranker models across all visible CUDA GPUs via
# HuggingFace device_map="auto".  Only applies when the detected device is cuda.
RAG_MULTI_GPU = os.getenv("RAG_MULTI_GPU", "false").strip().lower() in {"1", "true", "yes"}
# Use Flash Attention 2 on CUDA when available (requires flash-attn package).
# Falls back to eager attention if the package is missing.
RAG_USE_FLASH_ATTENTION = os.getenv("RAG_USE_FLASH_ATTENTION", "true").strip().lower() not in {"0", "false", "no"}


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
