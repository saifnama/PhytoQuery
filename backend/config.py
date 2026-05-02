"""
PhytoQuery — Unified Configuration
====================================
Usage:
    from backend.config import env, env_int, env_float, get_rag_provider, get_ner_provider
"""

import os
from pathlib import Path

# Auto-load .env file from project root (PhytoQuery/.env)
# override=False means real env vars (e.g. from Slurm) take priority over .env
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parent.parent
    load_dotenv(_project_root / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on system env vars


# ---------------------------------------------------------------------------
# Helpers — thin wrappers around os.getenv with type casting
# ---------------------------------------------------------------------------

def env(key: str, default: str = "") -> str:
    """Read a string from the environment."""
    return os.getenv(key, default).strip()


def env_int(key: str, default: int = 0) -> int:
    """Read an integer from the environment."""
    return int(os.getenv(key, str(default)))


def env_float(key: str, default: float = 0.0) -> float:
    """Read a float from the environment."""
    return float(os.getenv(key, str(default)))


def env_bool(key: str, default: bool = False) -> bool:
    """Read a boolean from the environment (accepts 1/true/yes)."""
    val = os.getenv(key, str(default)).strip().lower()
    return val in {"1", "true", "yes"}


def env_optional(key: str):
    """Read a value that may be intentionally unset (returns None if missing/empty)."""
    val = os.getenv(key, "").strip()
    return val if val else None


# ---------------------------------------------------------------------------
# RAG Settings
# ---------------------------------------------------------------------------

RAG_OPENROUTER_API_KEY = env("RAG_OPENROUTER_API_KEY")
RAG_OPENROUTER_MODEL = env("RAG_OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
RAG_OLLAMA_URL = env("RAG_OLLAMA_URL")
RAG_OLLAMA_MODEL = env("RAG_OLLAMA_MODEL", "llama3.1:8b")

RAG_TEMPERATURE = env_float("RAG_TEMPERATURE", 0.1)
RAG_CONTEXT_WINDOW = env_int("RAG_CONTEXT_WINDOW", 8192)
RAG_EMBEDDING_MODEL = env("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
RAG_FALLBACK_EMBEDDING_MODEL = env("RAG_FALLBACK_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
_dim = env_optional("RAG_EMBEDDING_DIM")
RAG_EMBEDDING_DIM = int(_dim) if _dim else None
RAG_EMBEDDING_INSTRUCTION = env("RAG_EMBEDDING_INSTRUCTION")
RAG_TOP_K = env_int("RAG_TOP_K", 10)
RAG_SIMILARITY_THRESHOLD = env_float("RAG_SIMILARITY_THRESHOLD", 0.85)
RAG_RERANKER_MODEL = env("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RAG_MULTI_GPU = env_bool("RAG_MULTI_GPU", False)
RAG_USE_FLASH_ATTENTION = env_bool("RAG_USE_FLASH_ATTENTION", True)

# ---------------------------------------------------------------------------
# NER Settings
# ---------------------------------------------------------------------------

NER_OLLAMA_URL = env("NER_OLLAMA_URL", "https://trycloudflare.com")
NER_OLLAMA_MODEL = env("NER_OLLAMA_MODEL", "llama3.1:8b")
NER_OPENROUTER_API_KEY = env("NER_OPENROUTER_API_KEY")
NER_OPENROUTER_MODEL = env("NER_OPENROUTER_MODEL", "qwen/qwen3.6-plus:free")
NER_CONFIDENCE_THRESHOLD = env_float("NER_CONFIDENCE_THRESHOLD", 0.7)
NER_CHUNK_SIZE_WORDS = env_int("NER_CHUNK_SIZE_WORDS", 250)
NER_MAX_CHUNKS = env_int("NER_MAX_CHUNKS", 3)


# ---------------------------------------------------------------------------
# Provider Selection Logic
# ---------------------------------------------------------------------------

def get_rag_provider() -> dict:
    """RAG: OpenRouter first (better quality), then Ollama as fallback."""
    has_valid_key = bool(RAG_OPENROUTER_API_KEY) and RAG_OPENROUTER_API_KEY not in {"sk-", "sk"}
    if has_valid_key:
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


def get_ner_provider() -> dict:
    """NER: Ollama first, then OpenRouter as fallback."""
    if NER_OLLAMA_URL:
        return {
            "provider": "ollama",
            "url": f"{NER_OLLAMA_URL}/api/chat",
            "model": NER_OLLAMA_MODEL,
        }
    elif NER_OPENROUTER_API_KEY:
        return {
            "provider": "openrouter",
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "model": NER_OPENROUTER_MODEL,
            "api_key": NER_OPENROUTER_API_KEY,
        }
    else:
        raise ValueError(
            "No NER provider configured. Set NER_OLLAMA_URL or NER_OPENROUTER_API_KEY"
        )
