"""
PhytoQuery — Unified Configuration
====================================
Usage:
    from backend.config import env, env_int, env_float, get_rag_provider, get_ner_provider
"""

import os
from pathlib import Path

# Auto-load env files. Precedence (highest → lowest):
#   1. Real OS env vars (always win — Slurm's CUDA_VISIBLE_DEVICES,
#      ``docker run -e``, systemd ``Environment=``).
#   2. ``.env.<PHYTOQUERY_PROFILE>`` — profile-specific overrides.
#      Profile = a single env var that switches the whole webapp's
#      config in one flip. Examples:
#        PHYTOQUERY_PROFILE=macbook  → loads .env.macbook
#        PHYTOQUERY_PROFILE=server   → loads .env.server
#        PHYTOQUERY_PROFILE=demo     → loads .env.demo
#      Unset/empty = no profile, just the base ``.env``. Replaces
#      the old "cp .env.macbook .env" shuffle — set the profile
#      once per environment (shell, systemd, Slurm batch) and the
#      right values load automatically.
#   3. ``.env`` — base / shared defaults.
#   4. Defaults declared in this module.
#
# ``override=False`` everywhere: a higher-priority source already
# in env stays put; each file only fills in values still unset.
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parent.parent

    _profile = os.environ.get("PHYTOQUERY_PROFILE", "").strip().lower()
    if _profile:
        _profile_file = _project_root / f".env.{_profile}"
        if _profile_file.exists():
            load_dotenv(_profile_file, override=False)

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
RAG_GROQ_API_KEY = env("RAG_GROQ_API_KEY")
RAG_GROQ_MODEL = env("RAG_GROQ_MODEL", "llama-3.3-70b-versatile")
RAG_OLLAMA_URL = env("RAG_OLLAMA_URL")
RAG_OLLAMA_MODEL = env("RAG_OLLAMA_MODEL", "llama3.1:8b")

# Optional remote Qdrant server (the Rust binary at qdrant/qdrant).
# Leave empty (the default) to keep using embedded local-mode Qdrant
# at ``data/qdrant/`` — zero deployment dependency, works on every
# OS. Set to ``http://host:6333`` to point at a Qdrant server
# instead (enables uvicorn --workers N, payload indexes,
# quantization, real HNSW; requires running ``docker run -p
# 6333:6333 -v ... qdrant/qdrant`` or equivalent). Single config
# flip; no other code path changes.
RAG_QDRANT_URL = env("RAG_QDRANT_URL")

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
NER_GROQ_API_KEY = env("NER_GROQ_API_KEY")
NER_GROQ_MODEL = env("NER_GROQ_MODEL", "llama-3.3-70b-versatile")
NER_CONFIDENCE_THRESHOLD = env_float("NER_CONFIDENCE_THRESHOLD", 0.7)
NER_CHUNK_SIZE_WORDS = env_int("NER_CHUNK_SIZE_WORDS", 250)
NER_MAX_CHUNKS = env_int("NER_MAX_CHUNKS", 3)


# ---------------------------------------------------------------------------
# Provider Selection Logic
# ---------------------------------------------------------------------------

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _has_real_key(key: str, sentinels: set) -> bool:
    """True if the key looks valid (non-empty, non-placeholder)."""
    return bool(key) and key not in sentinels


def get_rag_provider() -> dict:
    """RAG provider priority: Groq > OpenRouter > Ollama.

    Groq goes first because it's the fastest cloud option (free tier, sub-second
    latency on Llama 3.3 70b). OpenRouter is the diverse-model fallback;
    Ollama is the local fallback.
    """
    if _has_real_key(RAG_GROQ_API_KEY, {"gsk_", "gsk"}):
        return {
            "provider": "groq",
            "url": GROQ_URL,
            "model": RAG_GROQ_MODEL,
            "api_key": RAG_GROQ_API_KEY,
        }
    if _has_real_key(RAG_OPENROUTER_API_KEY, {"sk-", "sk"}):
        return {
            "provider": "openrouter",
            "url": OPENROUTER_URL,
            "model": RAG_OPENROUTER_MODEL,
            "api_key": RAG_OPENROUTER_API_KEY,
        }
    if RAG_OLLAMA_URL:
        return {
            "provider": "ollama",
            "url": f"{RAG_OLLAMA_URL}/api/chat",
            "model": RAG_OLLAMA_MODEL,
        }
    return {
        "provider": "unconfigured",
        "url": "",
        "model": "",
        "api_key": "",
    }


def get_ner_provider() -> dict:
    """NER provider priority: Ollama > Groq > OpenRouter.

    Local Ollama wins for NER because the workload is bulk per-paper extraction
    where local-GPU latency beats cloud round-trips. Groq is the cloud-fast
    fallback, OpenRouter the diverse-model fallback.
    """
    if NER_OLLAMA_URL:
        return {
            "provider": "ollama",
            "url": f"{NER_OLLAMA_URL}/api/chat",
            "model": NER_OLLAMA_MODEL,
        }
    if _has_real_key(NER_GROQ_API_KEY, {"gsk_", "gsk"}):
        return {
            "provider": "groq",
            "url": GROQ_URL,
            "model": NER_GROQ_MODEL,
            "api_key": NER_GROQ_API_KEY,
        }
    if NER_OPENROUTER_API_KEY:
        return {
            "provider": "openrouter",
            "url": OPENROUTER_URL,
            "model": NER_OPENROUTER_MODEL,
            "api_key": NER_OPENROUTER_API_KEY,
        }
    raise ValueError(
        "No NER provider configured. Set NER_OLLAMA_URL, NER_GROQ_API_KEY, or NER_OPENROUTER_API_KEY"
    )
