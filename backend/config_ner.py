"""
NER Configuration
================
Named Entity Recognition settings and providers.
"""

import os

# =============================================================================
# OLLAMA (Primary for NER)
# =============================================================================

NER_OLLAMA_URL = os.getenv("NER_OLLAMA_URL", "https://garcia-new-stress-effective.trycloudflare.com")
NER_OLLAMA_MODEL = os.getenv("NER_OLLAMA_MODEL", "llama3.1:8b")

# =============================================================================
# OPENROUTER (Fallback for NER)
# =============================================================================

NER_OPENROUTER_API_KEY = os.getenv("NER_OPENROUTER_API_KEY", "")
NER_OPENROUTER_MODEL = os.getenv("NER_OPENROUTER_MODEL", "qwen/qwen3.6-plus:free")

# =============================================================================
# NER SETTINGS
# =============================================================================

NER_CONFIDENCE_THRESHOLD = float(os.getenv("NER_CONFIDENCE_THRESHOLD", "0.7"))
NER_CHUNK_SIZE_WORDS = int(os.getenv("NER_CHUNK_SIZE_WORDS", "250"))
NER_MAX_CHUNKS = int(os.getenv("NER_MAX_CHUNKS", "3"))


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
