"""
Tests for RAG configuration.
"""

import pytest
import os
from unittest.mock import patch


def test_openrouter_is_primary_provider():
    """When RAG_OPENROUTER_API_KEY is set, use OpenRouter."""
    with patch.dict(
        os.environ,
        {"RAG_OPENROUTER_API_KEY": "sk-or-test123", "RAG_OLLAMA_URL": ""},
        clear=False,
    ):
        import importlib
        import backend.config_rag as config_rag

        importlib.reload(config_rag)
        from backend.config_rag import get_rag_provider

        assert get_rag_provider()["provider"] == "openrouter"


def test_ollama_is_fallback():
    """When RAG_OPENROUTER_API_KEY is empty but RAG_OLLAMA_URL is set, use Ollama."""
    with patch.dict(
        os.environ,
        {"RAG_OPENROUTER_API_KEY": "", "RAG_OLLAMA_URL": "http://localhost:11434"},
        clear=False,
    ):
        import importlib
        import backend.config_rag as config_rag

        importlib.reload(config_rag)
        from backend.config_rag import get_rag_provider

        assert get_rag_provider()["provider"] == "ollama"


def test_raises_error_when_none_configured():
    """When no provider is configured, raise ValueError."""
    with patch.dict(
        os.environ, {"RAG_OPENROUTER_API_KEY": "", "RAG_OLLAMA_URL": ""}, clear=False
    ):
        import importlib
        import backend.config_rag as config_rag

        importlib.reload(config_rag)
        from backend.config_rag import get_rag_provider

        with pytest.raises(ValueError, match="No RAG provider configured"):
            get_rag_provider()


def test_default_values():
    """Test default configuration values."""
    from backend.config_rag import (
        RAG_OLLAMA_MODEL,
        RAG_OPENROUTER_MODEL,
        RAG_TEMPERATURE,
        RAG_TOP_K,
        RAG_SIMILARITY_THRESHOLD,
    )

    assert RAG_OLLAMA_MODEL == "llama3.1:8b"
    assert RAG_OPENROUTER_MODEL == "stepfun/step-3.5-flash:free"
    assert RAG_TEMPERATURE == 0.1
    assert RAG_TOP_K == 10
    assert RAG_SIMILARITY_THRESHOLD == 0.85
