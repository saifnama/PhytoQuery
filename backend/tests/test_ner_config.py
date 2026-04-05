"""
Tests for NER configuration.
"""

import pytest
import os
from unittest.mock import patch


def test_ollama_is_primary_provider():
    """When NER_OLLAMA_URL is set, use Ollama."""
    with patch.dict(
        os.environ,
        {"NER_OLLAMA_URL": "http://localhost:11434", "NER_OPENROUTER_API_KEY": ""},
        clear=False,
    ):
        import importlib
        import backend.config_ner as config_ner

        importlib.reload(config_ner)
        from backend.config_ner import get_ner_provider

        assert get_ner_provider()["provider"] == "ollama"


def test_openrouter_is_fallback():
    """When NER_OLLAMA_URL is empty but NER_OPENROUTER_API_KEY is set, use OpenRouter."""
    with patch.dict(
        os.environ,
        {"NER_OLLAMA_URL": "", "NER_OPENROUTER_API_KEY": "sk-or-test123"},
        clear=False,
    ):
        import importlib
        import backend.config_ner as config_ner

        importlib.reload(config_ner)
        from backend.config_ner import get_ner_provider

        assert get_ner_provider()["provider"] == "openrouter"


def test_raises_error_when_none_configured():
    """When no provider is configured, raise ValueError."""
    with patch.dict(
        os.environ, {"NER_OLLAMA_URL": "", "NER_OPENROUTER_API_KEY": ""}, clear=False
    ):
        import importlib
        import backend.config_ner as config_ner

        importlib.reload(config_ner)
        from backend.config_ner import get_ner_provider

        with pytest.raises(ValueError, match="No NER provider configured"):
            get_ner_provider()


def test_default_values():
    """Test default configuration values."""
    from backend.config_ner import (
        NER_OLLAMA_MODEL,
        NER_OPENROUTER_MODEL,
        NER_CONFIDENCE_THRESHOLD,
        NER_CHUNK_SIZE_WORDS,
    )

    assert NER_OLLAMA_MODEL == "llama3.1:8b"
    assert NER_OPENROUTER_MODEL == "anthropic/claude-3-haiku:free"
    assert NER_CONFIDENCE_THRESHOLD == 0.7
    assert NER_CHUNK_SIZE_WORDS == 1000
