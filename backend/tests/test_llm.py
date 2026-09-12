"""Unification regression: one SDK boundary, mocked transports only.

No API key or live LLM needed — the SDK client is always faked.
"""
import os
import warnings

import httpx
import pytest

ENV_KEYS = (
    "LLM_API_BASE_URL", "LLM_API_KEY", "LLM_MODEL",
    "LLM_TIMEOUT_SECONDS", "LLM_MAX_RETRIES",
    "LLM_THINKING", "LLM_NO_THINK_DIRECTIVE", "LLM_CHAT_TEMPLATE_KWARGS",
    "RAG_LLAMACPP_URL", "RAG_LLAMACPP_MODEL", "RAG_LLAMACPP_API_KEY",
    "RAG_OPENROUTER_API_KEY", "RAG_OPENROUTER_MODEL",
    "RAG_OLLAMA_URL", "RAG_OLLAMA_MODEL",
    "NER_LLAMACPP_URL", "NER_LLAMACPP_MODEL",
    "NER_OLLAMA_URL", "NER_OLLAMA_MODEL", "NER_OPENROUTER_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _req():
    return httpx.Request("POST", "https://test/v1/chat/completions")


class _Delta:
    def __init__(self, content):
        self.content = content


class _Choice:
    def __init__(self, text=None, delta=None):
        from types import SimpleNamespace

        self.message = SimpleNamespace(content=text)
        self.delta = SimpleNamespace(content=delta)


class _Completion:
    def __init__(self, text):
        self.choices = [_Choice(text=text)]


class _StreamEvent:
    def __init__(self, delta):
        self.choices = [_Choice(delta=delta)]


class _FakeStream:
    def __init__(self, deltas):
        self._deltas = deltas

    def __aiter__(self):
        async def _gen():
            for d in self._deltas:
                yield _StreamEvent(d)

        return _gen()


class _FakeCompletions:
    def __init__(self, text="", stream_deltas=None, error=None):
        self.text = text
        self.stream_deltas = stream_deltas
        self.error = error
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return _FakeStream(self.stream_deltas or [])
        return _Completion(self.text)


class _FakeSDK:
    def __init__(self, **kwargs):
        from types import SimpleNamespace

        self.completions = _FakeCompletions(**kwargs)
        self.chat = SimpleNamespace(completions=self.completions)


# --- settings ---------------------------------------------------------------


def test_unified_settings_precedence(monkeypatch):
    from backend.config import resolve_llm_settings

    monkeypatch.setenv("LLM_API_BASE_URL", "https://api.openai.com/v1/")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "gpt-4o-mini")
    settings = resolve_llm_settings()
    assert settings.base_url == "https://api.openai.com/v1"
    assert settings.model == "gpt-4o-mini"
    assert "sk-test" not in repr(settings)


def test_base_url_rejects_native_ollama(monkeypatch):
    from backend.config import LLMConfigError, resolve_llm_settings

    monkeypatch.setenv("LLM_API_BASE_URL", "http://localhost:11434/api/chat")
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(LLMConfigError):
        resolve_llm_settings()


def test_base_url_requires_v1_root(monkeypatch):
    from backend.config import LLMConfigError, resolve_llm_settings

    monkeypatch.setenv("LLM_API_BASE_URL", "https://example.com")
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("LLM_MODEL", "m")
    with pytest.raises(LLMConfigError):
        resolve_llm_settings()


def test_operation_path_is_forgiven(monkeypatch):
    from backend.config import resolve_llm_settings

    monkeypatch.setenv("LLM_API_BASE_URL", "https://example.com/v1/chat/completions")
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("LLM_MODEL", "m")
    assert resolve_llm_settings().base_url == "https://example.com/v1"


def test_model_required(monkeypatch):
    from backend.config import LLMConfigError, resolve_llm_settings

    monkeypatch.setenv("LLM_API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    with pytest.raises(LLMConfigError):
        resolve_llm_settings()


def test_legacy_openrouter_fallback(monkeypatch):
    from backend.config import resolve_llm_settings

    monkeypatch.setenv("RAG_OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("RAG_OPENROUTER_MODEL", "some/model:free")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        settings = resolve_llm_settings()
    assert settings.base_url == "https://openrouter.ai/api/v1"
    assert settings.model == "some/model:free"


def test_unified_wins_over_legacy(monkeypatch):
    from backend.config import resolve_llm_settings

    monkeypatch.setenv("RAG_OPENROUTER_API_KEY", "sk-or-legacy")
    monkeypatch.setenv("RAG_OPENROUTER_MODEL", "legacy/model")
    monkeypatch.setenv("LLM_API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-new")
    monkeypatch.setenv("LLM_MODEL", "new-model")
    settings = resolve_llm_settings()
    assert settings.model == "new-model"


# --- client wrapper ----------------------------------------------------------


def _client_for(model="test-model", **fake_kwargs):
    from backend.core.llm_client import SharedLLMClient

    fake = _FakeSDK(**fake_kwargs)
    return SharedLLMClient(sdk_client=fake, model=model), fake


@pytest.mark.asyncio
async def test_invoke_extracts_text():
    client, fake = _client_for(text="hello world")
    response = await client.invoke(prompt="hi")
    assert response.content == "hello world"
    assert fake.completions.last_kwargs["model"] == "test-model"
    assert fake.completions.last_kwargs["messages"] == [
        {"role": "user", "content": "hi\n\n/no_think"}
    ]


@pytest.mark.asyncio
async def test_invoke_json_mode_param():
    client, fake = _client_for(text='{"a": 1}')
    await client.invoke(
        prompt="x", response_format={"type": "json_object"}
    )
    assert fake.completions.last_kwargs["response_format"] == {
        "type": "json_object"
    }


@pytest.mark.asyncio
async def test_no_think_appended_by_default():
    client, fake = _client_for(text="ok")
    await client.invoke(prompt="hello")
    sent = fake.completions.last_kwargs["messages"][-1]["content"]
    assert sent.endswith("\n\n/no_think")


@pytest.mark.asyncio
async def test_no_think_skipped_when_thinking_enabled():
    client, fake = _client_for(text="ok")
    await client.invoke(prompt="hello", thinking=True)
    sent = fake.completions.last_kwargs["messages"][-1]["content"]
    assert "/no_think" not in sent


@pytest.mark.asyncio
async def test_no_think_does_not_mutate_caller_messages():
    client, fake = _client_for(text="ok", stream_deltas=["ok"])
    original = [{"role": "user", "content": "hello"}]
    await client.invoke(messages=original)
    assert original == [{"role": "user", "content": "hello"}]
    assert fake.completions.last_kwargs["messages"][-1]["content"].endswith(
        "\n\n/no_think"
    )
    parts = [c async for c in client.astream(messages=original)]
    assert parts == ["ok"]
    assert original == [{"role": "user", "content": "hello"}]


@pytest.mark.asyncio
async def test_invoke_requires_prompt_or_messages():
    client, _ = _client_for()
    with pytest.raises(ValueError):
        await client.invoke()


@pytest.mark.asyncio
async def test_error_mapping():
    from openai import (
        APITimeoutError,
        APIStatusError,
        AuthenticationError,
        RateLimitError,
    )
    from backend.core.llm_client import (
        LLMAuthError,
        LLMRateLimitError,
        LLMTimeoutError,
        LLMUpstreamError,
        SharedLLMClient,
    )

    cases = [
        (AuthenticationError("nope", response=httpx.Response(401, request=_req()), body=None), LLMAuthError),
        (RateLimitError("slow", response=httpx.Response(429, request=_req()), body=None), LLMRateLimitError),
        (APITimeoutError(request=_req()), LLMTimeoutError),
        (APIStatusError("boom", response=httpx.Response(500, request=_req()), body=None), LLMUpstreamError),
    ]
    for sdk_error, app_error in cases:
        client, _ = _client_for(error=sdk_error)
        with pytest.raises(app_error):
            await client.invoke(prompt="hi")


@pytest.mark.asyncio
async def test_astream_yields_deltas():
    client, _ = _client_for(stream_deltas=["Hel", "lo", "", None, " world"])
    parts = [chunk async for chunk in client.astream(prompt="hi")]
    assert "".join(parts) == "Hello world"


# --- adapter contract ----------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_preserves_return_shape():
    from backend.services.rag_engine import SDKLLMAdapter

    client, _ = _client_for(text="answer text", stream_deltas=["answer text"])
    adapter = SDKLLMAdapter(_client=client)
    response = await adapter.invoke(
        prompt="q", response_format={"type": "json_object"}
    )
    assert response.content == "answer text"
    deltas = [c async for c in adapter.astream(prompt="q")]
    assert deltas == ["answer text"]


def test_alias_names_survive():
    from backend.services import rag_engine

    assert rag_engine.OllamaLLM is rag_engine.SDKLLMAdapter
    assert issubclass(rag_engine.RAGProviderAuthError, Exception)
    assert issubclass(rag_engine.RAGLLMTimeoutError, Exception)


# --- service-level shapes ------------------------------------------------------


def test_ner_parse_shape_preserved():
    from backend.services.ner_engine import NERService

    service = NERService()
    parsed = service.parse_llm_response(
        '[{"span": "Eugenol", "type": "CHEMICAL", "score": 0.9}]'
    )
    assert parsed == [
        {
            "text": "Eugenol",
            "label": "CHEMICAL",
            "score": 0.9,
            "name_type": None,
            "linked_to": None,
        }
    ]
    assert service.parse_llm_response("not json at all") == []


@pytest.mark.asyncio
async def test_ner_call_llm_uses_shared_client(monkeypatch):
    from backend.services.ner_engine import NERService
    import backend.services.ner_engine as ner_module

    client, fake = _client_for(text="[]")
    monkeypatch.setattr(ner_module, "get_llm_client", lambda: client)
    service = NERService()
    assert await service.call_llm("some text") == "[]"
    messages = fake.completions.last_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "some text" in messages[1]["content"]


def test_health_reports_configured(monkeypatch):
    from backend import config as config_module

    monkeypatch.setenv("LLM_API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "m")
    status = config_module.llm_status()
    assert status["llm"] == "configured"


def test_old_provider_chain_is_gone():
    import backend.config as config_module

    for name in (
        "get_rag_provider",
        "get_ner_provider",
        "OPENROUTER_URL",
        "_normalize_openai_compat_url",
        "RAG_OPENROUTER_API_KEY",
        "NER_OLLAMA_URL",
    ):
        assert not hasattr(config_module, name), name
    import backend.services.ner_engine as ner_module

    assert not hasattr(ner_module, "get_active_provider")
    assert not hasattr(
        ner_module.NERService, "_call_openai_compatible"
    )


def _unified_env(monkeypatch):
    monkeypatch.setenv("LLM_API_BASE_URL", "https://api.openai.com/v1")
    monkeypatch.setenv("LLM_API_KEY", "sk-test")
    monkeypatch.setenv("LLM_MODEL", "m")


def test_thinking_defaults_off(monkeypatch):
    from backend.config import resolve_llm_settings

    _unified_env(monkeypatch)
    settings = resolve_llm_settings()
    assert settings.thinking is False
    assert settings.no_think_directive == "/no_think"


def test_thinking_env_override(monkeypatch):
    from backend.config import resolve_llm_settings

    _unified_env(monkeypatch)
    monkeypatch.setenv("LLM_THINKING", "true")
    monkeypatch.setenv("LLM_NO_THINK_DIRECTIVE", "/think")
    settings = resolve_llm_settings()
    assert settings.thinking is True
    assert settings.no_think_directive == "/think"


@pytest.mark.asyncio
async def test_custom_directive_appended():
    from backend.core.llm_client import SharedLLMClient

    fake = _FakeSDK(text="ok")
    client = SharedLLMClient(
        sdk_client=fake, model="m", no_think_directive="/nothink"
    )
    await client.invoke(prompt="hello")
    sent = fake.completions.last_kwargs["messages"][-1]["content"]
    assert sent.endswith("\n\n/nothink")


@pytest.mark.asyncio
async def test_empty_directive_appends_nothing():
    from backend.core.llm_client import SharedLLMClient

    fake = _FakeSDK(text="ok")
    client = SharedLLMClient(
        sdk_client=fake, model="m", no_think_directive=""
    )
    await client.invoke(prompt="hello")
    sent = fake.completions.last_kwargs["messages"][-1]["content"]
    assert sent == "hello"


@pytest.mark.asyncio
async def test_client_level_thinking_true_skips_directive():
    from backend.core.llm_client import SharedLLMClient

    fake = _FakeSDK(text="ok")
    client = SharedLLMClient(sdk_client=fake, model="m", thinking=True)
    await client.invoke(prompt="hello")
    sent = fake.completions.last_kwargs["messages"][-1]["content"]
    assert sent == "hello"


@pytest.mark.asyncio
async def test_chat_template_kwargs_off_by_default():
    client, fake = _client_for(text="ok")
    await client.invoke(prompt="hello")
    assert "extra_body" not in fake.completions.last_kwargs


@pytest.mark.asyncio
async def test_chat_template_kwargs_sent_when_enabled():
    from backend.core.llm_client import SharedLLMClient

    fake = _FakeSDK(text="ok")
    client = SharedLLMClient(
        sdk_client=fake, model="m", chat_template_kwargs=True
    )
    await client.invoke(prompt="hello")
    assert fake.completions.last_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    await client.invoke(prompt="hello", thinking=True)
    assert fake.completions.last_kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": True}
    }


def test_chat_template_kwargs_env(monkeypatch):
    from backend.config import resolve_llm_settings

    _unified_env(monkeypatch)
    assert resolve_llm_settings().chat_template_kwargs is False
    monkeypatch.setenv("LLM_CHAT_TEMPLATE_KWARGS", "true")
    assert resolve_llm_settings().chat_template_kwargs is True


def test_lifespan_preloads_gazetteers(monkeypatch):
    import backend.services.ner_engine as ner_module

    calls = []
    monkeypatch.setattr(ner_module, "preload_gazetteers",
                        lambda: calls.append(1) or 8)
    from fastapi.testclient import TestClient
    from backend.app import app

    with TestClient(app):
        pass
    assert calls == [1]


def test_lifespan_survives_preload_failure(monkeypatch):
    import backend.services.ner_engine as ner_module

    def _boom():
        raise RuntimeError("disk gone")

    monkeypatch.setattr(ner_module, "preload_gazetteers", _boom)
    from fastapi.testclient import TestClient
    from backend.app import app

    with TestClient(app) as client:
        assert client.get("/health/ready").status_code in (200, 503)
