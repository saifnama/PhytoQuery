"""Deep coverage of every runtime LLM call site (RAG + NER).

All transports faked — no key, no network. Each test drives one call
path with a scripted fake client and asserts the contract.
"""
import pytest

from backend.src.common.llm_client import (
    LLMAuthError,
    LLMRateLimitError,
    LLMResponse,
    LLMTimeoutError,
    LLMUpstreamError,
)


class ScriptedClient:
    """Fake shared client: queued invoke results + stream deltas."""

    def __init__(self, invoke_results=None, stream_deltas=None,
                 stream_error=None):
        self.invoke_results = list(invoke_results or [])
        self.stream_deltas = list(stream_deltas or [])
        self.stream_error = stream_error
        self.invoke_calls = []
        self.astream_calls = 0

    async def invoke(self, **kwargs):
        self.invoke_calls.append(kwargs)
        if not self.invoke_results:
            raise AssertionError("invoke called with empty script")
        result = self.invoke_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return LLMResponse(result)

    async def astream(self, **kwargs):
        self.astream_calls += 1
        if self.stream_error is not None:
            raise self.stream_error
        for delta in self.stream_deltas:
            yield delta


def _rag_service(scripted):
    from backend.src.chat.service import RAGService
    from backend.src.chat.llm import SDKLLMAdapter

    return RAGService(llm=SDKLLMAdapter(_client=scripted))


def _sources():
    return [
        {"chunk_id": "c1", "chunk_text": "Eugenol is the main compound in clove oil."},
        {"chunk_id": "c2", "chunk_text": "Quercetin shows antioxidant activity in assays."},
    ]


# --- _invoke_llm ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_invoke_llm_progressive_drop():
    from backend.src.chat import service as rag_engine

    class LegacyFake:
        def __init__(self):
            self.calls = []

        async def invoke(self, prompt=None, messages=None):
            self.calls.append((prompt, messages))
            return LLMResponse("ok")

    svc = rag_engine.RAGService.__new__(rag_engine.RAGService)
    svc.llm = LegacyFake()
    # A direct full-kwarg call raises on the legacy signature...
    with pytest.raises(TypeError):
        await svc.llm.invoke(prompt="q", timeout_seconds=5)
    # ...but _invoke_llm sheds unknown kwargs until the call sticks.
    resp = await svc._invoke_llm(prompt="q", timeout_seconds=5,
                                 response_format={"type": "json_object"})
    assert resp.content == "ok"
    assert svc.llm.calls[-1] == ("q", None)


@pytest.mark.asyncio
async def test_invoke_llm_forwards_json_and_timeout():
    scripted = ScriptedClient(invoke_results=['{"chunk_ids": ["c1"]}'])
    svc = _rag_service(scripted)
    resp = await svc._invoke_llm(
        prompt="q", timeout_seconds=12.5,
        response_format={"type": "json_object"}, max_retries=1,
    )
    assert resp.content == '{"chunk_ids": ["c1"]}'
    call = scripted.invoke_calls[0]
    assert call["response_format"] == {"type": "json_object"}
    assert call["timeout_seconds"] == 12.5


# --- _select_used_chunks ----------------------------------------------------------


@pytest.mark.asyncio
async def test_select_used_chunks_first_try():
    scripted = ScriptedClient(invoke_results=['{"chunk_ids": ["c2", "c1"]}'])
    svc = _rag_service(scripted)
    out = await svc._select_used_chunks("q?", "answer text", _sources())
    assert out == ["c2", "c1"]
    assert len(scripted.invoke_calls) == 1


@pytest.mark.asyncio
async def test_select_used_chunks_retries_with_hint():
    scripted = ScriptedClient(invoke_results=[
        "not json {{{",
        '{"chunk_ids": ["c1"]}',
    ])
    svc = _rag_service(scripted)
    out = await svc._select_used_chunks("q?", "answer text", _sources(),
                                        max_attempts=2)
    assert out == ["c1"]
    assert len(scripted.invoke_calls) == 2
    assert "rejected" in scripted.invoke_calls[1]["prompt"]


@pytest.mark.asyncio
async def test_select_used_chunks_llm_error_gives_empty():
    scripted = ScriptedClient(
        invoke_results=[LLMUpstreamError("down"), LLMUpstreamError("down")])
    svc = _rag_service(scripted)
    assert await svc._select_used_chunks("q?", "a", _sources(),
                                         max_attempts=2) == []


@pytest.mark.asyncio
async def test_select_used_chunks_rejects_hallucinated_ids():
    scripted = ScriptedClient(invoke_results=['{"chunk_ids": ["c99"]}'])
    svc = _rag_service(scripted)
    assert await svc._select_used_chunks("q?", "a", _sources(),
                                         max_attempts=1) == []


# --- _extract_citations -------------------------------------------------------------


@pytest.mark.asyncio
async def test_extract_citations_skips_llm_without_markers():
    scripted = ScriptedClient(invoke_results=['{"citations": []}'])
    svc = _rag_service(scripted)
    assert await svc._extract_citations("plain answer", _sources()) == []
    assert scripted.invoke_calls == []


@pytest.mark.asyncio
async def test_extract_citations_validates_quotes():
    chunk = "Eugenol is the main compound in clove oil."
    scripted = ScriptedClient(invoke_results=[
        '{"citations": ['
        '{"chunk_id": "c1", "quote": "Eugenol is the main compound"},'
        '{"chunk_id": "c2", "quote": "totally invented text here"}]}',
    ])
    svc = _rag_service(scripted)
    out = await svc._extract_citations("answer [c1] and [c2]", [
        {"chunk_id": "c1", "chunk_text": chunk},
        {"chunk_id": "c2", "chunk_text": "Quercetin shows antioxidant activity."},
    ])
    assert [c["chunk_id"] for c in out] == ["c1"]
    assert out[0]["verified"] is True


@pytest.mark.asyncio
async def test_extract_citations_llm_error_gives_empty():
    scripted = ScriptedClient(invoke_results=[LLMTimeoutError("slow")])
    svc = _rag_service(scripted)
    assert await svc._extract_citations("answer [c1]", _sources()) == []


# --- query (non-streaming) -----------------------------------------------------------------


def _prepared(answer_text="Eugenol works [c1].", kb_mode=False):
    return {
        "messages": [{"role": "user", "content": "q"}],
        "sources": _sources(),
        "citable_sources": [
            {**s, "page": 1, "doc_title": "T", "source": "f.pdf",
             "title": "T"} for s in _sources()
        ],
        **({"is_kb_mode": True} if kb_mode else {}),
    }


@pytest.mark.asyncio
async def test_query_builds_references_and_strips_markers():
    scripted = ScriptedClient(invoke_results=["Eugenol works [c1]."])
    svc = _rag_service(scripted)

    async def _prep(*a, **k):
        return _prepared()

    svc._prepare_query = _prep
    out = await svc.query("q", user_id="u")
    assert "[c1]" not in out["answer"]
    assert "**References**" in out["answer"]
    assert out["sources"]


@pytest.mark.asyncio
async def test_query_no_context_short_circuits_without_llm():
    scripted = ScriptedClient(invoke_results=["SHOULD NOT BE USED"])
    svc = _rag_service(scripted)

    async def _prep(*a, **k):
        return {"answer": "canned", "sources": []}

    svc._prepare_query = _prep
    out = await svc.query("q", user_id="u")
    assert out["answer"] == "canned"
    assert scripted.invoke_calls == []


# --- query_stream ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_stream_frame_contract():
    scripted = ScriptedClient(
        invoke_results=['{"citations": []}'],
        stream_deltas=["Eugenol ", "works [c1]."],
    )
    svc = _rag_service(scripted)

    async def _prep(*a, **k):
        return _prepared()

    svc._prepare_query = _prep
    frames = [f async for f in svc.query_stream("q", user_id="u")]
    kinds = [f["type"] for f in frames]
    assert kinds[0] == "sources"
    assert "text_delta" in kinds
    assert kinds[-1] == "done"
    assert "".join(
        f.get("text", "") for f in frames if f["type"] == "text_delta"
    ) == "Eugenol works [c1]."


@pytest.mark.asyncio
async def test_query_stream_auth_error_frame():
    scripted = ScriptedClient(stream_error=LLMAuthError("bad key"))
    svc = _rag_service(scripted)

    async def _prep(*a, **k):
        return _prepared()

    svc._prepare_query = _prep
    frames = [f async for f in svc.query_stream("q", user_id="u")]
    assert frames[0]["type"] == "sources"
    assert frames[1]["type"] == "error"
    assert "Auth failed" in frames[1]["error"]


@pytest.mark.asyncio
async def test_query_stream_llm_error_frame():
    scripted = ScriptedClient(stream_error=RuntimeError("boom"))
    svc = _rag_service(scripted)

    async def _prep(*a, **k):
        return _prepared()

    svc._prepare_query = _prep
    frames = [f async for f in svc.query_stream("q", user_id="u")]
    assert frames[-1]["type"] == "error"


# --- suggest / summarize ----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_suggest_followups_cleans_markers():
    scripted = ScriptedClient(invoke_results=[
        "1. What is eugenol?\n- How is it extracted?\n* Why useful?\nOk?"
    ])
    svc = _rag_service(scripted)
    out = await svc.suggest_followups(
        [{"role": "user", "content": "Tell me about clove oil"}])
    # Numbering/bullets stripped; sub-6-char non-questions dropped.
    assert out == ["What is eugenol?", "How is it extracted?",
                   "Why useful?"]


@pytest.mark.asyncio
async def test_suggest_followups_empty_and_failure():
    svc = _rag_service(ScriptedClient(invoke_results=["x"]))
    assert await svc.suggest_followups([]) == []
    failing = _rag_service(
        ScriptedClient(invoke_results=[LLMUpstreamError("down")]))
    assert await failing.suggest_followups(
        [{"role": "user", "content": "hi"}]) == []


@pytest.mark.asyncio
async def test_summarize_document_paths():
    from backend.src.chat.llm import RAGLLMTimeoutError

    ok = _rag_service(ScriptedClient(invoke_results=["  A short summary.  "]))
    assert await ok.summarize_document("text", "f.pdf") == "A short summary."
    slow = _rag_service(
        ScriptedClient(invoke_results=[RAGLLMTimeoutError("t")]))
    assert await slow.summarize_document("text", "f.pdf") == ""


# --- adapter rate-limit -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapter_retries_rate_limit_then_succeeds(monkeypatch):
    import asyncio as _asyncio

    from backend.src.chat.llm import SDKLLMAdapter

    sleeps = []
    real_sleep = _asyncio.sleep
    monkeypatch.setattr(_asyncio, "sleep",
                        lambda d: sleeps.append(d) or real_sleep(0))
    scripted = ScriptedClient(invoke_results=[
        LLMRateLimitError("slow"), "recovered",
    ])
    adapter = SDKLLMAdapter(_client=scripted)
    resp = await adapter.invoke(prompt="q", max_retries=3, base_delay=0)
    assert resp.content == "recovered"
    assert len(scripted.invoke_calls) == 2


@pytest.mark.asyncio
async def test_adapter_exhausts_rate_limit_retries():
    from backend.src.chat.llm import SDKLLMAdapter

    scripted = ScriptedClient(invoke_results=[
        LLMRateLimitError("s1"), LLMRateLimitError("s2"),
    ])
    adapter = SDKLLMAdapter(_client=scripted)
    with pytest.raises(LLMRateLimitError):
        await adapter.invoke(prompt="q", max_retries=2, base_delay=0)


# --- NER call_llm ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ner_call_llm_request_shape(monkeypatch):
    from backend.src.ner.service import NERService
    import backend.src.ner.llm as ner_llm_module
    import backend.src.ner.service as ner_module

    scripted = ScriptedClient(invoke_results=["[]"])
    svc = NERService()
    monkeypatch.setattr(ner_llm_module, "get_llm_client", lambda: scripted)
    assert await svc.call_llm("leaf tissue") == "[]"
    call = scripted.invoke_calls[0]
    assert [m["role"] for m in call["messages"]] == ["system", "user"]
    assert "leaf tissue" in call["messages"][1]["content"]
    assert call["temperature"] == 0.0
    assert call["max_tokens"] == 2048
    assert call["json_mode"] is True


@pytest.mark.asyncio
async def test_ner_call_llm_error_hint_and_failures(monkeypatch):
    from backend.src.settings import LLMConfigError
    from backend.src.ner.service import NERService
    import backend.src.ner.llm as ner_llm_module
    import backend.src.ner.service as ner_module

    svc = NERService()
    scripted = ScriptedClient(invoke_results=["[]"])
    monkeypatch.setattr(ner_llm_module, "get_llm_client", lambda: scripted)
    await svc.call_llm("chunk", error_hint="bad json")
    user_content = scripted.invoke_calls[0]["messages"][1]["content"]
    assert "bad json" in user_content

    failing = ScriptedClient(invoke_results=[LLMTimeoutError("t")])
    monkeypatch.setattr(ner_llm_module, "get_llm_client", lambda: failing)
    assert await svc.call_llm("chunk") == ""

    def _raise():
        raise LLMConfigError("unconfigured")

    monkeypatch.setattr(ner_llm_module, "get_llm_client", _raise)
    assert await svc.call_llm("chunk") == ""


# --- NER validation-retry matrix -----------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ner_retry_matrix(monkeypatch):
    from backend.src.ner.service import NERService
    import backend.src.ner.llm as ner_llm_module
    import backend.src.ner.service as ner_module

    svc = NERService()

    async def run(script, **kwargs):
        scripted = ScriptedClient(invoke_results=script)
        monkeypatch.setattr(ner_llm_module, "get_llm_client",
                            lambda: scripted)
        out = await svc._extract_entities_with_retry("leaf tissue",
                                                     **kwargs)
        return out, scripted

    good = '[{"span": "Eugenol", "type": "CHEMICAL", "score": 0.9}]'
    out, scripted = await run([good])
    assert [(e["text"], e["label"]) for e in out] == [("Eugenol", "CHEMICAL")]
    assert len(scripted.invoke_calls) == 1

    out, scripted = await run(["", ""], max_attempts=2)
    assert out == []
    assert len(scripted.invoke_calls) == 2
    assert "empty" in scripted.invoke_calls[1]["messages"][1]["content"]

    out, scripted = await run(["{{{garbage", good], max_attempts=2)
    assert len(out) == 1
    hint = scripted.invoke_calls[1]["messages"][1]["content"].lower()
    assert "validation" in hint or "json" in hint

    out, scripted = await run(
        ['[{"span": "x", "type": "MOLECULE"}]', good], max_attempts=2)
    assert len(out) == 1
    assert "uppercase" in scripted.invoke_calls[1]["messages"][1][
        "content"].lower() or "CHEMICAL" in scripted.invoke_calls[1][
        "messages"][1]["content"]

    out, scripted = await run(
        ['{"entities": [{"span": "Neem", "type": "SPECIES"}]}'])
    assert [(e["text"], e["label"]) for e in out] == [("Neem", "SPECIES")]

    out, scripted = await run(["[]"], max_attempts=1)
    assert out == []


@pytest.mark.asyncio
async def test_ner_json_mode_400_falls_back_to_plain(monkeypatch):
    from backend.src.common.llm_client import LLMResponse, LLMUpstreamError
    from backend.src.ner.service import NERService
    import backend.src.ner.llm as ner_llm_module
    import backend.src.ner.service as ner_module

    class FlakyClient(ScriptedClient):
        async def invoke(self, **kwargs):
            self.invoke_calls.append(kwargs)
            if kwargs.get("response_format") is not None:
                raise LLMUpstreamError("LLM upstream error (400)")
            return LLMResponse("[]")

    svc = NERService()
    monkeypatch.setattr(ner_llm_module, "get_llm_client",
                        lambda: FlakyClient())
    assert await svc.call_llm("leaf tissue") == "[]"


@pytest.mark.asyncio
async def test_ner_rate_limit_fails_fast_without_retry(monkeypatch):
    from backend.src.common.llm_client import LLMRateLimitError
    from backend.src.ner.service import NERService
    import backend.src.ner.llm as ner_llm_module
    import backend.src.ner.service as ner_module

    svc = NERService()
    scripted = ScriptedClient(invoke_results=[
        LLMRateLimitError("slow"), LLMRateLimitError("slow"),
        LLMRateLimitError("slow"),
    ])
    monkeypatch.setattr(ner_llm_module, "get_llm_client", lambda: scripted)
    assert await svc._extract_entities_with_retry(
        "leaf tissue", max_attempts=3) == []
    assert len(scripted.invoke_calls) == 1

    with pytest.raises(LLMRateLimitError):
        await svc.call_llm("leaf tissue")


@pytest.mark.asyncio
async def test_ner_double_decodes_quoted_array(monkeypatch):
    from backend.src.ner.service import NERService
    import backend.src.ner.llm as ner_llm_module
    import backend.src.ner.service as ner_module

    svc = NERService()

    async def run(script, **kwargs):
        scripted = ScriptedClient(invoke_results=script)
        monkeypatch.setattr(ner_llm_module, "get_llm_client",
                            lambda: scripted)
        out = await svc._extract_entities_with_retry("leaf tissue",
                                                     **kwargs)
        return out, scripted

    quoted = '"[{\\"span\\": \\"Eugenol\\", \\"type\\": \\"CHEMICAL\\"}]"'
    out, _ = await run([quoted])
    assert [(e["text"], e["label"]) for e in out] == [("Eugenol", "CHEMICAL")]


@pytest.mark.asyncio
async def test_ner_single_bare_object_wrapped(monkeypatch):
    from backend.src.ner.service import NERService
    import backend.src.ner.llm as ner_llm_module
    import backend.src.ner.service as ner_module

    svc = NERService()
    # Exact shape from the local Qwen log: one entity, bare object.
    bare = ('{"span": "Eugenol", "type": "CHEMICAL", "start": 0, '
            '"end": 7, "name_type": null, "linked_to": null}')
    scripted = ScriptedClient(invoke_results=[bare])
    monkeypatch.setattr(ner_llm_module, "get_llm_client", lambda: scripted)
    out = await svc._extract_entities_with_retry("leaf tissue")
    assert [(e["text"], e["label"]) for e in out] == [("Eugenol", "CHEMICAL")]
    assert len(scripted.invoke_calls) == 1

    assert [(e["text"], e["label"]) for e in
            svc.parse_llm_response(bare)] == [("Eugenol", "CHEMICAL")]


@pytest.mark.asyncio
async def test_ner_hybrid_false_skips_all_llm(monkeypatch):
    from backend.src.ner.service import NERService
    import backend.src.ner.llm as ner_llm_module
    import backend.src.ner.service as ner_module

    svc = NERService()
    calls = []

    async def _spy(chunk, error_hint=None):
        calls.append(chunk)
        return "[]"

    monkeypatch.setattr(NERService, "call_llm", _spy)
    monkeypatch.setattr(ner_module, "NER_HYBRID", False)

    summary, filtered = await svc.process_text(
        "Eugenol from Ocimum sanctum leaves.", max_chunks=1)
    assert calls == []
    assert filtered  # dictionary-only still extracts

    summary, filtered = await svc.process_sections([
        {"title": "Abstract", "content": "Eugenol from clove leaves."},
    ])
    assert calls == []
    assert filtered


def test_ner_hybrid_defaults_on(monkeypatch):
    import backend.src.settings as config_module

    monkeypatch.delenv("NER_HYBRID", raising=False)
    import importlib

    importlib.reload(config_module)
    try:
        assert config_module.NER_HYBRID is True
    finally:
        importlib.reload(config_module)


@pytest.mark.asyncio
async def test_ner_process_sections_budget_skips_llm(monkeypatch):
    from backend.src.ner.service import NERService
    import backend.src.ner.llm as ner_llm_module
    import backend.src.ner.service as ner_module

    svc = NERService()
    calls = []

    async def _fake_call_llm(chunk, error_hint=None):
        calls.append(chunk)
        return "[]"

    monkeypatch.setattr(NERService, "call_llm", _fake_call_llm)
    monkeypatch.setattr(ner_module, "NER_BUDGET_SECONDS", -1.0)
    sections = [
        {"title": "Abstract", "content": "Eugenol from clove leaves."},
        {"title": "Methods", "content": "Methanol extraction was used."},
    ]
    summary, filtered = await svc.process_sections(sections)
    assert calls == []
    assert filtered  # dictionary entities still extracted


@pytest.mark.asyncio
async def test_ner_process_sections_merges_llm_with_tags(monkeypatch):
    from backend.src.ner.service import NERService

    svc = NERService()

    async def _fake_call_llm(chunk, error_hint=None):
        return '[{"span": "Zingiberene", "type": "CHEMICAL"}]'

    monkeypatch.setattr(NERService, "call_llm", _fake_call_llm)
    summary, filtered = await svc.process_sections([
        {"title": "Results", "content": "Zingiberene was detected."},
    ])
    llm_hits = [e for e in filtered if e.get("text") == "Zingiberene"]
    assert llm_hits and llm_hits[0]["section"] == "Results"


# --- embeddings: single model, fail loudly ----------------------------------------------------


def test_embeddings_single_model_no_silent_swap(monkeypatch):
    import sentence_transformers

    from backend.src.chat.embeddings import BloomIndexEmbeddings

    emb = BloomIndexEmbeddings(model="some/model")
    assert not hasattr(emb, "_fallback_model")
    assert not hasattr(emb, "_primary_model")

    def _boom(*a, **k):
        raise RuntimeError("nope")

    monkeypatch.setattr(sentence_transformers, "SentenceTransformer", _boom)
    with pytest.raises(RuntimeError, match="some/model"):
        emb._ensure_model_loaded()


def test_no_embedding_fallback_in_config():
    import backend.src.settings as config_module

    assert not hasattr(config_module, "RAG_FALLBACK_EMBEDDING_MODEL")


# --- static boundary assertions ----------------------------------------------------------------------------


def test_no_raw_http_llm_calls_in_engines():
    import inspect

    import backend.src.ner.llm as ner_llm_module
    import backend.src.ner.service as ner_module
    import backend.src.chat.citations as citations_module
    import backend.src.chat.embeddings as embeddings_module
    import backend.src.chat.ingest as ingest_module
    import backend.src.chat.llm as llm_module
    import backend.src.chat.retrieval as retrieval_module
    import backend.src.chat.service as service_module

    for module in (
        ner_module,
        citations_module,
        embeddings_module,
        ingest_module,
        llm_module,
        retrieval_module,
        service_module,
    ):
        source = inspect.getsource(module)
        assert "HttpClientManager" not in source
        assert "import httpx" not in source
        assert "/api/tags" not in source
        # Native Ollama endpoint construction (quoted); app REST-route
        # comments such as /api/chat/files/... are not LLM calls.
        assert '"/api/chat"' not in source
