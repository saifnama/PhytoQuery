"""Regression tests for backend.services.rag_engine after the LlamaIndex +
Qdrant migration.

The test loader injects minimal stubs for ``llama_index.core``,
``llama_index.core.llms``, ``llama_index.core.embeddings``,
``llama_index.vector_stores.qdrant``, ``qdrant_client``, and
``sentence_transformers`` so the rag_engine module imports cleanly even
when the real packages aren't installed in the test environment. Tests
that need RAGService.__init__ to run patch
``_configure_llama_index_settings`` to a no-op so they don't drag the
LlamaIndex Settings singleton into the picture.
"""

import asyncio
import importlib
import sys
import types
from types import SimpleNamespace


def _install_stub(name: str, module: types.ModuleType) -> None:
    sys.modules[name] = module


def _make_llama_index_stubs() -> None:
    """Build the minimal llama_index surface the rag_engine module imports."""

    li_pkg = types.ModuleType("llama_index")
    li_core = types.ModuleType("llama_index.core")

    class Document:
        def __init__(self, text: str = "", metadata=None, **kwargs):
            self.text = text
            self.metadata = metadata or {}

    class Settings:
        llm = None
        embed_model = None

    class StorageContext:
        def __init__(self, docstore=None, index_store=None, vector_store=None, persist_dir=None):
            self.docstore = docstore
            self.index_store = index_store
            self.vector_store = vector_store
            self.persist_dir = persist_dir

        @classmethod
        def from_defaults(cls, docstore=None, index_store=None, vector_store=None, persist_dir=None):
            return cls(docstore=docstore, index_store=index_store, vector_store=vector_store, persist_dir=persist_dir)

        def persist(self, persist_dir=None):
            return None

    class VectorStoreIndex:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        @classmethod
        def from_vector_store(cls, vector_store, storage_context=None, **kwargs):
            return cls(vector_store=vector_store, storage_context=storage_context, **kwargs)

        def as_retriever(self, **kwargs):
            class _R:
                async def aretrieve(self, *a, **kw):
                    return []

                def retrieve(self, *a, **kw):
                    return []

            return _R()

    li_core.Document = Document
    li_core.Settings = Settings
    li_core.StorageContext = StorageContext
    li_core.VectorStoreIndex = VectorStoreIndex
    li_pkg.core = li_core

    # llama_index.core.schema (NodeRelationship + NodeWithScore for
    # post-rerank parent resolution)
    schema_mod = types.ModuleType("llama_index.core.schema")

    class NodeRelationship:
        SOURCE = "source"
        PARENT = "parent"
        CHILD = "child"
        NEXT = "next"
        PREVIOUS = "previous"

    class NodeWithScore:
        def __init__(self, node=None, score=None):
            self.node = node
            self.score = score

    schema_mod.NodeRelationship = NodeRelationship
    schema_mod.NodeWithScore = NodeWithScore
    li_core.schema = schema_mod

    # llama_index.core.llms
    llms_mod = types.ModuleType("llama_index.core.llms")
    callbacks_mod = types.ModuleType("llama_index.core.llms.callbacks")

    def _decorator_factory(*_a, **_kw):
        def _wrap(fn):
            return fn

        return _wrap

    callbacks_mod.llm_chat_callback = _decorator_factory
    callbacks_mod.llm_completion_callback = _decorator_factory

    class CustomLLM:
        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class CompletionResponse:
        def __init__(self, text=""):
            self.text = text

    class CompletionResponseGen:
        pass

    class CompletionResponseAsyncGen:
        pass

    class ChatResponse:
        def __init__(self, message=None):
            self.message = message

    class ChatResponseGen:
        pass

    class ChatResponseAsyncGen:
        pass

    class ChatMessage:
        def __init__(self, role=None, content=""):
            self.role = role
            self.content = content

    class LLMMetadata:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class MessageRole:
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"

    llms_mod.CustomLLM = CustomLLM
    llms_mod.CompletionResponse = CompletionResponse
    llms_mod.CompletionResponseGen = CompletionResponseGen
    llms_mod.CompletionResponseAsyncGen = CompletionResponseAsyncGen
    llms_mod.ChatResponse = ChatResponse
    llms_mod.ChatResponseGen = ChatResponseGen
    llms_mod.ChatResponseAsyncGen = ChatResponseAsyncGen
    llms_mod.ChatMessage = ChatMessage
    llms_mod.LLMMetadata = LLMMetadata
    llms_mod.MessageRole = MessageRole
    llms_mod.callbacks = callbacks_mod
    li_core.llms = llms_mod

    # llama_index.core.embeddings
    embed_mod = types.ModuleType("llama_index.core.embeddings")

    class BaseEmbedding:
        model_name = "stub-embedding"
        embed_batch_size = 16

        def __init__(self, *args, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    embed_mod.BaseEmbedding = BaseEmbedding
    li_core.embeddings = embed_mod

    # llama_index.core.node_parser
    np_mod = types.ModuleType("llama_index.core.node_parser")

    class HierarchicalNodeParser:
        def __init__(self, *a, **kw):
            self.kw = kw

        @classmethod
        def from_defaults(cls, **kwargs):
            return cls(**kwargs)

        def get_nodes_from_documents(self, documents):
            nodes = []
            for d in documents:
                node = SimpleNamespace(
                    text=getattr(d, "text", ""),
                    metadata=dict(getattr(d, "metadata", {}) or {}),
                    relationships={},
                )
                nodes.append(node)
            return nodes

    def get_leaf_nodes(nodes):
        return list(nodes)

    def get_root_nodes(nodes):
        return list(nodes)

    np_mod.HierarchicalNodeParser = HierarchicalNodeParser
    np_mod.get_leaf_nodes = get_leaf_nodes
    np_mod.get_root_nodes = get_root_nodes
    li_core.node_parser = np_mod

    # llama_index.core.retrievers
    retr_mod = types.ModuleType("llama_index.core.retrievers")

    class _StubRetriever:
        async def aretrieve(self, *a, **kw):
            return []

        def retrieve(self, *a, **kw):
            return []

    class AutoMergingRetriever(_StubRetriever):
        def __init__(self, *a, **kw):
            pass

    class QueryFusionRetriever(_StubRetriever):
        def __init__(self, *a, **kw):
            pass

    retr_mod.AutoMergingRetriever = AutoMergingRetriever
    retr_mod.QueryFusionRetriever = QueryFusionRetriever
    li_core.retrievers = retr_mod

    # llama_index.core.storage.{docstore,index_store}
    storage_pkg = types.ModuleType("llama_index.core.storage")
    docstore_mod = types.ModuleType("llama_index.core.storage.docstore")
    indexstore_mod = types.ModuleType("llama_index.core.storage.index_store")

    class SimpleDocumentStore:
        def __init__(self):
            self.docs = {}

        @classmethod
        def from_persist_path(cls, path):
            return cls()

        def add_documents(self, nodes):
            for node in nodes:
                node_id = getattr(node, "id_", None) or getattr(node, "node_id", None) or str(id(node))
                self.docs[node_id] = node

        def delete_document(self, node_id, raise_error=True):
            self.docs.pop(node_id, None)

    class SimpleIndexStore:
        @classmethod
        def from_persist_path(cls, path):
            return cls()

    docstore_mod.SimpleDocumentStore = SimpleDocumentStore
    indexstore_mod.SimpleIndexStore = SimpleIndexStore
    storage_pkg.docstore = docstore_mod
    storage_pkg.index_store = indexstore_mod
    li_core.storage = storage_pkg

    # llama_index.core.vector_stores.types
    vs_pkg = types.ModuleType("llama_index.core.vector_stores")
    vs_types = types.ModuleType("llama_index.core.vector_stores.types")

    class FilterOperator:
        IN = "in"
        EQ = "=="

    class MetadataFilter:
        def __init__(self, key, value, operator=FilterOperator.EQ):
            self.key = key
            self.value = value
            self.operator = operator

    class MetadataFilters:
        def __init__(self, filters):
            self.filters = filters

    vs_types.FilterOperator = FilterOperator
    vs_types.MetadataFilter = MetadataFilter
    vs_types.MetadataFilters = MetadataFilters
    vs_pkg.types = vs_types
    li_core.vector_stores = vs_pkg

    # llama_index.vector_stores.qdrant
    li_vs_pkg = types.ModuleType("llama_index.vector_stores")
    li_vs_qdrant = types.ModuleType("llama_index.vector_stores.qdrant")

    class QdrantVectorStore:
        def __init__(self, client=None, collection_name=None, **kwargs):
            self.client = client
            self.collection_name = collection_name

    li_vs_qdrant.QdrantVectorStore = QdrantVectorStore
    li_vs_pkg.qdrant = li_vs_qdrant

    # llama_index.retrievers.bm25
    li_retr_pkg = types.ModuleType("llama_index.retrievers")
    li_retr_bm25 = types.ModuleType("llama_index.retrievers.bm25")

    class BM25Retriever(_StubRetriever):
        @classmethod
        def from_defaults(cls, nodes, similarity_top_k=10):
            return cls()

    li_retr_bm25.BM25Retriever = BM25Retriever
    li_retr_pkg.bm25 = li_retr_bm25

    _install_stub("llama_index", li_pkg)
    _install_stub("llama_index.core", li_core)
    _install_stub("llama_index.core.schema", schema_mod)
    _install_stub("llama_index.core.llms", llms_mod)
    _install_stub("llama_index.core.llms.callbacks", callbacks_mod)
    _install_stub("llama_index.core.embeddings", embed_mod)
    _install_stub("llama_index.core.node_parser", np_mod)
    _install_stub("llama_index.core.retrievers", retr_mod)
    _install_stub("llama_index.core.storage", storage_pkg)
    _install_stub("llama_index.core.storage.docstore", docstore_mod)
    _install_stub("llama_index.core.storage.index_store", indexstore_mod)
    _install_stub("llama_index.core.vector_stores", vs_pkg)
    _install_stub("llama_index.core.vector_stores.types", vs_types)
    _install_stub("llama_index.vector_stores", li_vs_pkg)
    _install_stub("llama_index.vector_stores.qdrant", li_vs_qdrant)
    _install_stub("llama_index.retrievers", li_retr_pkg)
    _install_stub("llama_index.retrievers.bm25", li_retr_bm25)


def _make_qdrant_stubs() -> None:
    qclient_pkg = types.ModuleType("qdrant_client")
    qhttp_pkg = types.ModuleType("qdrant_client.http")
    qmodels_mod = types.ModuleType("qdrant_client.http.models")

    class QdrantClient:
        def __init__(self, *args, **kwargs):
            self.kwargs = kwargs

        def get_collections(self):
            return SimpleNamespace(collections=[])

        def delete_collection(self, collection_name=None):
            return None

        def delete(self, *args, **kwargs):
            return None

    class FilterSelector:
        def __init__(self, filter=None):
            self.filter = filter

    class Filter:
        def __init__(self, should=None, must=None):
            self.should = should
            self.must = must

    class FieldCondition:
        def __init__(self, key=None, match=None):
            self.key = key
            self.match = match

    class MatchValue:
        def __init__(self, value=None):
            self.value = value

    qmodels_mod.FilterSelector = FilterSelector
    qmodels_mod.Filter = Filter
    qmodels_mod.FieldCondition = FieldCondition
    qmodels_mod.MatchValue = MatchValue
    qhttp_pkg.models = qmodels_mod
    qclient_pkg.QdrantClient = QdrantClient
    qclient_pkg.http = qhttp_pkg

    _install_stub("qdrant_client", qclient_pkg)
    _install_stub("qdrant_client.http", qhttp_pkg)
    _install_stub("qdrant_client.http.models", qmodels_mod)


def _make_sentence_transformers_stubs() -> None:
    st_mod = types.ModuleType("sentence_transformers")

    class CrossEncoder:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, pairs):
            return [0.0 for _ in pairs]

    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            pass

        def get_embedding_dimension(self):
            return 8

        def encode(self, texts, **kwargs):
            return [[0.0] * 8 for _ in texts]

    st_mod.CrossEncoder = CrossEncoder
    st_mod.SentenceTransformer = SentenceTransformer
    _install_stub("sentence_transformers", st_mod)


def load_rag_engine_module():
    """Reload rag_engine + adapters with fresh stubs in place."""
    sys.modules.pop("backend.services.rag_engine", None)
    sys.modules.pop("backend.services.llamaindex_adapters", None)

    _make_llama_index_stubs()
    _make_qdrant_stubs()
    _make_sentence_transformers_stubs()

    return importlib.import_module("backend.services.rag_engine")


def make_service(rag_engine):
    """Build a RAGService bypassing __init__ for unit tests."""
    import threading
    service = object.__new__(rag_engine.RAGService)
    service.embeddings = SimpleNamespace(
        begin_timing_session=lambda: None,
        consume_timing_session=lambda: {"calls": 0, "total_ms": 0.0, "texts": 0},
        model_name="stub",
        model_dim=8,
    )
    service._user_storage = {}
    service._user_storage_lock = threading.Lock()
    service._reranker = None
    return service


def _node_with_score(text, score=None, metadata=None):
    """Construct a LlamaIndex-shaped NodeWithScore for rerank tests."""
    node = SimpleNamespace(text=text, metadata=metadata or {})
    return SimpleNamespace(node=node, score=score)


# ---------------------------------------------------------------------------
# Rerank behavioral tests — target ``_rerank_nodes`` directly. The old tests
# went through ``query`` and mocked ``_hybrid_search``; with retrieval now
# delegated to LlamaIndex, the cleanest target is the helper that owns the
# defensive logic.
# ---------------------------------------------------------------------------


def test_rerank_filters_below_threshold(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)

    class FakeReranker:
        def predict(self, pairs):
            return [0.9, 0.0]

    service.reranker = FakeReranker()
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.5)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 10)

    nodes = [
        _node_with_score("first non-empty passage about plants", metadata={"source": "a.pdf"}),
        _node_with_score("second non-empty passage about dogs", metadata={"source": "b.pdf"}),
    ]

    result = service._rerank_nodes("question about plants", nodes)
    sources = [nws.node.metadata.get("source") for nws in result]

    assert sources == ["a.pdf"]


def test_rerank_returns_input_when_no_nodes():
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    service.reranker = SimpleNamespace(predict=lambda pairs: [])
    assert service._rerank_nodes("q", []) == []


def test_rerank_caps_candidates(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    captured = {"count": 0}

    class FakeReranker:
        def predict(self, pairs):
            captured["count"] = len(pairs)
            return [1.0 for _ in pairs]

    service.reranker = FakeReranker()
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 2)

    nodes = [_node_with_score(f"chunk {i}", metadata={"source": f"p{i}.pdf"}) for i in range(5)]
    service._rerank_nodes("question", nodes)

    assert captured["count"] == 2


def test_rerank_filters_blank_candidates(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    captured = {"pairs": None}

    class FakeReranker:
        def predict(self, pairs):
            captured["pairs"] = pairs
            return [1.0 for _ in pairs]

    service.reranker = FakeReranker()
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 4)

    nodes = [
        _node_with_score("good chunk"),
        _node_with_score("   "),
        _node_with_score(""),
        _node_with_score("another chunk"),
    ]
    service._rerank_nodes("question", nodes)

    assert captured["pairs"] is not None
    assert len(captured["pairs"]) == 2
    assert all(pair[1].strip() for pair in captured["pairs"])


def test_rerank_skipped_when_candidate_limit_non_positive(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    captured = {"called": False}

    class FakeReranker:
        def predict(self, pairs):
            captured["called"] = True
            return [1.0 for _ in pairs]

    service.reranker = FakeReranker()
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 0)

    nodes = [_node_with_score("chunk one")]
    result = service._rerank_nodes("question", nodes)

    assert captured["called"] is False
    assert result == nodes


def test_rerank_uses_stable_sub_batches(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    captured = {"sizes": []}

    class FakeReranker:
        def predict(self, pairs):
            captured["sizes"].append(len(pairs))
            return [float(len(pairs) - i) for i in range(len(pairs))]

    service.reranker = FakeReranker()
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 5)
    monkeypatch.setattr(rag_engine.config, "rerank_batch_size", 2)

    nodes = [_node_with_score(f"chunk {i}") for i in range(5)]
    service._rerank_nodes("question", nodes)

    assert captured["sizes"] == [2, 2, 1]


def test_rerank_uses_plain_question_when_not_zerank(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    captured = {"pairs": None}

    class FakeReranker:
        def predict(self, pairs):
            captured["pairs"] = pairs
            return [1.0 for _ in pairs]

    service.reranker = FakeReranker()
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 1)
    monkeypatch.setattr(rag_engine.config, "reranker_model", "ms-marco/some-model")

    nodes = [_node_with_score("chunk one")]
    service._rerank_nodes(" question ", nodes)

    assert captured["pairs"] == [["question", "chunk one"]]


def test_rerank_falls_back_when_score_count_mismatches(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)

    class BadReranker:
        def predict(self, pairs):
            return [1.0]

    service.reranker = BadReranker()
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 2)

    nodes = [_node_with_score(f"chunk {i}") for i in range(2)]
    result = service._rerank_nodes("question", nodes)

    assert result == nodes


def test_rerank_falls_back_when_scores_are_not_finite(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)

    class NanReranker:
        def predict(self, pairs):
            return [float("nan") for _ in pairs]

    service.reranker = NanReranker()
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 2)

    nodes = [_node_with_score(f"chunk {i}") for i in range(2)]
    result = service._rerank_nodes("question", nodes)

    assert result == nodes


def test_rerank_falls_back_when_threshold_filters_everything(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)

    class FlatReranker:
        def predict(self, pairs):
            return [1.0 for _ in pairs]

    service.reranker = FlatReranker()
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.9)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 2)

    nodes = [_node_with_score(f"chunk {i}") for i in range(2)]
    result = service._rerank_nodes("question", nodes)

    # min-max collapses identical scores to 0.5 → below threshold 0.9 → fallback.
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Query end-to-end behavior tests
# ---------------------------------------------------------------------------


def test_query_returns_no_context_without_calling_llm(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    called = {"value": False}

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None, **kwargs):
            called["value"] = True
            return SimpleNamespace(content="should not be called")

    service.llm = FakeLLM()
    service.reranker = None

    class EmptyRetriever:
        def retrieve(self, q):
            return []

        async def aretrieve(self, q):
            return []

    monkeypatch.setattr(
        service, "_retrieve_candidate_nodes", lambda *a, **kw: EmptyRetriever()
    )

    result = asyncio.run(service.query("question", user_id="sess_1"))

    assert called["value"] is False
    assert result["sources"] == []
    assert "couldn't find enough relevant context" in result["answer"].lower()


def test_query_builds_sources_from_reranked_nodes(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None, **kwargs):
            return SimpleNamespace(content="ok")

    service.llm = FakeLLM()
    service.reranker = None

    nodes = [
        _node_with_score(
            "child a",
            score=0.8,
            metadata={
                "source": "paper-a.pdf",
                "section_title": "Methods",
                "doc_title": "Some Paper",
                "doc_authors": "A. Researcher",
                "parser_type": "pymupdf",
                "content_type": "text",
            },
        ),
    ]

    class StubRetriever:
        def retrieve(self, q):
            return list(nodes)

        async def aretrieve(self, q):
            return list(nodes)

    monkeypatch.setattr(
        service, "_retrieve_candidate_nodes", lambda *a, **kw: StubRetriever()
    )

    result = asyncio.run(service.query("question", user_id="sess_1"))

    assert result["sources"]
    assert result["sources"][0]["source"] == "paper-a.pdf"
    assert result["sources"][0]["section"] == "Methods"
    assert result["answer"] == "ok"


# ---------------------------------------------------------------------------
# zerank version compatibility
# ---------------------------------------------------------------------------


def test_zerank_runtime_compatibility_rejects_transformers_drift(monkeypatch):
    rag_engine = load_rag_engine_module()

    monkeypatch.setattr(
        rag_engine.importlib_metadata,
        "version",
        lambda name: {
            "sentence-transformers": "5.4.1",
            "transformers": "4.58.0",
        }[name],
    )

    is_compatible, reason = rag_engine._zerank_runtime_compatible()

    assert is_compatible is False
    assert "transformers" in reason


def test_zerank_runtime_compatibility_accepts_expected_versions(monkeypatch):
    rag_engine = load_rag_engine_module()

    monkeypatch.setattr(
        rag_engine.importlib_metadata,
        "version",
        lambda name: {
            "sentence-transformers": "5.4.1",
            "transformers": "4.57.1",
        }[name],
    )

    is_compatible, reason = rag_engine._zerank_runtime_compatible()

    assert is_compatible is True
    assert reason == ""


# ---------------------------------------------------------------------------
# Singleton + peek tests
# ---------------------------------------------------------------------------


def test_lazy_get_rag_service_initializes_once(monkeypatch):
    rag_engine = load_rag_engine_module()

    class DummyService:
        pass

    created = {"count": 0}

    def factory():
        created["count"] += 1
        return DummyService()

    monkeypatch.setattr(rag_engine, "RAGService", factory)
    rag_engine._rag_service = None

    first = rag_engine.get_rag_service()
    second = rag_engine.get_rag_service()

    assert created["count"] == 1
    assert first is second


def test_peek_rag_service_does_not_initialize(monkeypatch):
    rag_engine = load_rag_engine_module()
    rag_engine._rag_service = None

    assert rag_engine.peek_rag_service() is None


def test_ragservice_does_not_retry_zerank_on_cpu_when_versions_are_incompatible(monkeypatch):
    rag_engine = load_rag_engine_module()

    monkeypatch.setattr(rag_engine, "get_optimal_device", lambda: "mps")
    monkeypatch.setattr(
        rag_engine.importlib_metadata,
        "version",
        lambda name: {
            "sentence-transformers": "5.4.1",
            "transformers": "4.58.0",
        }[name],
    )

    calls = {"count": 0}

    class FailingCrossEncoder:
        def __init__(self, *args, **kwargs):
            calls["count"] += 1

    monkeypatch.setattr(rag_engine, "CrossEncoder", FailingCrossEncoder)
    monkeypatch.setattr(rag_engine.config, "reranker_model", "zeroentropy/zerank-2")
    monkeypatch.setattr(
        rag_engine.RAGService,
        "_configure_llama_index_settings",
        lambda self: None,
    )

    service = rag_engine.RAGService()

    assert calls["count"] == 0
    assert service.reranker is None


# ---------------------------------------------------------------------------
# Indexing pipeline tests
# ---------------------------------------------------------------------------


def test_process_and_index_pdfs_with_texts_replaces_existing_sources(monkeypatch):
    """Re-uploading a source should call ``_delete_existing_sources`` first."""
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    Document = rag_engine.Document

    delete_calls = []
    persisted = {"called": False}

    service._process_pdf = lambda path, user_id="default", parser_type="pymupdf": (
        [Document(text="new chunk", metadata={"source": "paper.pdf", "content_type": "text"})],
        "extracted body text",
    )
    service._delete_existing_sources = lambda user_id, source_names: delete_calls.append(
        (user_id, list(source_names))
    )

    class FakeDocstore:
        def __init__(self):
            self.added = []

        def add_documents(self, nodes):
            self.added.extend(nodes)

    class FakeStorage:
        def __init__(self):
            self.collection_name = "user_test_xx"
            self.persist_dir = "/tmp/test"
            self.storage_context = SimpleNamespace(docstore=FakeDocstore())
            self.vector_store = SimpleNamespace()

    storage = FakeStorage()
    service._get_user_storage = lambda user_id: storage
    service._persist_user_storage = lambda user_id: persisted.update({"called": True})

    indexed_files, extracted_texts = service.process_and_index_pdfs_with_texts(
        ["C:/tmp/paper.pdf"],
        user_id="sess_1",
    )

    assert indexed_files == ["paper.pdf"]
    assert extracted_texts == {"paper.pdf": "extracted body text"}
    assert delete_calls == [("sess_1", ["paper.pdf"])]
    assert persisted["called"] is True


def test_sanitize_metadata_dict_coerces_complex_values():
    """Lists / dicts in metadata must be coerced to JSON strings before storage."""
    rag_engine = load_rag_engine_module()

    sanitized = rag_engine._sanitize_metadata_dict(
        {
            "source": "paper.pdf",
            "publication_flags": ["OPEN ACCESS"],
            "nested": {"a": 1},
            "year": 2024,
            "open": True,
            "missing": None,
        }
    )

    assert sanitized["source"] == "paper.pdf"
    assert sanitized["publication_flags"] == '["OPEN ACCESS"]'
    assert sanitized["nested"] == '{"a": 1}'
    assert sanitized["year"] == 2024
    assert sanitized["open"] is True
    assert sanitized["missing"] == ""


def test_process_pdf_pymupdf_returns_documents_with_parser_type(monkeypatch):
    """PyMuPDF path produces LlamaIndex Documents with parser_type='pymupdf'."""
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)

    service._extract_pdf_metadata = lambda path: {
        "title": "Paper Title",
        "authors": "",
        "doi": "",
        "journal": "",
    }
    service._extract_with_pymupdf = lambda path: (
        "Introduction\n\nSome body text for chunking.",
        [],
    )
    service._detect_sections = lambda text: [
        {"title": "Introduction", "text": "Some body text for chunking."}
    ]

    documents, full_text = service._process_pdf(
        "C:/tmp/paper.pdf",
        user_id="sess_1",
        parser_type="pymupdf",
    )

    assert full_text == "Introduction\n\nSome body text for chunking."
    assert len(documents) >= 1
    assert all(doc.metadata["parser_type"] == "pymupdf" for doc in documents)
    assert all(doc.metadata["source"] == "paper.pdf" for doc in documents)


def test_build_cch_header_handles_empty_inputs():
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    assert service._build_cch_header("", "") == ""
    assert service._build_cch_header("Title", "") == "Title\n\n"
    assert service._build_cch_header("Title", "Methods") == "Title > Methods\n\n"
    assert service._build_cch_header("", "Methods") == "Methods\n\n"
    # "Start" sentinel section title is suppressed.
    assert service._build_cch_header("Title", "Start") == "Title\n\n"


# ---------------------------------------------------------------------------
# OllamaLLM timeout
# ---------------------------------------------------------------------------


def test_ollama_invoke_enforces_timeout_budget(monkeypatch):
    rag_engine = load_rag_engine_module()

    class HangingClient:
        async def post(self, *args, **kwargs):
            await asyncio.sleep(0.05)

    async def get_client():
        return HangingClient()

    monkeypatch.setattr(rag_engine.HttpClientManager, "get_client", get_client)
    llm = rag_engine.OllamaLLM(
        base_url="http://localhost:11434",
        model="llama3.1:8b",
        provider="ollama",
    )

    try:
        asyncio.run(llm.invoke(prompt="hello", max_retries=1, timeout_seconds=0.01))
    except Exception as exc:
        assert exc.__class__.__name__ == "RAGLLMTimeoutError"
    else:
        raise AssertionError("Expected RAGLLMTimeoutError")


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------


def test_get_optimal_device_respects_env_override(monkeypatch):
    rag_engine = load_rag_engine_module()
    monkeypatch.setenv("RAG_DEVICE", "cuda")
    assert rag_engine.get_optimal_device() == "cuda"
    monkeypatch.setenv("RAG_DEVICE", "mps")
    assert rag_engine.get_optimal_device() == "mps"
    monkeypatch.setenv("RAG_DEVICE", "cpu")
    assert rag_engine.get_optimal_device() == "cpu"


def test_get_optimal_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    rag_engine = load_rag_engine_module()
    monkeypatch.delenv("RAG_DEVICE", raising=False)

    class FakeCuda:
        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 1)

    class FakeMps:
        is_available = staticmethod(lambda: True)

    class FakeBackends:
        mps = FakeMps()

    class TorchCuda:
        cuda = FakeCuda()
        backends = FakeBackends()

    class TorchMps:
        cuda = type("FakeCuda", (), {"is_available": staticmethod(lambda: False), "device_count": staticmethod(lambda: 0)})()
        backends = FakeBackends()

    class TorchNone:
        cuda = type("FakeCuda", (), {"is_available": staticmethod(lambda: False), "device_count": staticmethod(lambda: 0)})()
        backends = type("FakeBackends", (), {"mps": type("FakeMps", (), {"is_available": staticmethod(lambda: False)})()})()

    monkeypatch.setitem(sys.modules, "torch", TorchCuda())
    assert rag_engine.get_optimal_device() == "cuda"

    monkeypatch.setitem(sys.modules, "torch", TorchMps())
    assert rag_engine.get_optimal_device() == "mps"

    monkeypatch.setitem(sys.modules, "torch", TorchNone())
    assert rag_engine.get_optimal_device() == "cpu"


def test_build_cuda_model_kwargs_with_flash_attn_and_multi_gpu(monkeypatch):
    rag_engine = load_rag_engine_module()

    fake_flash = types.ModuleType("flash_attn")
    monkeypatch.setitem(sys.modules, "flash_attn", fake_flash)

    class FakeMultiGpu:
        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 2)

    monkeypatch.setitem(sys.modules, "torch", type("T", (), {"cuda": FakeMultiGpu()})())

    kwargs = rag_engine._build_cuda_model_kwargs(enable_flash_attn=True, enable_multi_gpu=True)
    assert kwargs["torch_dtype"] == "auto"
    assert kwargs["attn_implementation"] == "flash_attention_2"
    assert kwargs["device_map"] == "auto"


def test_build_cuda_model_kwargs_without_flash_attn(monkeypatch):
    rag_engine = load_rag_engine_module()
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    class FakeSingleGpu:
        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 1)

    monkeypatch.setitem(sys.modules, "torch", type("T", (), {"cuda": FakeSingleGpu()})())

    kwargs = rag_engine._build_cuda_model_kwargs(enable_flash_attn=True, enable_multi_gpu=False)
    assert kwargs["torch_dtype"] == "auto"
    assert "attn_implementation" not in kwargs
    assert "device_map" not in kwargs


def test_build_cuda_model_kwargs_disabled():
    rag_engine = load_rag_engine_module()
    kwargs = rag_engine._build_cuda_model_kwargs(enable_flash_attn=False, enable_multi_gpu=False)
    assert kwargs == {"torch_dtype": "auto"}


def test_get_runtime_diagnostics_reports_slurm_and_cuda(monkeypatch):
    rag_engine = load_rag_engine_module()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("SLURM_JOB_ID", "12345")
    monkeypatch.setenv("SLURM_NODELIST", "gpu001")
    monkeypatch.setenv("SLURM_LOCALID", "0")

    class FakeCuda:
        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 2)
        get_device_name = staticmethod(lambda index: f"GPU-{index}")

    class FakeBackends:
        mps = type("FakeMps", (), {"is_available": staticmethod(lambda: False)})()

    monkeypatch.setitem(sys.modules, "torch", type("T", (), {"cuda": FakeCuda(), "backends": FakeBackends()})())

    diagnostics = rag_engine.get_runtime_diagnostics()
    assert diagnostics["selected_device"] == "cuda"
    assert diagnostics["cuda_visible_devices"] == "2"
    assert diagnostics["slurm_job_id"] == "12345"
    assert diagnostics["slurm_nodelist"] == "gpu001"
    assert diagnostics["slurm_localid"] == "0"
    assert diagnostics["cuda_available"] is True
    assert diagnostics["cuda_device_count"] == 2
    assert diagnostics["cuda_device_names"] == ["GPU-0", "GPU-1"]


def test_format_phase_timings_is_stable():
    rag_engine = load_rag_engine_module()
    formatted = rag_engine._format_phase_timings({
        "parse": 12.34,
        "embed": 56.78,
        "store_overhead": 9.87,
    })
    assert formatted == "parse=12.3ms, embed=56.8ms, store_overhead=9.9ms"
