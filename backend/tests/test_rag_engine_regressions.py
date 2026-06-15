import asyncio
import importlib
import sys
import types
from types import SimpleNamespace


def load_rag_engine_module():
    sys.modules.pop("backend.services.rag_engine", None)

    # langchain_qdrant stub — only QdrantVectorStore is imported lazily
    # by rag_engine; tests bypass it via mocked vectorstores.
    qdrant_lc_mod = types.ModuleType("langchain_qdrant")

    class QdrantVectorStore:
        def __init__(self, client=None, collection_name=None, embedding=None, **kwargs):
            self.client = client
            self.collection_name = collection_name
            self.embedding = embedding

    qdrant_lc_mod.QdrantVectorStore = QdrantVectorStore

    # qdrant_client stub — needed for QdrantClient + qmodels filter classes.
    qclient_pkg = types.ModuleType("qdrant_client")
    qhttp_pkg = types.ModuleType("qdrant_client.http")
    qmodels_mod = types.ModuleType("qdrant_client.http.models")

    class QdrantClient:
        def __init__(self, *args, **kwargs):
            pass

        def get_collections(self):
            return SimpleNamespace(collections=[])

        def get_collection(self, *args, **kwargs):
            raise Exception("not found")

        def create_collection(self, *args, **kwargs):
            return None

        def delete_collection(self, *args, **kwargs):
            return None

        def delete(self, *args, **kwargs):
            return None

        def scroll(self, *args, **kwargs):
            return ([], None)

    class _QStub:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class Distance:
        COSINE = "Cosine"
        DOT = "Dot"
        EUCLID = "Euclid"

    qmodels_mod.VectorParams = _QStub
    qmodels_mod.SparseVectorParams = _QStub
    qmodels_mod.SparseIndexParams = _QStub
    qmodels_mod.Distance = Distance
    qmodels_mod.Filter = _QStub
    qmodels_mod.FieldCondition = _QStub
    qmodels_mod.MatchValue = _QStub
    qmodels_mod.FilterSelector = _QStub
    qmodels_mod.PointIdsList = _QStub
    qhttp_pkg.models = qmodels_mod
    qclient_pkg.QdrantClient = QdrantClient
    qclient_pkg.http = qhttp_pkg

    splitters_mod = types.ModuleType("langchain_text_splitters")

    class RecursiveCharacterTextSplitter:
        def __init__(self, *args, **kwargs):
            pass

        def split_text(self, text):
            return [text]

    class MarkdownTextSplitter(RecursiveCharacterTextSplitter):
        pass

    splitters_mod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    splitters_mod.MarkdownTextSplitter = MarkdownTextSplitter

    experimental_pkg = types.ModuleType("langchain_experimental")
    experimental_splitter_mod = types.ModuleType("langchain_experimental.text_splitter")

    class SemanticChunker:
        def __init__(self, *args, **kwargs):
            pass

        def split_text(self, text):
            return [text]

    experimental_splitter_mod.SemanticChunker = SemanticChunker
    experimental_pkg.text_splitter = experimental_splitter_mod

    sentence_transformers_mod = types.ModuleType("sentence_transformers")

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

    sentence_transformers_mod.CrossEncoder = CrossEncoder
    sentence_transformers_mod.SentenceTransformer = SentenceTransformer

    core_pkg = types.ModuleType("langchain_core")
    documents_mod = types.ModuleType("langchain_core.documents")

    class Document:
        def __init__(self, page_content, metadata=None):
            self.page_content = page_content
            self.metadata = metadata or {}

    documents_mod.Document = Document
    core_pkg.documents = documents_mod

    # langchain_core.embeddings stub — PhytoQueryEmbeddings now inherits
    # from this so langchain-qdrant's isinstance() check passes.
    embeddings_mod = types.ModuleType("langchain_core.embeddings")

    class Embeddings:
        def embed_documents(self, texts):
            raise NotImplementedError

        def embed_query(self, text):
            raise NotImplementedError

    embeddings_mod.Embeddings = Embeddings
    core_pkg.embeddings = embeddings_mod

    sys.modules["langchain_qdrant"] = qdrant_lc_mod
    sys.modules["qdrant_client"] = qclient_pkg
    sys.modules["qdrant_client.http"] = qhttp_pkg
    sys.modules["qdrant_client.http.models"] = qmodels_mod
    sys.modules["langchain_text_splitters"] = splitters_mod
    sys.modules["langchain_experimental"] = experimental_pkg
    sys.modules["langchain_experimental.text_splitter"] = experimental_splitter_mod
    sys.modules["sentence_transformers"] = sentence_transformers_mod
    sys.modules["langchain_core"] = core_pkg
    sys.modules["langchain_core.documents"] = documents_mod
    sys.modules["langchain_core.embeddings"] = embeddings_mod

    return importlib.import_module("backend.services.rag_engine")


def make_service(rag_engine):
    import threading
    service = object.__new__(rag_engine.RAGService)
    service._get_user_collection = lambda user_id: SimpleNamespace(collection_name="user_test_xx")
    service.embeddings = SimpleNamespace(
        begin_timing_session=lambda: None,
        consume_timing_session=lambda: {"calls": 0, "total_ms": 0.0, "texts": 0},
        # ``_get_collection_suffix`` now calls ``_ensure_model_loaded``
        # before reading model_name/model_dim, so the test stub needs
        # a no-op for it. (Real ``PhytoQueryEmbeddings`` lazy-loads
        # on first call; fakes have nothing to load.)
        _ensure_model_loaded=lambda: None,
        model_name="stub-model",
        model_dim=8,
    )
    service._vectorstore_cache = {}
    service._qdrant_client = None
    service._qdrant_lock = threading.Lock()
    return service


def test_query_filters_rerank_threshold_per_result(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document

    class FakeReranker:
        def predict(self, pairs):
            return [0.9, 0.0]

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            return SimpleNamespace(content="ok")

    service.reranker = FakeReranker()
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: {
        "p1": "parent text one",
        "p2": "parent text two",
    }.get(parent_id, "")

    search_results = [
        {
            "doc": doc_cls(
                "child text one",
                {
                    "content_type": "text",
                    "parent_id": "p1",
                    "source": "paper-a.pdf",
                    "section_title": "Intro",
                    "parser_type": "pymupdf",
                },
            )
        },
        {
            "doc": doc_cls(
                "child text two",
                {
                    "content_type": "text",
                    "parent_id": "p2",
                    "source": "paper-b.pdf",
                    "section_title": "Methods",
                    "parser_type": "pymupdf",
                },
            )
        },
    ]

    monkeypatch.setattr(service, "_hybrid_search", lambda *args, **kwargs: search_results)
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.5)

    result = asyncio.run(service.query("question", user_id="sess_1"))

    assert [source["source"] for source in result["sources"]] == ["paper-a.pdf"]


def test_query_returns_no_context_without_calling_llm(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    called = {"value": False}

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            called["value"] = True
            return SimpleNamespace(content="should not be called")

    service.reranker = None
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: ""
    monkeypatch.setattr(service, "_hybrid_search", lambda *args, **kwargs: [])

    result = asyncio.run(service.query("question", user_id="sess_1"))

    assert called["value"] is False
    assert result["sources"] == []
    assert "couldn't find enough relevant context" in result["answer"].lower()


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
    # Force a zerank model so the version check fires (the assertion is
    # specifically that incompatible zerank versions skip CPU retry).
    monkeypatch.setattr(rag_engine.config, "reranker_model", "zeroentropy/zerank-2")

    service = rag_engine.RAGService()

    assert calls["count"] == 0
    assert service.reranker is None


def test_process_and_index_pdfs_with_texts_replaces_existing_sources(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document

    class FakeVectorstore:
        def __init__(self):
            self.collection_name = "user_test_xx"
            self.added_documents = None

        def add_documents(self, docs):
            self.added_documents = docs

    vectorstore = FakeVectorstore()
    cleanup_calls = []
    delete_calls = []

    class FakeQdrantClient:
        def get_collections(self):
            return SimpleNamespace(collections=[])

        def get_collection(self, *a, **kw):
            return SimpleNamespace()

        def delete(self, collection_name=None, points_selector=None, **kw):
            # Qdrant-style filter delete — record the collection + selector
            # for assertion. The points_selector is a FilterSelector wrapping
            # a Filter with `should` clauses on metadata.source.
            delete_calls.append({
                "collection_name": collection_name,
                "selector": points_selector,
            })

        def delete_collection(self, *a, **kw):
            return None

        def scroll(self, *a, **kw):
            return ([], None)

    fake_client = FakeQdrantClient()
    service._get_qdrant_client = lambda: fake_client
    service._get_user_collection_name = lambda user_id: "user_test_xx"
    service._get_user_collection = lambda user_id: vectorstore
    service._invalidate_user_collection = lambda user_id: None
    service._cleanup_parent_store = lambda user_id: cleanup_calls.append(user_id)
    service._process_pdf = lambda path, user_id="default", parser_type="pymupdf": (
        [
            doc_cls(
                "new chunk",
                {
                    "source": "paper.pdf",
                    "chunk_id": "paper.pdf_0",
                    "content_type": "text",
                },
            )
        ],
        "extracted body text",
        [],
    )

    indexed_files, extracted_texts = service.process_and_index_pdfs_with_texts(
        ["C:/tmp/paper.pdf"],
        user_id="sess_1",
    )

    assert indexed_files == ["paper.pdf"]
    assert extracted_texts == {"paper.pdf": "extracted body text"}
    # Qdrant filter-delete fired with the right collection name.
    assert any(
        call["collection_name"] == "user_test_xx" for call in delete_calls
    )
    assert vectorstore.added_documents is not None
    assert cleanup_calls == ["sess_1"]


def test_process_pdf_pymupdf_uses_simple_chunking_not_semantic(monkeypatch):
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
        [{"content": "| A | B |\n| 1 | 2 |", "page": 1}],
    )
    service._detect_sections = lambda text: [
        {"title": "Introduction", "text": "Some body text for chunking."}
    ]
    service._add_parents = lambda user_id, parents: None
    service._deduplicate_chunks = lambda chunks: chunks

    called = {"semantic": False}

    def fail_if_semantic_used(text):
        called["semantic"] = True
        raise AssertionError("PyMuPDF fast path should not use semantic chunking")

    service._split_semantic_children = fail_if_semantic_used

    documents, full_text, parent_chunks = service._process_pdf(
        "C:/tmp/paper.pdf",
        user_id="sess_1",
        parser_type="pymupdf",
    )

    assert full_text == "Introduction\n\nSome body text for chunking."
    assert called["semantic"] is False
    assert len(documents) >= 1
    assert all(doc.metadata["parser_type"] == "pymupdf" for doc in documents)


def test_process_and_index_pdfs_with_texts_sanitizes_complex_metadata(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document

    class FakeCollection:
        def get(self, where=None, include=None):
            return {"ids": [], "metadatas": []}

    class FakeVectorstore:
        def __init__(self):
            self._collection = FakeCollection()
            self.added_documents = None

        def add_documents(self, docs):
            self.added_documents = docs
            for doc in docs:
                for value in doc.metadata.values():
                    assert isinstance(value, (str, int, float, bool)) or value is None

    vectorstore = FakeVectorstore()
    service._get_user_collection = lambda user_id: vectorstore
    service._invalidate_user_collection = lambda user_id: None
    service._cleanup_parent_store = lambda user_id: None
    service._process_pdf = lambda path, user_id="default", parser_type="pymupdf": (
        [
            doc_cls(
                "new chunk",
                {
                    "source": "paper.pdf",
                    "chunk_id": "paper.pdf_0",
                    "content_type": "text",
                    "publication_flags": ["OPEN ACCESS"],
                    "nested": {"a": 1},
                },
            )
        ],
        "extracted body text",
        [],
    )

    indexed_files, extracted_texts = service.process_and_index_pdfs_with_texts(
        ["C:/tmp/paper.pdf"],
        user_id="sess_1",
    )

    assert indexed_files == ["paper.pdf"]
    assert extracted_texts == {"paper.pdf": "extracted body text"}
    assert vectorstore.added_documents is not None


def test_query_caps_rerank_candidates(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document
    captured = {"count": 0}

    class FakeReranker:
        def predict(self, pairs):
            captured["count"] = len(pairs)
            return [1.0 for _ in pairs]

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            return SimpleNamespace(content="ok")

    service.reranker = FakeReranker()
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: f"parent::{parent_id}"
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 2)
    monkeypatch.setattr(
        service,
        "_hybrid_search",
        lambda *args, **kwargs: [
            {
                "doc": doc_cls(
                    f"chunk {i}",
                    {
                        "content_type": "text",
                        "parent_id": f"p{i}",
                        "source": f"paper-{i}.pdf",
                        "section_title": "Intro",
                        "parser_type": "pymupdf",
                    },
                )
            }
            for i in range(5)
        ],
    )

    asyncio.run(service.query("question", user_id="sess_1"))

    assert captured["count"] == 2


def test_query_filters_blank_rerank_candidates(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document
    captured = {"pairs": None}

    class FakeReranker:
        def predict(self, pairs):
            captured["pairs"] = pairs
            return [1.0 for _ in pairs]

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            return SimpleNamespace(content="ok")

    service.reranker = FakeReranker()
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: f"parent::{parent_id}"
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 4)
    monkeypatch.setattr(
        service,
        "_hybrid_search",
        lambda *args, **kwargs: [
            {
                "doc": doc_cls(
                    content,
                    {
                        "content_type": "text",
                        "parent_id": f"p{i}",
                        "source": f"paper-{i}.pdf",
                        "section_title": "Intro",
                        "parser_type": "pymupdf",
                    },
                )
            }
            for i, content in enumerate(["good chunk", "   ", "", "another chunk"])
        ],
    )

    asyncio.run(service.query("question", user_id="sess_1"))

    assert captured["pairs"] is not None
    assert len(captured["pairs"]) == 2
    assert all(pair[1].strip() for pair in captured["pairs"])


def test_query_skips_rerank_when_candidate_limit_is_non_positive(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document
    captured = {"called": False}

    class FakeReranker:
        def predict(self, pairs):
            captured["called"] = True
            return [1.0 for _ in pairs]

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            return SimpleNamespace(content="ok")

    service.reranker = FakeReranker()
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: f"parent::{parent_id}"
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 0)
    monkeypatch.setattr(
        service,
        "_hybrid_search",
        lambda *args, **kwargs: [
            {
                "doc": doc_cls(
                    "chunk one",
                    {
                        "content_type": "text",
                        "parent_id": "p1",
                        "source": "paper-1.pdf",
                        "section_title": "Intro",
                    },
                )
            }
        ],
    )

    result = asyncio.run(service.query("question", user_id="sess_1"))

    assert captured["called"] is False
    assert result["sources"]


def test_query_reranks_in_stable_sub_batches(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document
    captured = {"batch_sizes": []}

    class FakeReranker:
        def predict(self, pairs):
            captured["batch_sizes"].append(len(pairs))
            return [float(len(pairs) - i) for i in range(len(pairs))]

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            return SimpleNamespace(content="ok")

    service.reranker = FakeReranker()
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: f"parent::{parent_id}"
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 5)
    monkeypatch.setattr(rag_engine.config, "rerank_batch_size", 2)
    monkeypatch.setattr(
        service,
        "_hybrid_search",
        lambda *args, **kwargs: [
            {
                "doc": doc_cls(
                    f"chunk {i}",
                    {
                        "content_type": "text",
                        "parent_id": f"p{i}",
                        "source": f"paper-{i}.pdf",
                        "section_title": "Intro",
                    },
                )
            }
            for i in range(5)
        ],
    )

    asyncio.run(service.query("question", user_id="sess_1"))

    assert captured["batch_sizes"] == [2, 2, 1]


def test_query_uses_plain_question_for_zerank(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document
    captured = {"pairs": None}

    class FakeReranker:
        def predict(self, pairs):
            captured["pairs"] = pairs
            return [1.0 for _ in pairs]

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            return SimpleNamespace(content="ok")

    service.reranker = FakeReranker()
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: f"parent::{parent_id}"
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 1)
    monkeypatch.setattr(
        service,
        "_hybrid_search",
        lambda *args, **kwargs: [
            {
                "doc": doc_cls(
                    "chunk one",
                    {
                        "content_type": "text",
                        "parent_id": "p1",
                        "source": "paper-1.pdf",
                        "section_title": "Intro",
                    },
                )
            }
        ],
    )

    asyncio.run(service.query(" question ", user_id="sess_1"))

    assert captured["pairs"] == [["question", "chunk one"]]


def test_query_falls_back_when_rerank_score_count_mismatches_pairs(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document

    class BadReranker:
        def predict(self, pairs):
            return [1.0]

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            return SimpleNamespace(content="ok")

    service.reranker = BadReranker()
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: f"parent::{parent_id}"
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 2)
    monkeypatch.setattr(
        service,
        "_hybrid_search",
        lambda *args, **kwargs: [
            {
                "doc": doc_cls(
                    f"chunk {i}",
                    {
                        "content_type": "text",
                        "parent_id": f"p{i}",
                        "source": f"paper-{i}.pdf",
                        "section_title": "Intro",
                    },
                )
            }
            for i in range(2)
        ],
    )

    result = asyncio.run(service.query("question", user_id="sess_1"))

    assert result["sources"]


def test_query_falls_back_when_rerank_scores_are_not_finite(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document

    class BadReranker:
        def predict(self, pairs):
            return [float("nan") for _ in pairs]

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            return SimpleNamespace(content="ok")

    service.reranker = BadReranker()
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: f"parent::{parent_id}"
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 2)
    monkeypatch.setattr(
        service,
        "_hybrid_search",
        lambda *args, **kwargs: [
            {
                "doc": doc_cls(
                    f"chunk {i}",
                    {
                        "content_type": "text",
                        "parent_id": f"p{i}",
                        "source": f"paper-{i}.pdf",
                        "section_title": "Intro",
                    },
                )
            }
            for i in range(2)
        ],
    )

    result = asyncio.run(service.query("question", user_id="sess_1"))

    assert result["sources"]
    assert all(source["source"] != "paper-blank.pdf" for source in result["sources"])


def test_query_falls_back_to_filtered_candidates_instead_of_raw_search_results(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document

    class BadReranker:
        def predict(self, pairs):
            return [float("nan") for _ in pairs]

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            return SimpleNamespace(content="ok")

    service.reranker = BadReranker()
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: f"parent::{parent_id}"
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.0)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 3)
    monkeypatch.setattr(
        service,
        "_hybrid_search",
        lambda *args, **kwargs: [
            {
                "doc": doc_cls(
                    content,
                    {
                        "content_type": "text",
                        "parent_id": f"p{i}",
                        "source": source,
                        "section_title": "Intro",
                    },
                )
            }
            for i, (content, source) in enumerate(
                [
                    ("good chunk", "paper-good-1.pdf"),
                    ("   ", "paper-blank.pdf"),
                    ("another chunk", "paper-good-2.pdf"),
                ]
            )
        ],
    )

    result = asyncio.run(service.query("question", user_id="sess_1"))

    assert result["sources"]
    assert all(source["source"] != "paper-blank.pdf" for source in result["sources"])


def test_query_falls_back_when_threshold_filters_everything(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document

    class FlatReranker:
        def predict(self, pairs):
            return [1.0 for _ in pairs]

    class FakeLLM:
        async def invoke(self, messages=None, prompt=None):
            return SimpleNamespace(content="ok")

    service.reranker = FlatReranker()
    service.llm = FakeLLM()
    service._get_parent_text = lambda parent_id, user_id: f"parent::{parent_id}"
    monkeypatch.setattr(rag_engine.config, "rerank_threshold", 0.9)
    monkeypatch.setattr(rag_engine.config, "rerank_candidate_k", 2)
    monkeypatch.setattr(
        service,
        "_hybrid_search",
        lambda *args, **kwargs: [
            {
                "doc": doc_cls(
                    f"chunk {i}",
                    {
                        "content_type": "text",
                        "parent_id": f"p{i}",
                        "source": f"paper-{i}.pdf",
                        "section_title": "Intro",
                    },
                )
            }
            for i in range(2)
        ],
    )

    result = asyncio.run(service.query("question", user_id="sess_1"))

    assert result["sources"]


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


# --- Device Detection Tests ---

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

    # Simulate torch module with varying availability
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

    # CUDA available
    monkeypatch.setitem(sys.modules, "torch", TorchCuda())
    assert rag_engine.get_optimal_device() == "cuda"

    # MPS only
    monkeypatch.setitem(sys.modules, "torch", TorchMps())
    assert rag_engine.get_optimal_device() == "mps"

    # Neither
    monkeypatch.setitem(sys.modules, "torch", TorchNone())
    assert rag_engine.get_optimal_device() == "cpu"


def test_build_cuda_model_kwargs_with_flash_attn_and_multi_gpu(monkeypatch):
    rag_engine = load_rag_engine_module()

    # Simulate flash-attn available
    fake_flash = types.ModuleType("flash_attn")
    monkeypatch.setitem(sys.modules, "flash_attn", fake_flash)

    # Simulate 2 GPUs
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

    # No flash-attn module
    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)

    class FakeSingleGpu:
        is_available = staticmethod(lambda: True)
        device_count = staticmethod(lambda: 1)

    monkeypatch.setitem(sys.modules, "torch", type("T", (), {"cuda": FakeSingleGpu()})())

    kwargs = rag_engine._build_cuda_model_kwargs(enable_flash_attn=True, enable_multi_gpu=False)
    assert kwargs["torch_dtype"] == "auto"
    assert "attn_implementation" not in kwargs
    assert "device_map" not in kwargs


def test_build_cuda_model_kwargs_disabled(monkeypatch):
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
        "parse_and_chunk": 12.34,
        "embed": 56.78,
        "chroma_overhead": 9.87,
    })
    assert formatted == "parse_and_chunk=12.3ms, embed=56.8ms, chroma_overhead=9.9ms"
