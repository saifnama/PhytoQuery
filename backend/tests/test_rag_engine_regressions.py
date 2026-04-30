import asyncio
import importlib
import sys
import types
from types import SimpleNamespace


def load_rag_engine_module():
    sys.modules.pop("backend.services.rag_engine", None)

    chroma_mod = types.ModuleType("langchain_chroma")

    class Chroma:
        pass

    chroma_mod.Chroma = Chroma

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

    sys.modules["langchain_chroma"] = chroma_mod
    sys.modules["langchain_text_splitters"] = splitters_mod
    sys.modules["langchain_experimental"] = experimental_pkg
    sys.modules["langchain_experimental.text_splitter"] = experimental_splitter_mod
    sys.modules["sentence_transformers"] = sentence_transformers_mod
    sys.modules["langchain_core"] = core_pkg
    sys.modules["langchain_core.documents"] = documents_mod

    return importlib.import_module("backend.services.rag_engine")


def make_service(rag_engine):
    service = object.__new__(rag_engine.RAGService)
    service._get_user_collection = lambda user_id: SimpleNamespace()
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


def test_process_and_index_pdfs_with_texts_replaces_existing_sources(monkeypatch):
    rag_engine = load_rag_engine_module()
    service = make_service(rag_engine)
    doc_cls = rag_engine.Document

    class FakeCollection:
        def __init__(self):
            self.deleted_ids = []

        def get(self, where=None, include=None):
            if where == {"source": "paper.pdf"}:
                return {"ids": ["old-chunk-1"], "metadatas": [{"parent_id": "old-parent"}]}
            return {"ids": [], "metadatas": []}

        def delete(self, ids=None):
            self.deleted_ids.append(list(ids or []))

    class FakeVectorstore:
        def __init__(self):
            self._collection = FakeCollection()
            self.added_documents = None

        def add_documents(self, docs):
            self.added_documents = docs

    vectorstore = FakeVectorstore()
    cleanup_calls = []
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
    )

    indexed_files, extracted_texts = service.process_and_index_pdfs_with_texts(
        ["C:/tmp/paper.pdf"],
        user_id="sess_1",
    )

    assert indexed_files == ["paper.pdf"]
    assert extracted_texts == {"paper.pdf": "extracted body text"}
    assert vectorstore._collection.deleted_ids == [["old-chunk-1"]]
    assert vectorstore.added_documents is not None
    assert cleanup_calls == ["sess_1"]


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
