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
    service.embeddings = SimpleNamespace(
        begin_timing_session=lambda: None,
        consume_timing_session=lambda: {"calls": 0, "total_ms": 0.0, "texts": 0},
    )
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

    documents, full_text = service._process_pdf(
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
