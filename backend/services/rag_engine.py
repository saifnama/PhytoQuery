import os
# Suppress transformers tokenizer sequence length warnings globally
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import re
import gc
import time
import hashlib
import json
import asyncio
import logging
import threading
import warnings

# Local Qdrant doesn't support payload indexes; LlamaIndex still tries to
# create them under enable_hybrid=True and emits a warning every time.
# Filtering still works (Qdrant scans instead), so the warning is benign
# noise at our scale.
warnings.filterwarnings(
    "ignore",
    message=".*Payload indexes have no effect in the local Qdrant.*",
)
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from backend.core.http_client import HttpClientManager
from backend.core.rag_storage import delete_user_upload_file, delete_user_uploads

# ---------------------------------------------------------------------------
# HEAVY IMPORTS ARE DEFERRED (lazy) to avoid loading PyTorch / transformers
# into RAM at server startup.  Each class is imported locally inside the
# method that first needs it.  This keeps startup at ~200 MB and < 2 seconds.
#
# Deferred:
#   from sentence_transformers import SentenceTransformer
#   from llama_index.core.node_parser import HierarchicalNodeParser, ...
#   from llama_index.core.retrievers import AutoMergingRetriever, QueryFusionRetriever
#   from llama_index.retrievers.bm25 import BM25Retriever
#   from llama_index.vector_stores.qdrant import QdrantVectorStore
#   from qdrant_client import QdrantClient
#   from docling.chunking import HybridChunker
#   from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
#
# Re-exported for test patchability: Document, CrossEncoder.
# ---------------------------------------------------------------------------

from llama_index.core import Document  # re-exported for test patchability

# CrossEncoder is patched by tests via ``monkeypatch.setattr(rag_engine,
# "CrossEncoder", ...)``. Keep it as a module-level name so the patch
# actually takes effect inside the lazy-load path below.
try:
    from sentence_transformers import CrossEncoder  # noqa: F401
except Exception:  # pragma: no cover - allow tests to inject a stub
    CrossEncoder = None  # type: ignore[assignment]


# --- Import from RAG config ---
from backend.config import (
    RAG_TEMPERATURE,
    RAG_CONTEXT_WINDOW,
    RAG_EMBEDDING_MODEL,
    RAG_FALLBACK_EMBEDDING_MODEL,
    RAG_EMBEDDING_DIM,
    RAG_EMBEDDING_INSTRUCTION,
    RAG_TOP_K,
    RAG_SIMILARITY_THRESHOLD,
    RAG_RERANKER_MODEL,
    RAG_MULTI_GPU,
    RAG_USE_FLASH_ATTENTION,
    get_rag_provider,
)


# --- Device Detection ---
def get_optimal_device() -> str:
    """Detect the best available accelerator: cuda > mps > cpu.

    Priority:
      1. NVIDIA CUDA (Linux/Windows servers, A100, etc.)
      2. Apple MPS (MacBook Pro M4, M3, M2, M1)
      3. CPU fallback (universal, slowest)

    Can be overridden via the ``RAG_DEVICE`` environment variable.
    """
    env_override = os.getenv("RAG_DEVICE", "").strip().lower()
    if env_override:
        return env_override

    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
        return "cuda"

    # Apple Silicon MPS support (torch >= 1.12)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def _has_multiple_gpus() -> bool:
    """Return True when CUDA is available and more than one GPU is visible."""
    try:
        import torch
        return torch.cuda.is_available() and torch.cuda.device_count() > 1
    except ImportError:
        return False


def _flash_attn_available() -> bool:
    """Check whether flash-attn is installed and importable."""
    try:
        import flash_attn  # noqa: F401
        return True
    except ImportError:
        return False


def _build_cuda_model_kwargs(enable_flash_attn: bool = True, enable_multi_gpu: bool = False) -> Dict[str, Any]:
    """Build model_kwargs for CUDA loading.

    Args:
        enable_flash_attn: Whether to try Flash Attention 2 (requires flash-attn package).
        enable_multi_gpu: Whether to shard across all visible GPUs via device_map="auto".

    Returns:
        Dict suitable for passing as ``model_kwargs`` to SentenceTransformer / CrossEncoder.
    """
    kwargs: Dict[str, Any] = {"torch_dtype": "auto"}
    if enable_multi_gpu and _has_multiple_gpus():
        kwargs["device_map"] = "auto"
    if enable_flash_attn and _flash_attn_available():
        kwargs["attn_implementation"] = "flash_attention_2"
    return kwargs


def get_runtime_diagnostics() -> Dict[str, Any]:
    """Collect runtime diagnostics useful on Slurm GPU nodes."""
    diagnostics: Dict[str, Any] = {
        "selected_device": get_optimal_device(),
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
        "slurm_job_id": os.getenv("SLURM_JOB_ID", ""),
        "slurm_nodelist": os.getenv("SLURM_NODELIST", ""),
        "slurm_procid": os.getenv("SLURM_PROCID", ""),
        "slurm_localid": os.getenv("SLURM_LOCALID", ""),
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_names": [],
    }
    try:
        import torch

        diagnostics["cuda_available"] = torch.cuda.is_available()
        diagnostics["cuda_device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            diagnostics["cuda_device_names"] = [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
    except ImportError:
        pass
    return diagnostics


def log_runtime_diagnostics() -> None:
    diagnostics = get_runtime_diagnostics()
    logger.info(
        "RAG runtime diagnostics: selected_device=%s cuda_available=%s cuda_device_count=%s "
        "cuda_visible_devices=%r slurm_job_id=%r slurm_localid=%r slurm_nodelist=%r gpu_names=%s",
        diagnostics["selected_device"],
        diagnostics["cuda_available"],
        diagnostics["cuda_device_count"],
        diagnostics["cuda_visible_devices"],
        diagnostics["slurm_job_id"],
        diagnostics["slurm_localid"],
        diagnostics["slurm_nodelist"],
        diagnostics["cuda_device_names"],
    )
    if (
        diagnostics["selected_device"].startswith("cuda")
        and diagnostics["cuda_device_count"] > 1
        and not diagnostics["cuda_visible_devices"]
    ):
        logger.warning(
            "Multiple CUDA devices are visible but CUDA_VISIBLE_DEVICES is unset. "
            "On shared Slurm nodes, pin the backend to a specific GPU before startup."
        )


def _format_phase_timings(timings_ms: Dict[str, float]) -> str:
    ordered = []
    for key, value in timings_ms.items():
        ordered.append(f"{key}={value:.1f}ms")
    return ", ".join(ordered)


def _sanitize_metadata_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (str, int, float)):
        return value
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def _sanitize_metadata_dict(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce metadata into JSON-friendly scalar/string values for Qdrant payloads."""
    return {key: _sanitize_metadata_value(value) for key, value in metadata.items()}


# --- RAG Configuration ---
LLM_PROVIDER = get_rag_provider()
LLM_TEMPERATURE = RAG_TEMPERATURE
LLM_CONTEXT_WINDOW = RAG_CONTEXT_WINDOW
RAG_QUERY_TIMEOUT_SECONDS = float(os.getenv("RAG_QUERY_TIMEOUT_SECONDS", "45"))
RAG_SUMMARY_TIMEOUT_SECONDS = float(os.getenv("RAG_SUMMARY_TIMEOUT_SECONDS", "20"))
RAG_RERANK_CANDIDATE_K = int(os.getenv("RAG_RERANK_CANDIDATE_K", "24"))
RAG_RERANK_BATCH_SIZE = int(os.getenv("RAG_RERANK_BATCH_SIZE", "8"))
ZERANK_EXPECTED_SENTENCE_TRANSFORMERS_VERSION = "5.4.1"
ZERANK_EXPECTED_TRANSFORMERS_VERSION = "4.57.1"


def _zerank_runtime_compatible() -> tuple[bool, str]:
    try:
        sentence_transformers_version = importlib_metadata.version("sentence-transformers")
        transformers_version = importlib_metadata.version("transformers")
    except importlib_metadata.PackageNotFoundError as exc:
        return False, f"Required package missing for zerank-2 runtime: {exc}"

    if sentence_transformers_version != ZERANK_EXPECTED_SENTENCE_TRANSFORMERS_VERSION:
        return False, (
            "sentence-transformers version drift detected for zerank-2: "
            f"expected {ZERANK_EXPECTED_SENTENCE_TRANSFORMERS_VERSION}, got {sentence_transformers_version}"
        )

    if transformers_version != ZERANK_EXPECTED_TRANSFORMERS_VERSION:
        return False, (
            "transformers version drift detected for zerank-2: "
            f"expected {ZERANK_EXPECTED_TRANSFORMERS_VERSION}, got {transformers_version}"
        )

    return True, ""


@dataclass
class RAGConfig:
    # Parent chunks: large context for LLM
    parent_chunk_size: int = 2500
    parent_chunk_overlap: int = 300
    # Child chunks: small for precise retrieval
    child_chunk_size: int = 250
    child_chunk_overlap: int = 50
    # Retrieval settings
    retrieve_k: int = 60  # Initial vector search: how many child chunks to fetch
    rerank_threshold: float = 0.1  # Minimum normalized rerank score (0-1) to keep
    max_parents: int = 10  # Max unique parent chunks passed to LLM
    rerank_candidate_k: int = RAG_RERANK_CANDIDATE_K
    rerank_batch_size: int = RAG_RERANK_BATCH_SIZE
    min_chunk_size: int = 50
    similarity_threshold: float = RAG_SIMILARITY_THRESHOLD
    embedding_model: str = RAG_EMBEDDING_MODEL
    top_k: int = RAG_TOP_K
    parser_type: str = "pymupdf"  # "pymupdf" (fast) or "docling" (detailed)
    max_num_pages: int = 200
    max_file_size: int = 104_857_600  # 100 MB
    # Embedding models
    fallback_embedding_model: str = RAG_FALLBACK_EMBEDDING_MODEL
    embedding_dim: Optional[int] = RAG_EMBEDDING_DIM  # MRL truncation (None = full dim)
    embedding_instruction: Optional[str] = RAG_EMBEDDING_INSTRUCTION or None  # Query instruction for Qwen3
    # Reranker model
    reranker_model: str = RAG_RERANKER_MODEL
    reranker_max_length: int = 2048
    # Instruction prepended to queries for domain-aware reranking
    reranker_instruction: str = (
        "You are ranking passages from life science and biology research papers. "
        "Prioritize content about: genes, proteins, enzymes, metabolic pathways, molecular biology, "
        "cell biology, genetics, genomics, transcriptomics, proteomics, metabolomics, "
        "bioactive compounds, natural products, phytochemistry, plant extracts, "
        "analytical techniques (HPLC, GC-MS, NMR, sequencing), biological activity "
        "(antioxidant, antimicrobial, anti-inflammatory, cytotoxicity), "
        "medicinal plants, ethnobotany, and traditional medicine. "
        "Methods, results, and data-driven findings are highly relevant."
    )
    # Vector store and per-user docstore directories.
    qdrant_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "qdrant",
    )
    storage_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "qdrant_storage",
    )


config = RAGConfig()

logger = logging.getLogger(__name__)


class RAGProviderAuthError(Exception):
    """Raised when the configured LLM provider rejects authentication/config."""


class RAGLLMTimeoutError(Exception):
    """Raised when an RAG LLM request exceeds the configured wall-clock budget."""


class OllamaLLM:
    """Wrapper for Ollama or OpenRouter API"""

    def __init__(
        self,
        base_url,
        model,
        temperature=0.1,
        num_ctx=4096,
        provider="ollama",
        api_key=None,
    ):
        self.base_url = base_url
        self.url = f"{base_url}/api/chat" if provider == "ollama" else base_url
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.provider = provider
        self.api_key = api_key

    async def invoke(
        self,
        prompt: str = None,
        messages: list = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
        timeout_seconds: Optional[float] = None,
    ):
        """Invoke the LLM with either a simple prompt or a full messages list.

        Args:
            prompt: Simple string prompt (converted to single user message).
            messages: Full messages list for multi-turn conversations.
        """
        # Build messages list from either argument
        if messages is not None:
            msg_list = messages
        elif prompt is not None:
            msg_list = [{"role": "user", "content": prompt}]
        else:
            raise ValueError("Either prompt or messages must be provided")

        if self.provider == "unconfigured":
            raise RAGProviderAuthError(
                "RAG is not configured. Set RAG_GROQ_API_KEY, RAG_OPENROUTER_API_KEY, "
                "or configure RAG_OLLAMA_URL."
            )

        headers = {}
        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": msg_list,
                "stream": False,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            }
        else:  # OpenAI-compatible: openrouter, groq
            payload = {
                "model": self.model,
                "messages": msg_list,
                "temperature": self.temperature,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}

        last_exception = None
        for attempt in range(max_retries):
            try:
                client = await HttpClientManager.get_client()
                kwargs = {"json": payload, "timeout": None}
                if self.provider != "ollama":
                    kwargs["headers"] = headers
                try:
                    if timeout_seconds is not None:
                        response = await asyncio.wait_for(client.post(self.url, **kwargs), timeout=timeout_seconds)
                    else:
                        response = await client.post(self.url, **kwargs)
                except asyncio.TimeoutError as exc:
                    raise RAGLLMTimeoutError(
                        f"RAG {self.provider} request exceeded {timeout_seconds}s timeout"
                    ) from exc

                if self.provider != "ollama" and response.status_code == 401:
                    env_var = {
                        "groq": "RAG_GROQ_API_KEY",
                        "openrouter": "RAG_OPENROUTER_API_KEY",
                    }.get(self.provider, "the provider's API key")
                    raise RAGProviderAuthError(
                        f"{self.provider.title()} authentication failed. "
                        f"Check {env_var} or configure RAG_OLLAMA_URL as a fallback."
                    )

                # Handle rate limiting (429) with retry
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get("retry-after", base_delay * (2**attempt))
                    )
                    logger.warning(
                        f"Rate limited (429). Retrying after {retry_after}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                result = response.json()

                class Response:
                    def __init__(self, content):
                        self.content = content

                if self.provider == "ollama":
                    return Response(result["message"]["content"])
                else:  # OpenAI-compatible: openrouter, groq
                    return Response(result["choices"][0]["message"]["content"])
            except Exception as e:
                last_exception = e
                # Retry on network errors or 5xx errors
                if (
                    hasattr(e, "status_code")
                    and e.status_code
                    and 500 <= e.status_code < 600
                ):
                    delay = base_delay * (2**attempt)
                    logger.warning(
                        f"Server error {e.status_code}. Retrying after {delay}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(delay)
                    continue
                # Surface SSL/TLS handshake failures with a hint — usually
                # caused by an https:// URL pointing at a plain-HTTP server
                # (or vice versa).
                err_str = str(e)
                if "WRONG_VERSION_NUMBER" in err_str or "SSL" in err_str.upper():
                    logger.error(
                        f"SSL handshake failed calling {self.provider} LLM at {self.url}: {e}. "
                        f"Check that the URL scheme (http vs https) matches what the server is "
                        f"actually serving. Plain Ollama runs on http; a Cloudflare tunnel needs https."
                    )
                else:
                    logger.error(f"Error calling {self.provider} LLM for RAG: {e}")
                raise

        # All retries exhausted
        import traceback

        err_msg = f"RAG LLM Request URL: {self.url}\nPayload: {payload}\nError: {last_exception}\n{traceback.format_exc()}"
        with open("rag_error.log", "w") as f:
            f.write(err_msg)
        logger.error(f"All retries failed for RAG LLM: {last_exception}")
        raise last_exception


class PhytoQueryEmbeddings:
    """Custom embeddings with Qwen3-Embedding-4B primary and bge-m3 fallback.

    Implements a minimal embed_documents / embed_query interface, used both
    directly and via the ``PhytoQueryLIEmbedding`` adapter so LlamaIndex can
    consume the same wrapper. Queries on Qwen3 use ``prompt_name="query"``
    for instruction-aware retrieval; documents are encoded without prompts.
    Falls back to bge-m3 on load failure.
    """

    def __init__(
        self,
        primary_model: str = "Qwen/Qwen3-Embedding-4B",
        fallback_model: str = "BAAI/bge-m3",
        device: Optional[str] = None,
        mrl_dim: Optional[int] = None,
        query_instruction: Optional[str] = None,
    ):
        self.device = device or get_optimal_device()
        self.mrl_dim = mrl_dim
        self.query_instruction = query_instruction
        self.model_name = primary_model
        self.model_dim: int = 2560  # Qwen3-Embedding-4B default
        self._timing_local = threading.local()

        # Defer model loading to first use so RAGService construction is lightweight
        self._primary_model = primary_model
        self._fallback_model = fallback_model
        self._model = None
        self._model_lock = threading.Lock()

    def begin_timing_session(self) -> None:
        self._timing_local.session = {"calls": 0, "total_ms": 0.0, "texts": 0}

    def consume_timing_session(self) -> Dict[str, float]:
        session = getattr(self._timing_local, "session", None)
        self._timing_local.session = None
        if not session:
            return {"calls": 0, "total_ms": 0.0, "texts": 0}
        return session

    def _load_model(self, primary: str, fallback: str):
        """Load embedding model with fallback on failure.

        For CUDA we enable fp16 (auto dtype) for speed/memory savings.
        For Apple MPS we keep fp32 because MPS fp16 support is still maturing.
        CPU always stays fp32.
        """
        from sentence_transformers import SentenceTransformer

        for model_name in [primary, fallback]:
            try:
                logger.info(f"Loading embedding model: {model_name} on {self.device}...")
                kwargs: Dict[str, Any] = {"trust_remote_code": True}

                if self.device.startswith("cuda"):
                    cuda_kwargs = _build_cuda_model_kwargs(
                        enable_flash_attn=RAG_USE_FLASH_ATTENTION,
                        enable_multi_gpu=RAG_MULTI_GPU,
                    )
                    if "device_map" in cuda_kwargs:
                        # device_map="auto" handles its own device placement;
                        # passing device= as well can raise a conflict.
                        kwargs["model_kwargs"] = cuda_kwargs
                    else:
                        kwargs["device"] = self.device
                        kwargs["model_kwargs"] = cuda_kwargs
                    if _flash_attn_available():
                        logger.info("Flash Attention 2 enabled for embedding model.")
                    if RAG_MULTI_GPU and _has_multiple_gpus():
                        try:
                            import torch as _torch
                            gpu_count = _torch.cuda.device_count()
                        except ImportError:
                            gpu_count = 0
                        logger.info(f"Multi-GPU enabled: sharding across {gpu_count} GPUs.")
                else:
                    kwargs["device"] = self.device

                model = SentenceTransformer(model_name, **kwargs)
                self.model_name = model_name
                # Detect dimension from the model (support both old and new API)
                self.model_dim = (
                    getattr(model, "get_embedding_dimension", None)()
                    or getattr(model, "get_sentence_embedding_dimension", None)()
                    or self.model_dim
                )
                logger.info(
                    f"Embedding model loaded: {model_name} "
                    f"(dim={self.model_dim}, mrl_dim={self.mrl_dim or 'full'}, "
                    f"instruction={'yes' if self.query_instruction else 'no'}, "
                    f"device={self.device})"
                )
                return model
            except Exception as e:
                logger.warning(
                    f"Failed to load embedding model {model_name} on {self.device}: {e}"
                )
                # If MPS failed, silently retry on CPU before giving up entirely
                if self.device == "mps" and model_name == primary:
                    try:
                        logger.info("Retrying embedding model load on CPU due to MPS failure...")
                        kwargs = {"device": "cpu", "trust_remote_code": True}
                        model = SentenceTransformer(model_name, **kwargs)
                        self.model_name = model_name
                        self.model_dim = (
                            getattr(model, "get_embedding_dimension", None)()
                            or getattr(model, "get_sentence_embedding_dimension", None)()
                            or self.model_dim
                        )
                        self.device = "cpu"
                        logger.info(
                            f"Embedding model loaded (CPU fallback): {model_name} "
                            f"(dim={self.model_dim}, device=cpu)"
                        )
                        return model
                    except Exception:
                        pass  # fall through to normal fallback flow
                if model_name == primary:
                    logger.info(f"Falling back to {fallback}...")
                else:
                    raise RuntimeError(
                        f"Both primary ({primary}) and fallback ({fallback}) "
                        f"embedding models failed to load."
                    )
        return None  # unreachable, but satisfies type checker

    def _ensure_model_loaded(self):
        """Lazy-load the embedding model on first use."""
        if self._model is not None:
            return
        with self._model_lock:
            if self._model is not None:
                return
            self._model = self._load_model(self._primary_model, self._fallback_model)

    def _maybe_truncate(self, embeddings: List[List[float]]) -> List[List[float]]:
        """Truncate embeddings to MRL dimension if configured."""
        if self.mrl_dim and self.mrl_dim < self.model_dim:
            return [emb[: self.mrl_dim] for emb in embeddings]
        return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of documents. No instruction prompt for documents."""
        self._ensure_model_loaded()
        started = time.perf_counter()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).tolist()
        elapsed_ms = (time.perf_counter() - started) * 1000
        session = getattr(self._timing_local, "session", None)
        if session is not None:
            session["calls"] += 1
            session["texts"] += len(texts)
            session["total_ms"] += elapsed_ms
        return self._maybe_truncate(embeddings)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query. Uses instruction prompt for Qwen3 models.

        Qwen3-Embedding models ship with built-in prompt templates ("query").
        If RAG_EMBEDDING_INSTRUCTION is set, it is used as a custom prompt.
        Non-Qwen models are encoded without prompts.
        """
        self._ensure_model_loaded()
        if "qwen" in self.model_name.lower():
            # Qwen3 models have built-in prompt_name="query" for retrieval
            encode_kwargs = {
                "normalize_embeddings": True,
                "show_progress_bar": False,
                "convert_to_numpy": True,
            }
            if self.query_instruction:
                # Custom domain instruction overrides the built-in prompt
                encode_kwargs["prompt"] = self.query_instruction
            else:
                # Use the model's built-in "query" prompt template
                encode_kwargs["prompt_name"] = "query"
            embeddings = self._model.encode([text], **encode_kwargs)
        else:
            embeddings = self._model.encode(
                [text],
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        result = embeddings[0].tolist()
        if self.mrl_dim and self.mrl_dim < self.model_dim:
            result = result[: self.mrl_dim]
        return result


# ---------------------------------------------------------------------------
# Per-user storage state.
#
# Each RAG user gets:
#   - A Qdrant collection in the shared local Qdrant DB (data/qdrant/).
#   - A LlamaIndex StorageContext whose docstore is a SimpleDocumentStore
#     persisted under data/qdrant_storage/<safe_user_id>/.
#
# Parents and children both live in the docstore (so AutoMergingRetriever
# can walk children → parents). Only LEAF nodes are embedded into Qdrant.
# ---------------------------------------------------------------------------


@dataclass
class _UserStorage:
    """Cached per-user LlamaIndex storage handles."""

    collection_name: str
    persist_dir: str
    storage_context: Any  # llama_index.core.StorageContext
    vector_store: Any     # llama_index.vector_stores.qdrant.QdrantVectorStore


class RAGService:
    def __init__(self):
        self._device = get_optimal_device()
        self.embeddings = PhytoQueryEmbeddings(
            primary_model=config.embedding_model,
            fallback_model=config.fallback_embedding_model,
            device=self._device,
            mrl_dim=config.embedding_dim,
            query_instruction=config.embedding_instruction,
        )
        # Sync service device with embeddings (embeddings may have fallen back to cpu)
        self._device = self.embeddings.device

        # Defer reranker loading to first use so service construction is lightweight
        self._reranker = ...  # sentinel: not loaded yet
        self._reranker_lock = threading.Lock()

        self.llm = OllamaLLM(
            base_url=LLM_PROVIDER.get("url", "").replace("/api/chat", ""),
            model=LLM_PROVIDER["model"],
            temperature=LLM_TEMPERATURE,
            num_ctx=LLM_CONTEXT_WINDOW,
            provider=LLM_PROVIDER["provider"],
            api_key=LLM_PROVIDER.get("api_key"),
        )
        # Per-user storage cache (Qdrant collection + LlamaIndex StorageContext).
        self._user_storage: Dict[str, _UserStorage] = {}
        self._user_storage_lock = threading.Lock()
        # Cache for Docling converter to avoid reloading models.
        self._docling_converter = None
        # Shared Qdrant client (single local DB at config.qdrant_dir).
        self._qdrant_client = None
        self._qdrant_lock = threading.Lock()
        # Configure LlamaIndex Settings once (replaces deprecated ServiceContext).
        self._configure_llama_index_settings()

    # ------------------------------------------------------------------
    # LlamaIndex Settings wiring
    # ------------------------------------------------------------------

    def _configure_llama_index_settings(self) -> None:
        """Plug our LLM + embeddings into LlamaIndex's global Settings.

        This is idempotent — every RAGService construction overwrites the
        Settings.llm / Settings.embed_model with the freshly-built adapters
        so tests that swap the singleton get coherent state.
        """
        from llama_index.core import Settings
        from backend.services.llamaindex_adapters import (
            PhytoQueryLLM,
            PhytoQueryLIEmbedding,
        )

        Settings.llm = PhytoQueryLLM(
            ollama_llm=self.llm,
            context_window=LLM_CONTEXT_WINDOW,
            num_output=2048,
            request_timeout=RAG_QUERY_TIMEOUT_SECONDS,
        )
        Settings.embed_model = PhytoQueryLIEmbedding(
            embeddings=self.embeddings,
            embed_batch_size=16,
        )

    async def _invoke_llm(self, *, prompt: str = None, messages: list = None, timeout_seconds: Optional[float] = None, max_retries: int = 3):
        try:
            return await self.llm.invoke(
                prompt=prompt,
                messages=messages,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        except TypeError as exc:
            if "timeout_seconds" not in str(exc):
                raise
            return await self.llm.invoke(prompt=prompt, messages=messages)

    def _reranker_max_length(self) -> Optional[int]:
        """Resolve a safe ``max_length`` for the configured reranker.

        ``config.reranker_max_length`` (default 2048) is only safe for
        long-context rerankers like zerank-2 and bge-reranker-v2-m3.
        Classic 512-token models (ms-marco-MiniLM, bge-reranker base)
        crash with a tensor-size mismatch when fed inputs longer than
        their architecture supports. Returning ``None`` here lets the
        CrossEncoder fall back to the tokenizer's ``model_max_length``
        — the right answer for any pretrained model.
        """
        name = config.reranker_model.lower()
        long_context_markers = ("zerank", "bge-reranker-v2", "bge-reranker-v3")
        if any(marker in name for marker in long_context_markers):
            return config.reranker_max_length
        return None  # use tokenizer-native max length

    def _make_reranker_kwargs(self) -> Dict[str, Any]:
        """Build CrossEncoder constructor kwargs honoring the resolved
        max_length and zerank-specific options."""
        kwargs: Dict[str, Any] = {}
        max_len = self._reranker_max_length()
        if max_len is not None:
            kwargs["max_length"] = max_len
        if "zerank" in config.reranker_model.lower():
            kwargs["trust_remote_code"] = True
        return kwargs

    @property
    def reranker(self):
        """Lazy-load the reranker on first access."""
        if self._reranker is not ...:
            return self._reranker
        with self._reranker_lock:
            if self._reranker is not ...:
                return self._reranker
            try:
                # Resolve via module-level CrossEncoder so tests can monkeypatch it.
                _ce_cls = globals().get("CrossEncoder")
                if _ce_cls is None:
                    from sentence_transformers import CrossEncoder as _ce_cls
                logger.info(f"Loading reranker: {config.reranker_model} on {self._device}...")
                kwargs: Dict[str, Any] = dict(self._make_reranker_kwargs())
                if "zerank" in config.reranker_model.lower():
                    compatible, reason = _zerank_runtime_compatible()
                    if not compatible:
                        raise RuntimeError(reason)

                if self._device.startswith("cuda"):
                    cuda_kwargs = _build_cuda_model_kwargs(
                        enable_flash_attn=RAG_USE_FLASH_ATTENTION,
                        enable_multi_gpu=RAG_MULTI_GPU,
                    )
                    if "device_map" in cuda_kwargs:
                        kwargs["model_kwargs"] = cuda_kwargs
                    else:
                        kwargs["device"] = self._device
                        kwargs["model_kwargs"] = cuda_kwargs
                else:
                    kwargs["device"] = self._device

                self._reranker = _ce_cls(config.reranker_model, **kwargs)
                logger.info(
                    "Reranker loaded (model=%s, max_length=%s, device=%s)",
                    config.reranker_model,
                    kwargs.get("max_length", "tokenizer-default"),
                    self._device,
                )
            except Exception as e:
                logger.warning(f"Failed to load reranker on {self._device}: {e}")
                if self._device == "mps":
                    try:
                        _ce_cls = globals().get("CrossEncoder")
                        if _ce_cls is None:
                            from sentence_transformers import CrossEncoder as _ce_cls
                        logger.info("Retrying reranker load on CPU due to MPS failure...")
                        kwargs = dict(self._make_reranker_kwargs())
                        kwargs["device"] = "cpu"
                        if "zerank" in config.reranker_model.lower():
                            compatible, reason = _zerank_runtime_compatible()
                            if not compatible:
                                raise RuntimeError(reason)
                        self._reranker = _ce_cls(config.reranker_model, **kwargs)
                        self._device = "cpu"
                        logger.info("Reranker loaded successfully on CPU fallback.")
                    except Exception as cpu_e:
                        logger.warning(f"Failed to load reranker on CPU fallback: {cpu_e}")
                        self._reranker = None
                else:
                    self._reranker = None
            return self._reranker

    @reranker.setter
    def reranker(self, value):
        """Allow tests and callers to inject a mock reranker directly."""
        self._reranker = value

    # ------------------------------------------------------------------
    # Qdrant + per-user storage helpers
    # ------------------------------------------------------------------

    def _safe_user_id(self, user_id: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_]", "_", user_id)

    def _get_collection_suffix(self) -> str:
        """Short hash key versioning collections by embedding model + dim.

        This keeps collections from colliding across embedding model upgrades
        (different dims would otherwise raise on insert).
        """
        model_key = f"{self.embeddings.model_name}:{self.embeddings.model_dim}"
        return hashlib.md5(model_key.encode()).hexdigest()[:8]

    def _get_user_collection_name(self, user_id: str) -> str:
        return f"user_{self._safe_user_id(user_id)}_{self._get_collection_suffix()}"

    def _get_user_persist_dir(self, user_id: str) -> str:
        return os.path.join(config.storage_dir, self._safe_user_id(user_id))

    def _get_qdrant_client(self):
        """Lazy-init the shared local QdrantClient.

        We use a single client pointed at ``data/qdrant/`` rather than
        per-user directories. Qdrant collections are first-class so this
        gives us the same isolation Chroma offered via per-user dirs but
        without the SQLite-tenant gymnastics that plagued
        ``_reset_user_chroma_in_place``.
        """
        if self._qdrant_client is not None:
            return self._qdrant_client
        with self._qdrant_lock:
            if self._qdrant_client is not None:
                return self._qdrant_client
            from qdrant_client import QdrantClient

            os.makedirs(config.qdrant_dir, exist_ok=True)
            self._qdrant_client = QdrantClient(path=config.qdrant_dir)
            logger.info(f"Initialized Qdrant local client at {config.qdrant_dir}")
            return self._qdrant_client

    def _build_user_storage(self, user_id: str) -> _UserStorage:
        """Construct the StorageContext + Qdrant vector store for a user.

        Loads a persisted SimpleDocumentStore if one exists under
        ``data/qdrant_storage/<safe_user_id>/``; otherwise creates an
        empty one. The Qdrant collection is created on first insert by
        QdrantVectorStore — we don't need to pre-create it.
        """
        from llama_index.core import StorageContext
        from llama_index.core.storage.docstore import SimpleDocumentStore
        from llama_index.core.storage.index_store import SimpleIndexStore
        from llama_index.vector_stores.qdrant import QdrantVectorStore

        client = self._get_qdrant_client()
        collection_name = self._get_user_collection_name(user_id)
        persist_dir = self._get_user_persist_dir(user_id)
        os.makedirs(persist_dir, exist_ok=True)

        # Hybrid mode: dense vectors come from our PhytoQueryEmbeddings
        # (Qwen3 / bge-m3); sparse vectors come from FastEmbed's BM25
        # tokenizer (NOT the default SPLADE++ — that's a 500MB neural
        # model we don't want). ``Qdrant/bm25`` is just a vocabulary +
        # IDF table, ships small, runs on CPU, and matches the algorithm
        # we used to run in Python via rank_bm25 — only now sparse
        # vectors live in Qdrant and RRF fusion happens server-side.
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            enable_hybrid=True,
            fastembed_sparse_model="Qdrant/bm25",
            batch_size=20,
        )

        docstore_path = os.path.join(persist_dir, "docstore.json")
        index_store_path = os.path.join(persist_dir, "index_store.json")
        if os.path.exists(docstore_path):
            try:
                docstore = SimpleDocumentStore.from_persist_path(docstore_path)
            except Exception as exc:
                logger.warning(
                    f"Failed to load docstore for {user_id} ({exc!r}); starting empty."
                )
                docstore = SimpleDocumentStore()
        else:
            docstore = SimpleDocumentStore()

        if os.path.exists(index_store_path):
            try:
                index_store = SimpleIndexStore.from_persist_path(index_store_path)
            except Exception as exc:
                logger.warning(
                    f"Failed to load index_store for {user_id} ({exc!r}); starting empty."
                )
                index_store = SimpleIndexStore()
        else:
            index_store = SimpleIndexStore()

        storage_context = StorageContext.from_defaults(
            docstore=docstore,
            index_store=index_store,
            vector_store=vector_store,
            persist_dir=persist_dir,
        )
        return _UserStorage(
            collection_name=collection_name,
            persist_dir=persist_dir,
            storage_context=storage_context,
            vector_store=vector_store,
        )

    def _get_user_storage(self, user_id: str) -> _UserStorage:
        """Return cached storage for a user, building it on first access."""
        if user_id in self._user_storage:
            return self._user_storage[user_id]
        with self._user_storage_lock:
            if user_id in self._user_storage:
                return self._user_storage[user_id]
            storage = self._build_user_storage(user_id)
            self._user_storage[user_id] = storage
            logger.info(
                f"Created storage context for user: {user_id} "
                f"(collection={storage.collection_name})"
            )
            return storage

    def _persist_user_storage(self, user_id: str) -> None:
        storage = self._user_storage.get(user_id)
        if storage is None:
            return
        try:
            storage.storage_context.persist(persist_dir=storage.persist_dir)
        except Exception as exc:
            logger.warning(f"Failed to persist storage for {user_id}: {exc!r}")

    def _invalidate_user_storage(self, user_id: str) -> None:
        """Drop cached storage so the next access reloads from disk."""
        self._user_storage.pop(user_id, None)
        gc.collect()

    def _reset_user_qdrant_in_place(self, user_id: str) -> None:
        """Delete a user's Qdrant collection.

        Replacement for the old ``_reset_user_chroma_in_place``. Qdrant
        ``delete_collection`` is a single atomic call — no SQLite-tenant
        gymnastics, no SQLITE_READONLY_DBMOVED race, no shutil.rmtree
        fallback dance.
        """
        try:
            client = self._get_qdrant_client()
            collection_name = self._get_user_collection_name(user_id)
            try:
                client.delete_collection(collection_name=collection_name)
                logger.info(
                    f"Deleted Qdrant collection {collection_name} for user {user_id}"
                )
            except Exception as exc:
                # Likely "collection not found" — benign.
                logger.info(
                    f"Qdrant collection delete for {user_id} skipped/failed: {exc!r}"
                )
        finally:
            self._invalidate_user_storage(user_id)

    # ------------------------------------------------------------------
    # Source management
    # ------------------------------------------------------------------

    def _delete_existing_sources(self, user_id: str, source_names: List[str]) -> None:
        """Remove all nodes (parents + children) for the given source filenames."""
        if not source_names:
            return

        from qdrant_client.http import models as qmodels

        storage = self._get_user_storage(user_id)
        client = self._get_qdrant_client()
        unique_sources = sorted(set(source_names))

        # 1. Delete from Qdrant (vectors). Collection is created lazily on
        #    first insert — if it doesn't exist yet, skip.
        try:
            existing = {c.name for c in client.get_collections().collections}
        except Exception:
            existing = set()
        if storage.collection_name in existing:
            try:
                client.delete(
                    collection_name=storage.collection_name,
                    points_selector=qmodels.FilterSelector(
                        filter=qmodels.Filter(
                            should=[
                                qmodels.FieldCondition(
                                    key="source",
                                    match=qmodels.MatchValue(value=name),
                                )
                                for name in unique_sources
                            ]
                        )
                    ),
                )
                logger.info(
                    f"Replaced existing indexed chunks for {unique_sources} from user {user_id}'s Qdrant collection"
                )
            except Exception as exc:
                logger.warning(
                    f"Qdrant point delete failed for {user_id} sources={unique_sources}: {exc!r}"
                )

        # 2. Delete from docstore (parents + children).
        docstore = storage.storage_context.docstore
        try:
            doomed_ids = []
            for node_id, node in list(docstore.docs.items()):
                src = (getattr(node, "metadata", None) or {}).get("source")
                if src in unique_sources:
                    doomed_ids.append(node_id)
            for node_id in doomed_ids:
                try:
                    docstore.delete_document(node_id, raise_error=False)
                except TypeError:
                    docstore.delete_document(node_id)
        except Exception as exc:
            logger.warning(f"Docstore prune failed for {user_id}: {exc!r}")

        self._persist_user_storage(user_id)

    # ------------------------------------------------------------------
    # Indexing pipeline
    # ------------------------------------------------------------------

    def process_and_index_pdfs_with_texts(
        self,
        pdf_paths: List[str],
        parser_type: str = "pymupdf",
        user_id: str = "default",
    ):
        """Extract, chunk, index, and return per-file extracted text for summaries."""
        from llama_index.core import VectorStoreIndex
        from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes

        total_started = time.perf_counter()
        all_documents: List[Any] = []
        extracted_texts: Dict[str, str] = {}
        source_names = [os.path.basename(path) for path in pdf_paths]
        parse_and_chunk_ms = 0.0

        # Pass 1: extract per-file text + LlamaIndex Documents.
        for path in pdf_paths:
            file_started = time.perf_counter()
            documents, extracted_text = self._process_pdf(
                path,
                user_id=user_id,
                parser_type=parser_type,
            )
            file_elapsed_ms = (time.perf_counter() - file_started) * 1000
            parse_and_chunk_ms += file_elapsed_ms
            all_documents.extend(documents)
            if extracted_text:
                extracted_texts[os.path.basename(path)] = extracted_text
            logger.info(
                "RAG upload phase: parser=%s file=%s parse=%.1fms documents=%s extracted_chars=%s",
                parser_type,
                os.path.basename(path),
                file_elapsed_ms,
                len(documents),
                len(extracted_text or ""),
            )

        if not all_documents:
            logger.info(
                "RAG upload timings: parser=%s user=%s files=%s chunks=0 parse=%.1fms total=%.1fms",
                parser_type,
                user_id,
                len(pdf_paths),
                parse_and_chunk_ms,
                (time.perf_counter() - total_started) * 1000,
            )
            return source_names, extracted_texts

        # Pass 2: hierarchical chunking — one shared parser across all docs.
        node_parser = HierarchicalNodeParser.from_defaults(
            chunk_sizes=[config.parent_chunk_size, config.child_chunk_size],
        )
        all_nodes = node_parser.get_nodes_from_documents(all_documents)
        leaf_nodes = get_leaf_nodes(all_nodes)

        # Distribute the contextual chunk header (Doc Title > Section)
        # across leaf nodes so each embedded child carries the breadcrumb.
        # Parents keep their natural text (which already starts with the
        # header from _process_pdf).
        for leaf in leaf_nodes:
            header = (leaf.metadata or {}).get("cch_header") or ""
            if header and not leaf.text.startswith(header):
                leaf.text = header + leaf.text

        index_started = time.perf_counter()
        self.embeddings.begin_timing_session()
        try:
            self._delete_existing_sources(user_id, source_names)
            storage = self._get_user_storage(user_id)
            # Add parents + children to docstore so AutoMergingRetriever
            # can resolve children → parents at query time.
            storage.storage_context.docstore.add_documents(all_nodes)

            # Embed and insert ONLY leaves into Qdrant. VectorStoreIndex
            # constructor handles the embedding + insertion.
            try:
                VectorStoreIndex(
                    nodes=leaf_nodes,
                    storage_context=storage.storage_context,
                    show_progress=False,
                )
            except Exception as e:
                # Detect known Qdrant corruption / lock signatures and
                # recover by resetting the collection in place. We do NOT
                # reset on every error — point inserts can also fail for
                # transient reasons; only recover on clearly persistent
                # signatures.
                msg = str(e).lower()
                is_corrupt = (
                    "wrong vector size" in msg
                    or "dimension" in msg
                    or ("collection" in msg and "not found" in msg)
                )
                logger.warning(
                    f"Indexing failed for user {user_id} ({e!r}); "
                    f"{'resetting Qdrant collection and ' if is_corrupt else ''}"
                    f"invalidating cached storage and retrying once."
                )
                self._invalidate_user_storage(user_id)
                if is_corrupt:
                    self._reset_user_qdrant_in_place(user_id)
                self._delete_existing_sources(user_id, source_names)
                storage = self._get_user_storage(user_id)
                storage.storage_context.docstore.add_documents(all_nodes)
                try:
                    VectorStoreIndex(
                        nodes=leaf_nodes,
                        storage_context=storage.storage_context,
                        show_progress=False,
                    )
                except Exception as retry_err:
                    logger.error(
                        f"Retry failed even after cache invalidation for {user_id}: {retry_err}"
                    )
                    raise retry_err from retry_err

            self._persist_user_storage(user_id)
        finally:
            embed_stats = self.embeddings.consume_timing_session()

        index_total_ms = (time.perf_counter() - index_started) * 1000
        embed_ms = float(embed_stats.get("total_ms", 0.0))
        embed_calls = int(embed_stats.get("calls", 0))
        embed_texts = int(embed_stats.get("texts", 0))
        store_overhead_ms = max(index_total_ms - embed_ms, 0.0)
        total_elapsed_ms = (time.perf_counter() - total_started) * 1000
        logger.info(
            "RAG upload timings: parser=%s user=%s files=%s parents=%s leaves=%s %s embed_calls=%s embed_texts=%s total=%.1fms",
            parser_type,
            user_id,
            len(pdf_paths),
            len(all_nodes) - len(leaf_nodes),
            len(leaf_nodes),
            _format_phase_timings({
                "parse": parse_and_chunk_ms,
                "embed": embed_ms,
                "store_overhead": store_overhead_ms,
                "index_total": index_total_ms,
            }),
            embed_calls,
            embed_texts,
            total_elapsed_ms,
        )

        return source_names, extracted_texts

    def process_and_index_pdfs(
        self,
        pdf_paths: List[str],
        parser_type: str = "pymupdf",
        user_id: str = "default",
    ):
        """Extract, chunk, and index PDFs for a specific user."""
        indexed_files, _ = self.process_and_index_pdfs_with_texts(
            pdf_paths,
            parser_type=parser_type,
            user_id=user_id,
        )
        return indexed_files

    def _extract_pdf_metadata(self, pdf_path: str) -> Dict[str, str]:
        """Extract DOI, authors, and journal from PDF metadata and first pages."""
        metadata = {"authors": "", "doi": "", "journal": "", "title": ""}
        try:
            import fitz

            doc = fitz.open(pdf_path)

            # Try PDF document info first
            pdf_info = doc.metadata or {}
            if pdf_info.get("author"):
                metadata["authors"] = pdf_info["author"]
            if pdf_info.get("title"):
                metadata["title"] = pdf_info["title"].strip()

            # Extract text from first 2 pages for regex-based extraction
            first_pages_text = ""
            for i in range(min(2, len(doc))):
                first_pages_text += doc[i].get_text("text") + "\n"
            doc.close()

            # DOI pattern
            doi_match = re.search(r'(10\.\d{4,}/[^\s,;"\'>]+)', first_pages_text)
            if doi_match:
                metadata["doi"] = doi_match.group(1).rstrip(".")

            # Journal detection: look for common patterns
            journal_patterns = [
                r"(?:Published in|Journal of|Proceedings of)[:\s]+([^\n]+)",
                r"(?:^|\n)([A-Z][a-z]+(?: [A-Z][a-z]+)* (?:Journal|Review|Letters|Research|Science|Chemistry|Pharmacology|Phytochemistry|Biochemistry|Biology)[^\n]*)",
            ]
            for pattern in journal_patterns:
                match = re.search(pattern, first_pages_text, re.IGNORECASE)
                if match:
                    metadata["journal"] = match.group(1).strip()[:100]
                    break

        except Exception as e:
            logger.warning(f"Metadata extraction failed for {pdf_path}: {e}")
        return metadata

    def _build_cch_header(self, doc_title: str, section_title: str) -> str:
        """``[Doc Title] > [Section Title]\\n\\n`` — only emitted when non-empty."""
        parts = []
        if doc_title:
            parts.append(doc_title)
        if section_title and section_title != "Start":
            parts.append(section_title)
        if parts:
            return " > ".join(parts) + "\n\n"
        return ""

    def _make_document(self, text: str, metadata: Dict[str, Any]) -> Any:
        """Construct a LlamaIndex ``Document`` with metadata excluded from
        embed and LLM templates.

        By default LlamaIndex prepends every metadata key into the text
        the embedder sees ("source: x\\nparser_type: y\\n..."). With our
        12-key metadata that consumes ~200 chars per chunk — out of a
        250-char child chunk, only ~50 chars of actual content reach the
        embedding model, which destroys retrieval quality.

        We exclude all metadata from both templates because:
          * The semantic breadcrumb the embedder needs is already
            prepended to leaf text via ``cch_header`` in
            ``process_and_index_pdfs_with_texts``.
          * Our ``query()`` reads metadata via ``node.metadata.get(...)``
            directly, bypassing any LlamaIndex template — so excluding
            keys from the LLM template doesn't lose us anything either.
        """
        keys = list(metadata.keys())
        return Document(
            text=text,
            metadata=metadata,
            excluded_embed_metadata_keys=keys,
            excluded_llm_metadata_keys=keys,
        )

    def _process_pdf(self, pdf_path: str, user_id: str = "default", parser_type: str = "pymupdf"):
        """PDF processing pipeline.

        Returns ``(documents, extracted_text)`` where ``documents`` is a list
        of LlamaIndex ``Document`` objects (one per detected section, plus
        one per extracted table). HierarchicalNodeParser will further split
        these into 2500-char parents and 250-char children.
        """
        source = os.path.basename(pdf_path)

        # 0. Extract metadata (DOI, authors, journal)
        pdf_metadata = self._extract_pdf_metadata(pdf_path)

        # 1. Extract using selected parser.
        # The user-facing toggle dictates the engine: Fast=PyMuPDF,
        # Detailed=Docling. We do NOT silently swap engines on failure —
        # if Docling crashes the user's "Detailed" choice should surface
        # as an error, not produce a Fast-quality result mislabeled as
        # Detailed. Within Docling, the advanced HybridChunker skill
        # path falls back to plain Docling extraction (still Docling),
        # which is fine because both are the same engine.
        if parser_type == "docling":
            try:
                return self._process_with_docling_skill(pdf_path, source, pdf_metadata, user_id)
            except Exception as e:
                logger.error(
                    f"Docling skill processing failed for {source} ({e!r}); "
                    f"retrying with plain Docling extraction (still Detailed)."
                )
                # Fall through to plain Docling — same engine, lighter pipeline.

        if parser_type == "pymupdf":
            full_text, tables = self._extract_with_pymupdf(pdf_path)
        else:
            full_text, tables = self._extract_with_docling(pdf_path)

        if not full_text:
            return [], ""

        # 2. Section detection
        sections = self._detect_sections(full_text)
        documents: List[Any] = []
        file_ext = os.path.splitext(source)[1].lower() or ".pdf"
        indexed_at = datetime.now(timezone.utc).isoformat()
        doc_title = pdf_metadata.get("title", "")

        # One Document per section. The section header is prepended into
        # the text so the parent chunk preserves it; the leaf-level CCH is
        # added later via the cch_header metadata key.
        for sec_idx, section in enumerate(sections):
            sec_text = (section.get("text") or "").strip()
            if not sec_text:
                continue
            section_title = section.get("title", "")
            cch_header = self._build_cch_header(doc_title, section_title)

            if section_title and section_title != "Start":
                section_text_with_header = f"## {section_title}\n\n{sec_text}"
            else:
                section_text_with_header = sec_text

            metadata = _sanitize_metadata_dict({
                "source": source,
                "parser_type": parser_type,
                "file_type": file_ext,
                "indexed_at": indexed_at,
                "content_type": "text",
                "section_title": section_title,
                "cch_header": cch_header,
                "doc_title": doc_title,
                "doc_authors": pdf_metadata.get("authors", ""),
                "doc_doi": pdf_metadata.get("doi", ""),
                "doc_journal": pdf_metadata.get("journal", ""),
                "section_index": sec_idx,
            })
            documents.append(self._make_document(section_text_with_header, metadata))

        # Tables: index each as its own short Document. Table content is
        # usually small enough that HierarchicalNodeParser keeps it as a
        # single leaf with no parent (which is what we want).
        for tbl_idx, table in enumerate(tables):
            content = (table.get("content") or "").strip()
            if not content:
                continue
            metadata = _sanitize_metadata_dict({
                "source": source,
                "parser_type": parser_type,
                "file_type": file_ext,
                "indexed_at": indexed_at,
                "content_type": "table",
                "section_title": "",
                "cch_header": self._build_cch_header(doc_title, "Table"),
                "doc_title": doc_title,
                "doc_authors": pdf_metadata.get("authors", ""),
                "doc_doi": pdf_metadata.get("doi", ""),
                "doc_journal": pdf_metadata.get("journal", ""),
                "page": int(table.get("page", 0) or 0),
                "table_index": tbl_idx,
            })
            documents.append(self._make_document(f"## Table\n\n{content}", metadata))

        return documents, full_text

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def _build_metadata_filters(self, filter_files: Optional[List[str]]):
        """Translate the public ``filter_files`` list into LlamaIndex MetadataFilters."""
        if not filter_files:
            return None
        from llama_index.core.vector_stores.types import (
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )

        return MetadataFilters(
            filters=[
                MetadataFilter(
                    key="source",
                    value=list(filter_files),
                    operator=FilterOperator.IN,
                )
            ]
        )

    def _retrieve_candidate_nodes(
        self,
        question: str,
        user_id: str,
        filter_files: Optional[List[str]],
    ):
        """Hybrid (vector + sparse) retrieval returning CHILD nodes.

        Uses Qdrant's native hybrid mode: sparse vectors (FastEmbed BM25)
        are stored alongside dense vectors at index time and fused
        server-side via RRF on every query.

        Importantly, this does NOT wrap in ``AutoMergingRetriever``.
        Auto-merging swaps children for parents *before* the cross-encoder
        rerank, which sends 2500-char parent passages into a reranker
        capped at 512 tokens — overflows the tokenizer and the reranker
        silently fails. We instead resolve children → parents AFTER
        reranking (see ``_resolve_parents`` and ``query``), matching the
        proven LangChain-era ordering.
        """
        from llama_index.core import VectorStoreIndex

        storage = self._get_user_storage(user_id)
        metadata_filters = self._build_metadata_filters(filter_files)

        index = VectorStoreIndex.from_vector_store(
            storage.vector_store,
            storage_context=storage.storage_context,
        )
        return index.as_retriever(
            vector_store_query_mode="hybrid",
            similarity_top_k=max(config.rerank_candidate_k, config.top_k),
            sparse_top_k=config.retrieve_k,
            filters=metadata_filters,
        )

    def _resolve_parents(
        self,
        child_nodes_with_scores: List[Any],
        user_id: str,
    ) -> List[Any]:
        """For each reranked child, swap in its parent node from the
        docstore. De-dupe parents (multiple top-K children sharing one
        parent collapse to a single parent, scored by the highest-ranked
        sibling). This matches the LangChain pipeline's
        rerank-then-resolve ordering, NOT LlamaIndex's
        ``AutoMergingRetriever`` which would do it the other way around.

        Falls through to the original child node when:
          * the node has no parent relationship (e.g. table-only
            documents that the hierarchical parser kept as a single leaf)
          * the parent can't be loaded from the docstore for any reason
        """
        if not child_nodes_with_scores:
            return []

        from llama_index.core.schema import NodeRelationship, NodeWithScore

        # Storage is fetched lazily on first node that actually has a
        # parent relationship — keeps the unit-test path (mocked
        # retriever returning empty results / table-only nodes) free
        # of any Qdrant / docstore dependency.
        docstore = None

        seen_parent_ids: set = set()
        results: List[Any] = []
        for nws in child_nodes_with_scores:
            child = nws.node
            parent_rel = (getattr(child, "relationships", None) or {}).get(
                NodeRelationship.PARENT
            )
            if parent_rel is None:
                # Standalone leaf (table, short doc) — use as-is.
                results.append(nws)
                continue

            parent_id = parent_rel.node_id
            if parent_id in seen_parent_ids:
                # Higher-scored sibling already pulled this parent.
                continue
            seen_parent_ids.add(parent_id)

            if docstore is None:
                try:
                    docstore = self._get_user_storage(user_id).storage_context.docstore
                except Exception as exc:
                    logger.debug(f"Storage unavailable for parent resolution ({exc!r}); using child")
                    results.append(nws)
                    continue

            try:
                parent_node = docstore.get_node(parent_id)
            except Exception as exc:
                logger.debug(f"Parent lookup failed for {parent_id} ({exc!r}); using child")
                results.append(nws)
                continue

            results.append(NodeWithScore(node=parent_node, score=nws.score))
        return results

    def _rerank_nodes(self, question: str, nodes_with_scores: List[Any]) -> List[Any]:
        """Defensive cross-encoder rerank with three-layer empty-input defense.

        Identical semantics to the old ``query`` rerank block: caps at
        ``rerank_candidate_k``, drops blank/whitespace passages, prepends
        the zerank-2 instruction, batches at ``rerank_batch_size``,
        normalizes scores 0–1 via min-max, filters by ``rerank_threshold``,
        and falls back to retrieval order if everything fails.

        Operates on LlamaIndex ``NodeWithScore`` objects in-place — sets
        ``node.score`` to the reranker's normalized 0–1 value.
        """
        if not self.reranker or not nodes_with_scores:
            return nodes_with_scores

        try:
            import numpy as np

            candidate_limit = int(config.rerank_candidate_k)
            if candidate_limit <= 0:
                logger.info(
                    "RAG rerank skipped: non-positive candidate limit=%s",
                    candidate_limit,
                )
                return nodes_with_scores
            candidates = nodes_with_scores[:candidate_limit]

            blank_count = 0
            non_blank: List[Any] = []
            for nws in candidates:
                text = (getattr(nws.node, "text", "") or "").strip()
                if not text:
                    blank_count += 1
                    continue
                non_blank.append(nws)

            cand_lengths = [len((nws.node.text or "").strip()) for nws in non_blank]
            logger.info(
                "RAG rerank boundary: total_candidates=%s filtered_candidates=%s blank_candidates=%s min_chars=%s max_chars=%s",
                len(candidates),
                len(non_blank),
                blank_count,
                min(cand_lengths) if cand_lengths else 0,
                max(cand_lengths) if cand_lengths else 0,
            )

            if not non_blank:
                logger.warning("No non-empty rerank candidates available")
                return nodes_with_scores

            query_text = (question or "").strip()
            if not query_text:
                logger.warning("Empty query — skipping rerank")
                return non_blank

            # zerank-2 supports instruction-aware reranking.
            if config.reranker_instruction and "zerank" in config.reranker_model.lower():
                instructed_query = f"{config.reranker_instruction}\n{query_text}"
            else:
                instructed_query = query_text

            pairs: List[List[str]] = []
            valid: List[Any] = []
            for nws in non_blank:
                passage = (nws.node.text or "").strip()
                if not passage:
                    continue
                pairs.append([instructed_query, passage])
                valid.append(nws)

            if not pairs:
                logger.warning("No tokenizable rerank pairs after filtering")
                return non_blank

            rerank_batch_size = max(1, int(config.rerank_batch_size))
            rerank_scores: List[float] = []
            for batch_start in range(0, len(pairs), rerank_batch_size):
                batch = pairs[batch_start: batch_start + rerank_batch_size]
                batch_scores = self.reranker.predict(batch)
                rerank_scores.extend(list(batch_scores))

            if len(rerank_scores) != len(valid):
                logger.warning(
                    f"Reranker score count mismatch: expected {len(valid)} got {len(rerank_scores)}"
                )
                return non_blank

            scores = np.array(rerank_scores, dtype=float)
            if not np.isfinite(scores).all():
                logger.warning("Reranker returned non-finite scores")
                return non_blank

            # Normalize to 0-1 via min-max.
            min_s, max_s = scores.min(), scores.max()
            if max_s > min_s:
                normalized = (scores - min_s) / (max_s - min_s)
            else:
                normalized = np.ones_like(scores) * 0.5

            for i, nws in enumerate(valid):
                nws.score = float(normalized[i])

            survivors = [
                nws for i, nws in enumerate(valid)
                if normalized[i] >= config.rerank_threshold
            ]
            if not survivors:
                logger.warning(
                    "RAG rerank produced zero survivors after threshold=%s; falling back to retrieval order.",
                    config.rerank_threshold,
                )
                return valid

            survivors.sort(key=lambda x: x.score or 0.0, reverse=True)
            return survivors
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return nodes_with_scores

    async def query(
        self,
        question: str,
        filter_files: Optional[List[str]] = None,
        user_id: str = "default",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Query the RAG pipeline with conversation memory and hybrid search.

        Args:
            question: The user's natural language question.
            filter_files: If provided, only search chunks from these source filenames.
            user_id: User identifier to isolate their documents.
            chat_history: Previous conversation turns for context.
        """
        # 1. Hybrid retrieval — returns CHILDREN (small, ~250 chars).
        # We deliberately do NOT auto-merge yet because the cross-encoder
        # tokenizer caps at 512 tokens and parent chunks (~2500 chars)
        # would overflow it.
        #
        # We run sync ``retrieve`` in a worker thread rather than calling
        # ``aretrieve`` because local-mode Qdrant only ships a sync
        # client (RocksDB lock prevents pairing it with AsyncQdrantClient
        # on the same path). ``asyncio.to_thread`` keeps the FastAPI
        # event loop responsive during the retrieval call without
        # requiring an async Qdrant client.
        retriever = self._retrieve_candidate_nodes(question, user_id, filter_files)
        nodes_with_scores = await asyncio.to_thread(retriever.retrieve, question)

        # 2. Defensive cross-encoder rerank on CHILDREN (fits the 512-token
        # window). Identical semantics to the old LangChain pipeline.
        reranked = self._rerank_nodes(question, list(nodes_with_scores))

        # 3. NOW resolve top reranked children → parents from the
        # docstore. Same as the old LangChain MD5 parent-store lookup,
        # implemented natively over LlamaIndex node relationships.
        resolved = self._resolve_parents(reranked, user_id)
        nodes = resolved[: config.max_parents]

        if not nodes:
            return {
                "answer": "I couldn't find enough relevant context in the selected sources to answer that question.",
                "sources": [],
            }

        # 4. Build LLM context from resolved nodes (matches existing format).
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        for nws in nodes:
            node = nws.node
            metadata = node.metadata or {}
            ctype = metadata.get("content_type", "text")
            src = metadata.get("source", "")
            sec = metadata.get("section_title", "")
            title = metadata.get("doc_title", "")
            authors = metadata.get("doc_authors", "")

            header_elements = []
            if title:
                header_elements.append(f"Title: {title}")
            if authors:
                header_elements.append(f"Authors: {authors}")
            header_elements.append(f"File: {src}")
            if sec:
                header_elements.append(f"Section: {sec}")
            header_str = " | ".join(header_elements)

            if ctype == "table":
                context_parts.append(f"[TABLE | {header_str}]:\n{node.text}")
            else:
                context_parts.append(f"[{header_str}]:\n{node.text}")

            score = nws.score if nws.score is not None else 0.0
            sources.append(
                {
                    "source": src,
                    "section": sec,
                    "parser_type": metadata.get("parser_type", "docling"),
                    "score": round(float(score) * 100),
                    "chunk_text": (node.text or "")[:500],
                }
            )

        context = "\n\n".join(context_parts)

        # 5. Build multi-turn messages with last 5 Q&A pairs (10 messages).
        system_msg = {
            "role": "system",
            "content": "You are a scientific research assistant. Answer questions using ONLY the provided context from research papers. Use markdown formatting. If the context doesn't contain enough information, say so clearly.",
        }
        messages = [system_msg]
        if chat_history:
            for msg in chat_history[-10:]:
                messages.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    }
                )

        user_msg = f"""Context from research papers:
{context}

Question: {question}"""
        messages.append({"role": "user", "content": user_msg})

        response = await self._invoke_llm(
            messages=messages,
            timeout_seconds=RAG_QUERY_TIMEOUT_SECONDS,
        )
        return {"answer": response.content.strip(), "sources": sources}

    async def summarize_document(self, text: str, filename: str) -> str:
        """Generate a brief summary of a document using the LLM."""
        try:
            # Use first ~2000 chars for summarization to stay within limits
            excerpt = text[:2000]
            prompt = f"""Summarize this research paper excerpt in exactly 2-3 sentences. Focus on the main topic, methods, and key findings.

Excerpt:
{excerpt}

Summary:"""
            response = await self._invoke_llm(
                prompt=prompt,
                max_retries=1,
                timeout_seconds=RAG_SUMMARY_TIMEOUT_SECONDS,
            )
            return response.content.strip()
        except RAGLLMTimeoutError as e:
            logger.warning(f"Summary generation timed out for {filename}: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Summarization failed for {filename}: {e}")
            return ""

    # ------------------------------------------------------------------
    # File listing / deletion / reset / cleanup
    # ------------------------------------------------------------------

    def list_indexed_files(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """Aggregate per-source info from the docstore (parents + children)."""
        try:
            storage = self._get_user_storage(user_id)
            file_map: Dict[str, Dict[str, Any]] = {}
            for node in storage.storage_context.docstore.docs.values():
                metadata = getattr(node, "metadata", None) or {}
                src = metadata.get("source", "")
                if not src:
                    continue
                if src not in file_map:
                    file_map[src] = {
                        "name": src,
                        "file_type": metadata.get(
                            "file_type", os.path.splitext(src)[1] or ".pdf"
                        ),
                        "chunk_count": 0,
                        "indexed_at": metadata.get("indexed_at", ""),
                        "parser_type": metadata.get("parser_type", "docling"),
                        "authors": metadata.get("doc_authors", ""),
                        "doi": metadata.get("doc_doi", ""),
                        "journal": metadata.get("doc_journal", ""),
                    }
                file_map[src]["chunk_count"] += 1

            return list(file_map.values())

        except Exception as e:
            logger.error(f"Error listing indexed files: {e}")
            return []

    def delete_source(self, filename: str, user_id: str = "default") -> bool:
        """Remove a source completely for a specific user.

        Deletes:
          - All Qdrant points whose ``source`` payload matches ``filename``.
          - All docstore nodes (parents + children) tagged with that source.
          - The uploaded PDF file under ``data/uploads/<user>/``.
        """
        try:
            self._delete_existing_sources(user_id, [filename])
            delete_user_upload_file(user_id, filename)
            self._invalidate_user_storage(user_id)
            return True
        except Exception as e:
            self._invalidate_user_storage(user_id)
            logger.error(f"Error deleting source '{filename}': {e}")
            return False

    def reset_rag(self, user_id: str = "default") -> bool:
        """Permanently delete all indexed data for a specific user."""
        try:
            self._reset_user_qdrant_in_place(user_id)

            persist_dir = self._get_user_persist_dir(user_id)
            if os.path.isdir(persist_dir):
                import shutil
                shutil.rmtree(persist_dir, ignore_errors=True)
                logger.info(f"Cleared docstore folder for user {user_id}")

            delete_user_uploads(user_id)
            self._invalidate_user_storage(user_id)
            return True
        except Exception as e:
            self._invalidate_user_storage(user_id)
            logger.error(f"Error resetting RAG data for user {user_id}: {e}")
            return False

    def cleanup_user(self, user_id: str) -> bool:
        """Delete all data for a user when they close their browser.

        Drops the Qdrant collection, the docstore folder, and the uploads
        folder. Idempotent — safe to call repeatedly.
        """
        success = True
        try:
            self.reset_rag(user_id)
        except Exception as e:
            logger.warning(f"Could not empty user storage cleanly: {e}")
            success = False

        self._invalidate_user_storage(user_id)

        try:
            delete_user_uploads(user_id)
        except Exception as e:
            logger.error(f"Error deleting uploads folder: {e}")
            success = False

        if success:
            logger.info(f"Cleaned up all data for user: {user_id}")
        return success

    # ------------------------------------------------------------------
    # Docling skill (advanced extraction)
    # ------------------------------------------------------------------

    def _get_or_init_docling_converter(self):
        """Lazy-build the Docling DocumentConverter with memory-friendly
        defaults. Cached on ``self._docling_converter`` for reuse.

        Memory-tuning knobs in priority order (per Docling's own
        ``advanced_options`` guidance):

        1. ``layout_batch_size`` / ``table_batch_size`` / ``ocr_batch_size``
           — Docling defaults each to 4 (parallel pages per stage).
           Set to 1 to process pages sequentially. This is the actual
           fix for ``Stage preprocess failed for run 1, pages [N..N+3]``
           errors: each "run" processes a batch of 4 pages, so any
           dense page in a batch can OOM the whole batch.
        2. ``accelerator_options.num_threads`` — Docling defaults to 4
           CPU threads. Setting to 1 reduces concurrency-driven peak
           memory.
        3. ``images_scale`` — page bitmap render resolution. Docling's
           default of 1.0 allocates full-DPI bitmaps; 0.5 quarters that.
        4. ``TableFormerMode.FAST`` instead of ACCURATE — ~5x lighter.

        All env-overridable so a beefier machine can crank performance
        back up without code changes:

          RAG_DOCLING_TABLEFORMER_MODE=accurate
          RAG_DOCLING_IMAGES_SCALE=1.0
          RAG_DOCLING_BATCH_SIZE=4
          RAG_DOCLING_NUM_THREADS=4

        Used by both ``_extract_with_docling`` (plain) and
        ``_process_with_docling_skill`` (HybridChunker).
        """
        if self._docling_converter is not None:
            return self._docling_converter

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
        from docling.datamodel.base_models import InputFormat

        opts = PdfPipelineOptions()
        opts.do_table_structure = True
        opts.table_structure_options.do_cell_matching = False

        # 1. TableFormer mode (env-driven; FAST default)
        tf_mode = os.getenv("RAG_DOCLING_TABLEFORMER_MODE", "fast").strip().lower()
        opts.table_structure_options.mode = (
            TableFormerMode.ACCURATE if tf_mode == "accurate" else TableFormerMode.FAST
        )

        # 2. Per-stage batch sizes. Docling defaults each to 4. Setting
        # to 1 cuts peak memory ~4x at the cost of slower throughput,
        # which is the right trade for memory-constrained environments
        # and is what prevents std::bad_alloc on dense PDFs.
        try:
            batch = max(1, int(os.getenv("RAG_DOCLING_BATCH_SIZE", "1")))
        except ValueError:
            batch = 1
        opts.layout_batch_size = batch
        opts.table_batch_size = batch
        opts.ocr_batch_size = batch  # ignored since do_ocr=False, but consistent.

        # 3. Accelerator: single-threaded by default to keep concurrent
        # allocations low. Docling honors DOCLING_NUM_THREADS env too,
        # but we expose RAG_DOCLING_NUM_THREADS for naming consistency.
        try:
            num_threads = max(1, int(os.getenv("RAG_DOCLING_NUM_THREADS", "1")))
        except ValueError:
            num_threads = 1
        opts.accelerator_options = AcceleratorOptions(
            num_threads=num_threads,
            device=AcceleratorDevice.AUTO,
        )

        # 4. Page bitmap render scale. 1.0 = full DPI; 0.5 = quarter-area
        # = ~4x less memory during page rasterization. Combine with the
        # batch-size knob above for compounding memory savings.
        try:
            opts.images_scale = float(os.getenv("RAG_DOCLING_IMAGES_SCALE", "0.5"))
        except ValueError:
            opts.images_scale = 0.5

        # Image generation flags are False by default — set explicitly
        # so future Docling default changes don't surprise us.
        opts.generate_page_images = False
        opts.generate_picture_images = False
        opts.generate_table_images = False
        opts.generate_parsed_pages = False

        opts.do_ocr = False
        opts.do_code_enrichment = False
        opts.do_formula_enrichment = False

        logger.info(
            "Initializing Docling DocumentConverter "
            "(tableformer=%s, images_scale=%.2f, batch_size=%d, num_threads=%d)",
            opts.table_structure_options.mode.name,
            opts.images_scale,
            batch,
            num_threads,
        )
        self._docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=opts)
            }
        )
        return self._docling_converter

    def _process_with_docling_skill(
        self,
        pdf_path: str,
        source: str,
        pdf_metadata: Dict,
        user_id: str = "default",
    ):
        """Advanced Docling processing using HybridChunker.

        Returns ``(documents, extracted_text)``. Each HybridChunker chunk
        becomes a single LlamaIndex ``Document`` keyed on its section
        breadcrumb; HierarchicalNodeParser handles the parent/child split.
        """
        converter = self._get_or_init_docling_converter()

        abs_path = os.path.abspath(pdf_path)
        result = converter.convert(
            abs_path,
            max_num_pages=config.max_num_pages,
            max_file_size=config.max_file_size,
        )

        if not result or not result.document:
            raise ValueError("Docling returned empty result")

        extracted_text = result.document.export_to_markdown()

        # HybridChunker produces structurally-aware chunks.
        try:
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            _hybrid_available = True
        except ImportError:
            _hybrid_available = False
        if not _hybrid_available:
            logger.warning("Docling chunking components missing, falling back to basic extraction")
            raise ImportError("Docling chunking components missing")

        logger.info("Using Docling HybridChunker for semantic splitting...")
        from transformers import AutoTokenizer
        tokenizer = HuggingFaceTokenizer(
            tokenizer=AutoTokenizer.from_pretrained(config.embedding_model),
            max_tokens=min(config.parent_chunk_size, 512),
        )
        chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
        doc_chunks = list(chunker.chunk(result.document))

        file_ext = os.path.splitext(source)[1].lower() or ".pdf"
        indexed_at = datetime.now(timezone.utc).isoformat()
        doc_title = pdf_metadata.get("title", "")

        documents: List[Any] = []
        for i, chunk in enumerate(doc_chunks):
            chunk_text = chunker.contextualize(chunk)
            headings = chunk.meta.headings or []
            section_title = headings[0] if headings else ""
            cch_header = self._build_cch_header(doc_title, section_title)

            metadata = _sanitize_metadata_dict({
                "source": source,
                "parser_type": "docling_skill",
                "file_type": file_ext,
                "indexed_at": indexed_at,
                "content_type": "text",
                "doc_title": doc_title,
                "doc_authors": pdf_metadata.get("authors", ""),
                "doc_doi": pdf_metadata.get("doi", ""),
                "doc_journal": pdf_metadata.get("journal", ""),
                "page": getattr(chunk.meta.origin, "page_no", 0) if getattr(chunk.meta, "origin", None) else 0,
                "section_title": section_title,
                "headings": headings,
                "cch_header": cch_header,
                "chunk_index": i,
            })
            documents.append(self._make_document(chunk_text, metadata))

        logger.info(
            f"Docling skill produced {len(documents)} documents for {source}; "
            f"hierarchical parser will split them into parents/children."
        )
        return documents, extracted_text

    def _extract_with_docling(self, pdf_path):
        logger.info(f"Starting detailed extraction with Docling for: {pdf_path}")
        try:
            converter = self._get_or_init_docling_converter()

            # Use absolute path to avoid any confusion
            abs_path = os.path.abspath(pdf_path)
            result = converter.convert(
                abs_path,
                max_num_pages=config.max_num_pages,
                max_file_size=config.max_file_size,
            )

            if not result or not result.document:
                logger.error(f"Docling returned empty result for {pdf_path}")
                return None, []

            md_text = result.document.export_to_markdown()

            docling_tables = []
            # In Docling v2, tables are accessible via result.document.tables
            if hasattr(result.document, "tables"):
                for table in result.document.tables:
                    try:
                        # Ensure we are calling the correct export method
                        table_md = table.export_to_markdown(doc=result.document)
                        pnum = 0
                        if hasattr(table, "prov") and table.prov and len(table.prov) > 0:
                            pnum = getattr(table.prov[0], "page_no", 1) # Default to 1 if found
                        docling_tables.append({"content": table_md, "page": pnum})
                    except Exception as te:
                        logger.warning(f"Failed to export table in {pdf_path}: {te}")
                        continue

            logger.info(f"Docling extraction successful for {pdf_path} ({len(md_text)} chars, {len(docling_tables)} tables)")
            return md_text, docling_tables
        except Exception as e:
            logger.error(f"Docling extraction failed for {pdf_path}: {str(e)}", exc_info=True)
            return None, []

    @staticmethod
    def _ensure_pymupdf_layout_int64_patch() -> None:
        """Workaround for pymupdf 1.27.2.3's layout ONNX model.

        The bundled BoxRFDGNN model declares its `edge_index` input as
        ``tensor(int64)``, but the calling code in
        ``pymupdf.layout.onnx.BoxRFDGNN.predict`` sometimes constructs it
        as int32. ONNX Runtime is strict about input types, so on real-world
        multi-page PDFs this raises::

            InvalidArgument: Unexpected input data type.
                Actual: tensor(int32), expected: tensor(int64)

        We wrap ``self.session.run`` inside ``predict`` to coerce any int
        input narrower than int64 up to int64 before the model sees it.
        Idempotent — runs at most once per process.
        """
        try:
            from pymupdf.layout.onnx import BoxRFDGNN as _BRG
        except ImportError as e:
            # Could be: pymupdf_layout uninstalled, OR pymupdf upstream
            # restructured the layout submodule. We log so the no-op is
            # visible rather than mysterious if extractions start failing.
            logger.info(
                "pymupdf.layout.onnx not importable (%s); skipping int64 patch. "
                "If the upstream ONNX dtype bug is unfixed, layout extraction "
                "may crash on multi-page PDFs.", e
            )
            return

        if getattr(_BRG.BoxRFDGNN, "_phytoquery_int64_patch", False):
            return

        if not hasattr(_BRG, "BoxRFDGNN") or not hasattr(_BRG.BoxRFDGNN, "predict"):
            logger.warning(
                "pymupdf.layout.onnx.BoxRFDGNN.predict not found; "
                "upstream may have renamed the layout class. Patch skipped."
            )
            return

        import numpy as _np

        _orig_predict = _BRG.BoxRFDGNN.predict

        def _patched_predict(self, *args, **kwargs):
            _orig_run = self.session.run

            def _cast_run(output_names, input_feed, run_options=None):
                coerced = {}
                for k, v in input_feed.items():
                    if (
                        hasattr(v, "dtype")
                        and v.dtype.kind == "i"
                        and v.dtype.itemsize < 8
                    ):
                        coerced[k] = v.astype(_np.int64)
                    else:
                        coerced[k] = v
                return _orig_run(output_names, coerced, run_options)

            self.session.run = _cast_run
            try:
                return _orig_predict(self, *args, **kwargs)
            finally:
                self.session.run = _orig_run

        _BRG.BoxRFDGNN.predict = _patched_predict
        _BRG.BoxRFDGNN._phytoquery_int64_patch = True
        logger.info(
            "Installed pymupdf_layout int32→int64 coercion patch (one-time)."
        )

    def _extract_with_pymupdf(self, pdf_path):
        """Layout-aware PDF extraction using pymupdf4llm.

        Built on top of PyMuPDF, pymupdf4llm.to_markdown() adds:
          • Multi-column reading order detection
          • Heading hierarchy preserved as Markdown headers (#, ##, …)
          • Tables emitted inline as GFM pipe tables
          • Image positions preserved (text reflows around them)

        Returns (full_text, tables) where ``tables`` is always an empty list:
        tables are now embedded inside ``full_text`` as Markdown so the
        downstream HierarchicalNodeParser handles them naturally without a
        separate index path. The empty list is kept for API compatibility
        with the docling branch which still returns tables separately.
        """
        try:
            import pymupdf4llm

            # One-time runtime fix for pymupdf_layout's ONNX int dtype mismatch.
            self._ensure_pymupdf_layout_int64_patch()

            # page_chunks=True returns per-page dicts so we can preserve the
            # "<!-- Page N -->" markers that the rest of the pipeline relies
            # on for citations and section attribution.
            chunks = pymupdf4llm.to_markdown(pdf_path, page_chunks=True)

            text_parts = []
            for idx, chunk in enumerate(chunks):
                meta = chunk.get("metadata", {}) or {}
                # pymupdf4llm 0.0.27 uses 1-based "page". Newer versions
                # use 0-based "page_number". Handle both, fall back to the
                # enumeration index if neither is present.
                if "page" in meta and meta["page"] is not None:
                    page_num = meta["page"]
                elif "page_number" in meta and meta["page_number"] is not None:
                    page_num = meta["page_number"] + 1
                else:
                    page_num = idx + 1
                text = (chunk.get("text") or "").strip()
                if text:
                    text_parts.append(f"<!-- Page {page_num} -->\n\n{text}")

            full_text = "\n\n".join(text_parts)
            if not full_text.strip():
                logger.warning(
                    f"pymupdf4llm extracted empty text from {pdf_path}"
                )
                return None, []
            return full_text, []
        except Exception as e:
            logger.warning(
                f"pymupdf4llm extraction failed for {pdf_path}: {e}"
            )
            return None, []

    def _detect_sections(self, text):
        """Detect scientific paper sections from plain text or markdown.

        Handles both Docling markdown output (# headers) and PyMuPDF plain text
        by recognizing standard scientific section names.
        """
        # Standard scientific paper section names (case-insensitive)
        SECTION_PATTERNS = [
            r"^(?:Abstract|Summary)\s*$",
            r"^(?:Introduction|Background|Literature Review|Related Work)\s*$",
            r"^(?:Methods|Methodology|Materials and Methods|Experimental(?: Setup)?|Procedure|Protocol)\s*$",
            r"^(?:Results|Findings)\s*$",
            r"^(?:Discussion)\s*$",
            r"^(?:Conclusion|Conclusions)\s*$",
            r"^(?:Acknowledgments?|Acknowledgements?)\s*$",
            r"^(?:References|Bibliography|Literature Cited)\s*$",
            r"^(?:Supplementary(?: Material| Information)?|Appendix(?:es)?)\s*$",
            r"^(?:Declarations?|Funding|Author Contributions|Ethics Statement|Data Availability|Conflicts? of Interest)\s*$",
        ]
        # Combine into one regex for efficiency
        section_regex = re.compile(
            "|".join(SECTION_PATTERNS),
            re.IGNORECASE | re.MULTILINE,
        )

        lines = text.split("\n")
        sections = []
        current = {"title": "Start", "level": 0, "content": [], "start": 0}

        for i, line in enumerate(lines):
            stripped = line.strip()
            is_header = False
            header_level = 0
            header_title = ""

            # 1. Markdown headers (from Docling)
            if stripped.startswith("#"):
                match = re.match(r"^(#{1,4})\s+(.+)$", stripped)
                if match:
                    is_header = True
                    header_level = len(match.group(1))
                    header_title = match.group(2).strip()

            # 2. Numbered sections (e.g., "1. Introduction", "2. Methods")
            if not is_header and re.match(r"^\d+\.?\s+[A-Z]", stripped):
                is_header = True
                header_level = 1
                header_title = stripped

            # 3. ALL CAPS section headers (common in PDFs)
            if not is_header and re.match(r"^[A-Z][A-Z0-9&\s\-\.]{2,}$", stripped) and len(stripped) < 60:
                # Verify it's a known section name
                if section_regex.search(stripped):
                    is_header = True
                    header_level = 1
                    header_title = stripped.title()

            # 4. Title-case section headers on their own line
            if not is_header and len(stripped) < 50 and stripped:
                if section_regex.search(stripped):
                    is_header = True
                    header_level = 1
                    header_title = stripped

            if is_header:
                # Save previous section
                if current["content"]:
                    current["text"] = "\n".join(current["content"])
                    sections.append(current)
                current = {
                    "title": header_title,
                    "level": header_level,
                    "content": [],
                    "start": i,
                }
                continue

            current["content"].append(line)

        if current["content"]:
            current["text"] = "\n".join(current["content"])
            sections.append(current)

        return sections


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def peek_rag_service() -> Optional[RAGService]:
    return _rag_service
