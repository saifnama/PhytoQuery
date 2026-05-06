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
import tempfile
import threading
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
#   from langchain_qdrant import QdrantVectorStore
#   from qdrant_client import QdrantClient
#   from qdrant_client.http import models as qmodels
#   from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownTextSplitter
#   from langchain_experimental.text_splitter import SemanticChunker
#   from sentence_transformers import CrossEncoder
#   from docling.chunking import HybridChunker
#   from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
# ---------------------------------------------------------------------------
# Re-exported for tests that monkeypatch / reference these by module-attr.
from langchain_core.documents import Document  # noqa: F401

try:
    from sentence_transformers import CrossEncoder  # noqa: F401
except Exception:  # allow tests / minimal envs to inject a stub later
    CrossEncoder = None  # type: ignore[assignment]
# ---------------------------------------------------------------------------

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


def _sanitize_documents_for_qdrant(documents):
    from langchain_core.documents import Document
    sanitized = []
    for doc in documents:
        sanitized.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    key: _sanitize_metadata_value(value)
                    for key, value in doc.metadata.items()
                },
            )
        )
    return sanitized


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
    # Single shared local Qdrant DB. Per-user isolation happens via
    # collection naming (``user_<safe_user_id>_<8-char-hash>``) inside
    # this one directory, not via separate folders. Replaces the old
    # ``chroma_dir`` per-user-folder layout.
    qdrant_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "qdrant",
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


from langchain_core.embeddings import Embeddings as _LCEmbeddings


class PhytoQueryEmbeddings(_LCEmbeddings):
    """Custom embeddings with Qwen3-Embedding-4B primary and bge-m3 fallback.

    Inherits from ``langchain_core.embeddings.Embeddings`` so strict
    ``isinstance`` checks (``langchain-qdrant`` does one in
    QdrantVectorStore.__init__) pass. The interface ``embed_documents``
    / ``embed_query`` is unchanged from before — the base class only
    declares those as abstract methods.

    For Qwen3, queries use ``prompt_name="query"`` for instruction-aware
    retrieval; documents are encoded without prompts. Falls back to
    bge-m3 on load failure.
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
        # Cache for per-user vectorstores
        self._vectorstore_cache: Dict[str, Any] = {}
        # Shared Qdrant local client (one DB, many per-user collections).
        self._qdrant_client = None
        self._qdrant_lock = threading.Lock()
        # Cache for Docling converter to avoid reloading models
        self._docling_converter = None
        # Lazy-init semantic child splitter
        self._semantic_splitter = None

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

    @property
    def reranker(self):
        """Lazy-load the reranker on first access."""
        if self._reranker is not ...:
            return self._reranker
        with self._reranker_lock:
            if self._reranker is not ...:
                return self._reranker
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Loading reranker: {config.reranker_model} on {self._device}...")
                kwargs: Dict[str, Any] = {"max_length": config.reranker_max_length}
                if "zerank" in config.reranker_model.lower():
                    compatible, reason = _zerank_runtime_compatible()
                    if not compatible:
                        raise RuntimeError(reason)
                    kwargs["trust_remote_code"] = True

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

                self._reranker = CrossEncoder(config.reranker_model, **kwargs)
            except Exception as e:
                logger.warning(f"Failed to load reranker on {self._device}: {e}")
                if self._device == "mps":
                    try:
                        from sentence_transformers import CrossEncoder
                        logger.info("Retrying reranker load on CPU due to MPS failure...")
                        kwargs = {"max_length": config.reranker_max_length, "device": "cpu"}
                        if "zerank" in config.reranker_model.lower():
                            compatible, reason = _zerank_runtime_compatible()
                            if not compatible:
                                raise RuntimeError(reason)
                            kwargs["trust_remote_code"] = True
                        self._reranker = CrossEncoder(config.reranker_model, **kwargs)
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

    def _get_semantic_splitter(self):
        """Lazy-init SemanticChunker for child-level semantic splitting."""
        if self._semantic_splitter is None:
            from langchain_experimental.text_splitter import SemanticChunker
            logger.info("Initializing SemanticChunker for child splitting...")
            self._semantic_splitter = SemanticChunker(
                self.embeddings,
                breakpoint_threshold_type="standard_deviation",
                breakpoint_threshold_amount=1.5,
            )
        return self._semantic_splitter

    def _split_semantic_children(self, text: str) -> List[str]:
        """Split parent text into semantically coherent child chunks.

        Uses SemanticChunker to group sentences by meaning, then applies
        size guards to ensure chunks stay within configured bounds.
        """
        try:
            splitter = self._get_semantic_splitter()
            raw_chunks = splitter.split_text(text)
        except Exception as e:
            from langchain_text_splitters import MarkdownTextSplitter
            logger.warning(f"SemanticChunker failed, falling back to MarkdownTextSplitter: {e}")
            fallback = MarkdownTextSplitter(
                chunk_size=config.child_chunk_size,
                chunk_overlap=config.child_chunk_overlap,
            )
            return fallback.split_text(text)

        result: List[str] = []
        for chunk in raw_chunks:
            if len(chunk) < config.min_chunk_size:
                continue
            if len(chunk) > config.child_chunk_size * 2:
                # Oversized semantic chunk → fallback to character-based split
                from langchain_text_splitters import RecursiveCharacterTextSplitter
                safety = RecursiveCharacterTextSplitter(
                    chunk_size=config.child_chunk_size,
                    chunk_overlap=config.child_chunk_overlap,
                    separators=["\n\n", "\n", ". ", " ", ""],
                )
                result.extend(safety.split_text(chunk))
            else:
                result.append(chunk)
        return result

    def _get_collection_suffix(self) -> str:
        """Generate a short suffix based on the active embedding model and dimension.

        This ensures Qdrant collections are versioned by embedding config,
        preventing dimension mismatch when switching models.
        """
        model_key = f"{self.embeddings.model_name}:{self.embeddings.model_dim}"
        short_hash = hashlib.md5(model_key.encode()).hexdigest()[:8]
        return short_hash

    def _get_qdrant_client(self):
        """Lazy-init the shared local QdrantClient.

        One client points at ``data/qdrant/``. Per-user isolation is
        achieved via collection names (``user_<safe_user_id>_<8-char-hash>``)
        within that single directory rather than per-user folders. This
        avoids the SQLite-tenant gymnastics that plagued the old Chroma
        path and lets ``delete_collection`` be a single atomic call.
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

    def _ensure_qdrant_collection(self, client, collection_name: str) -> None:
        """Create the per-user Qdrant collection if it doesn't exist."""
        from qdrant_client.http import models as qmodels

        try:
            client.get_collection(collection_name)
            return
        except Exception:
            pass  # collection doesn't exist; fall through to create

        # Embedder must be loaded so model_dim is known. Cosine distance
        # matches Chroma's default and the metric the Qdrant LangChain
        # integration assumes for normalized vectors.
        self.embeddings._ensure_model_loaded()
        vec_dim = config.embedding_dim or self.embeddings.model_dim
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=vec_dim,
                distance=qmodels.Distance.COSINE,
            ),
        )
        logger.info(
            f"Created Qdrant collection {collection_name} (size={vec_dim}, distance=COSINE)"
        )

    def _get_user_collection(self, user_id: str):
        """Get or create a Qdrant-backed LangChain VectorStore for a user.

        Wraps a per-user Qdrant collection in ``QdrantVectorStore`` so the
        rest of the pipeline (``add_documents``, ``similarity_search_with_score``)
        keeps working unchanged. The collection name encodes the
        embedding-model+dim hash so switching models cannot collide.
        """
        from langchain_qdrant import QdrantVectorStore

        if user_id in self._vectorstore_cache:
            return self._vectorstore_cache[user_id]

        safe_user_id = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        model_suffix = self._get_collection_suffix()
        collection_name = f"user_{safe_user_id}_{model_suffix}"

        client = self._get_qdrant_client()
        self._ensure_qdrant_collection(client, collection_name)

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
        self._vectorstore_cache[user_id] = vectorstore
        logger.info(f"Created Qdrant vectorstore wrapper for user: {user_id} (collection={collection_name})")
        return vectorstore

    def _get_user_collection_name(self, user_id: str) -> str:
        """Return the Qdrant collection name for a user (without instantiating the vectorstore)."""
        safe_user_id = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        return f"user_{safe_user_id}_{self._get_collection_suffix()}"

    def _invalidate_user_collection(self, user_id: str) -> None:
        """Drop the cached QdrantVectorStore wrapper for a user.

        Qdrant's local client doesn't have Chroma's SQLite-tenant
        process-singleton pitfalls, so this is just a cache pop — no
        process-wide teardown needed.
        """
        self._vectorstore_cache.pop(user_id, None)

    def _reset_user_chroma_in_place(self, user_id: str) -> None:
        """Drop a user's Qdrant collection (kept under the old name for
        call-site compatibility — the operation is now a clean
        ``delete_collection``, no SQLite gymnastics needed).
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
            self._invalidate_user_collection(user_id)

    # --- Parent-Child Chunking: Parent Store ---

    def _get_parent_store_path(self, user_id: str) -> str:
        safe_user_id = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        return os.path.join(config.qdrant_dir, f"{safe_user_id}_parents.json")

    def _load_parent_store(self, user_id: str) -> Dict[str, str]:
        path = self._get_parent_store_path(user_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_parent_store(self, user_id: str, store: Dict[str, str]) -> None:
        path = self._get_parent_store_path(user_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd, temp_path = tempfile.mkstemp(
            dir=os.path.dirname(path),
            prefix=os.path.basename(path),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(store, f, ensure_ascii=False)
            os.replace(temp_path, path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def _delete_existing_sources(self, user_id: str, source_names: List[str]) -> None:
        """Delete existing chunks for sources that are being re-uploaded."""
        if not source_names:
            return

        from qdrant_client.http import models as qmodels

        # Force collection creation if first upload — saves an extra
        # roundtrip vs catching the "collection not found" exception.
        self._get_user_collection(user_id)

        client = self._get_qdrant_client()
        collection_name = self._get_user_collection_name(user_id)
        unique_sources = sorted(set(source_names))

        try:
            client.delete(
                collection_name=collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        should=[
                            qmodels.FieldCondition(
                                key="metadata.source",
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
            self._cleanup_parent_store(user_id)
        except Exception as exc:
            # Common when the collection has zero matching points; treat as
            # a no-op rather than failing the whole upload.
            logger.info(
                f"Qdrant filter-delete for {user_id} sources={unique_sources} returned: {exc!r}"
            )

    def process_and_index_pdfs_with_texts(
        self,
        pdf_paths: List[str],
        parser_type: str = "pymupdf",
        user_id: str = "default",
    ):
        """Extract, chunk, index, and return per-file extracted text for summaries."""
        total_started = time.perf_counter()
        all_docs = []
        extracted_texts: Dict[str, str] = {}
        source_names = [os.path.basename(path) for path in pdf_paths]
        parse_and_chunk_ms = 0.0
        for path in pdf_paths:
            file_started = time.perf_counter()
            docs, extracted_text = self._process_pdf(
                path,
                user_id=user_id,
                parser_type=parser_type,
            )
            file_elapsed_ms = (time.perf_counter() - file_started) * 1000
            parse_and_chunk_ms += file_elapsed_ms
            all_docs.extend(docs)
            if extracted_text:
                extracted_texts[os.path.basename(path)] = extracted_text
            logger.info(
                "RAG upload phase: parser=%s file=%s parse_and_chunk=%.1fms chunks=%s extracted_chars=%s",
                parser_type,
                os.path.basename(path),
                file_elapsed_ms,
                len(docs),
                len(extracted_text or ""),
            )

        if all_docs:
            sanitized_docs = _sanitize_documents_for_qdrant(all_docs)
            self.embeddings.begin_timing_session()
            index_started = time.perf_counter()
            try:
                self._delete_existing_sources(user_id, source_names)
                vectorstore = self._get_user_collection(user_id)
                vectorstore.add_documents(sanitized_docs)
            except Exception as e:
                # Qdrant corruption / dimension-mismatch signatures.
                # ``Wrong vector size`` fires when the embedding model dim
                # changed between indexings on the same collection.
                # ``not found`` / ``404`` fires when the cached vectorstore
                # references a collection that has since been dropped.
                msg = str(e).lower()
                is_corrupt = (
                    "wrong vector size" in msg
                    or "wrong vector dimension" in msg
                    or ("collection" in msg and "not found" in msg)
                    or "404" in msg
                )
                logger.warning(
                    f"Indexing failed for user {user_id} ({e!r}); "
                    f"{'resetting Qdrant collection and ' if is_corrupt else ''}"
                    f"invalidating cached client and retrying once."
                )
                self._invalidate_user_collection(user_id)
                if is_corrupt:
                    self._reset_user_chroma_in_place(user_id)
                self._delete_existing_sources(user_id, source_names)
                vectorstore = self._get_user_collection(user_id)
                try:
                    vectorstore.add_documents(sanitized_docs)
                except Exception as retry_err:
                    logger.error(
                        f"Retry failed even after cache invalidation for {user_id}: {retry_err}"
                    )
                    raise retry_err from retry_err
            finally:
                embed_stats = self.embeddings.consume_timing_session()

            index_total_ms = (time.perf_counter() - index_started) * 1000
            embed_ms = float(embed_stats.get("total_ms", 0.0))
            embed_calls = int(embed_stats.get("calls", 0))
            embed_texts = int(embed_stats.get("texts", 0))
            store_overhead_ms = max(index_total_ms - embed_ms, 0.0)
            total_elapsed_ms = (time.perf_counter() - total_started) * 1000
            logger.info(
                "RAG upload timings: parser=%s user=%s files=%s chunks=%s %s embed_calls=%s embed_texts=%s total=%.1fms",
                parser_type,
                user_id,
                len(pdf_paths),
                len(all_docs),
                _format_phase_timings({
                    "parse_and_chunk": parse_and_chunk_ms,
                    "embed": embed_ms,
                    "store_overhead": store_overhead_ms,
                    "index_total": index_total_ms,
                }),
                embed_calls,
                embed_texts,
                total_elapsed_ms,
            )
        else:
            logger.info(
                "RAG upload timings: parser=%s user=%s files=%s chunks=0 parse_and_chunk=%.1fms total=%.1fms",
                parser_type,
                user_id,
                len(pdf_paths),
                parse_and_chunk_ms,
                (time.perf_counter() - total_started) * 1000,
            )

        return source_names, extracted_texts

    def _add_parents(self, user_id: str, parent_chunks: List[Dict[str, str]]) -> None:
        store = self._load_parent_store(user_id)
        for p in parent_chunks:
            store[p["parent_id"]] = p["text"]
        self._save_parent_store(user_id, store)

    def _get_parent_text(self, parent_id: str, user_id: str) -> str:
        store = self._load_parent_store(user_id)
        return store.get(parent_id, "")

    def _cleanup_parent_store(self, user_id: str) -> None:
        """Remove parent-store entries no longer referenced by any child.

        Iterates the user's Qdrant collection via ``scroll`` (paginated)
        and collects all referenced ``metadata.parent_id`` values, then
        prunes the JSON parent store to only entries still in use.
        """
        # Force collection creation if needed (idempotent on existing).
        self._get_user_collection(user_id)
        client = self._get_qdrant_client()
        collection_name = self._get_user_collection_name(user_id)

        try:
            active_parent_ids: set = set()
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    metadata = payload.get("metadata", {}) or {}
                    pid = metadata.get("parent_id")
                    if pid:
                        active_parent_ids.add(pid)
                if offset is None:
                    break

            store = self._load_parent_store(user_id)
            new_store = {k: v for k, v in store.items() if k in active_parent_ids}
            self._save_parent_store(user_id, new_store)
        except Exception:
            # Collection might not exist yet or scan can fail mid-way; the
            # parent store is best-effort cleanup, not a correctness concern.
            pass

    def process_and_index_pdfs(
        self,
        pdf_paths: List[str],
        parser_type: str = "pymupdf",
        user_id: str = "default",
    ):
        """Extract, chunk, and index PDFs for a specific user.

        Args:
            pdf_paths: List of PDF file paths to process
            parser_type: "pymupdf" for fast extraction, "docling" for detailed
            user_id: Unique identifier for the user (isolates their documents)
        """
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

    def _process_pdf(self, pdf_path: str, user_id: str = "default", parser_type: str = "pymupdf"):
        """PDF processing pipeline (Preserved from notebook)"""
        source = os.path.basename(pdf_path)

        # 0. Extract metadata (DOI, authors, journal)
        pdf_metadata = self._extract_pdf_metadata(pdf_path)

        # 1. Extract using selected parser
        if parser_type == "docling":
            try:
                return self._process_with_docling_skill(pdf_path, source, pdf_metadata, user_id)
            except Exception as e:
                logger.error(f"Docling skill processing failed, falling back: {e}")
                # Fall through to standard extraction if skill fails
        
        # Fallback/Standard Pipeline (PyMuPDF or Docling fallback)
        if parser_type == "pymupdf":
            full_text, tables = self._extract_with_pymupdf(pdf_path)
        else:
            full_text, tables = self._extract_with_docling(pdf_path)

        if not full_text:
            return [], ""

        # 2. Section detection & Chunking (Regex-based fallback)
        sections = self._detect_sections(full_text)
        use_semantic_children = parser_type != "pymupdf"
        parent_chunks, chunks = self._chunk_by_sections(
            sections,
            tables,
            pdf_metadata,
            source,
            use_semantic_children=use_semantic_children,
        )

        # Store parent chunks for parent-child retrieval
        self._add_parents(user_id, parent_chunks)

        # 3. Deduplication
        unique_chunks = self._deduplicate_chunks(chunks)

        documents = []
        file_ext = os.path.splitext(source)[1].lower() or ".pdf"
        indexed_at = datetime.now(timezone.utc).isoformat()
        for i, chunk in enumerate(unique_chunks):
            meta = chunk.get("metadata", {})
            meta["source"] = source
            meta["chunk_id"] = f"{source}_{i}"
            meta["parser_type"] = parser_type
            meta["file_type"] = file_ext
            meta["indexed_at"] = indexed_at
            meta["total_chunks"] = len(unique_chunks)
            # Add document-level metadata to every chunk
            meta["doc_title"] = pdf_metadata.get("title", "")
            meta["doc_authors"] = pdf_metadata.get("authors", "")
            meta["doc_doi"] = pdf_metadata.get("doi", "")
            meta["doc_journal"] = pdf_metadata.get("journal", "")
            from langchain_core.documents import Document
            documents.append(Document(page_content=chunk["text"], metadata=meta))

        return documents, full_text

    def _hybrid_search(
        self,
        question: str,
        vectorstore,  # langchain_qdrant.QdrantVectorStore (lazy import)
        filter_files: Optional[List[str]] = None,
        k: int = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector similarity + BM25 keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.
        Returns list of dicts with 'doc', 'score' keys.
        """
        from qdrant_client.http import models as qmodels

        if k is None:
            k = config.top_k
        rrf_k = 60  # RRF constant

        # --- 1. Vector similarity search ---
        # langchain-qdrant accepts a qdrant_client Filter directly; no
        # Chroma-style ``$in`` dict syntax. Build a "should" filter
        # (any-of) over selected sources.
        search_kwargs: Dict[str, Any] = {"k": k}
        if filter_files:
            search_kwargs["filter"] = qmodels.Filter(
                should=[
                    qmodels.FieldCondition(
                        key="metadata.source",
                        match=qmodels.MatchValue(value=src),
                    )
                    for src in filter_files
                ]
            )

        try:
            vector_results = vectorstore.similarity_search_with_score(
                question, **search_kwargs
            )
        except Exception:
            # Fallback to regular search if scores not supported
            docs = vectorstore.similarity_search(question, **search_kwargs)
            vector_results = [(doc, 0.5) for doc in docs]

        # --- 2. BM25 keyword search ---
        # Pull every chunk text (for the selected sources, if any) out of
        # Qdrant via paginated scroll, then build a BM25Okapi index over
        # them in-memory. Same algorithm we used over Chroma's
        # ``collection.get``; just a different way to enumerate points.
        bm25_results = []
        try:
            from rank_bm25 import BM25Okapi
            from langchain_core.documents import Document

            client = self._get_qdrant_client()
            collection_name = vectorstore.collection_name

            scroll_filter = None
            if filter_files:
                scroll_filter = qmodels.Filter(
                    should=[
                        qmodels.FieldCondition(
                            key="metadata.source",
                            match=qmodels.MatchValue(value=src),
                        )
                        for src in filter_files
                    ]
                )

            all_docs_text: List[str] = []
            all_metas: List[Dict[str, Any]] = []
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=collection_name,
                    scroll_filter=scroll_filter,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    all_docs_text.append(payload.get("page_content", "") or "")
                    all_metas.append(payload.get("metadata", {}) or {})
                if offset is None:
                    break

            if all_docs_text:
                # Tokenize for BM25
                tokenized = [doc.lower().split() for doc in all_docs_text]
                bm25 = BM25Okapi(tokenized)
                query_tokens = question.lower().split()
                bm25_scores = bm25.get_scores(query_tokens)

                # Get top-k BM25 results
                top_indices = sorted(
                    range(len(bm25_scores)),
                    key=lambda i: bm25_scores[i],
                    reverse=True,
                )[:k]

                for idx in top_indices:
                    if bm25_scores[idx] > 0:
                        doc = Document(
                            page_content=all_docs_text[idx],
                            metadata=all_metas[idx] if idx < len(all_metas) else {},
                        )
                        bm25_results.append((doc, bm25_scores[idx]))
        except ImportError:
            logger.warning(
                "rank-bm25 not installed, falling back to vector-only search"
            )
        except Exception as e:
            logger.warning(f"BM25 search failed, using vector-only: {e}")

        # --- 3. Reciprocal Rank Fusion ---
        chunk_scores: Dict[str, Dict[str, Any]] = {}

        for rank, (doc, score) in enumerate(vector_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {"doc": doc, "rrf": 0.0, "vector_score": score}
            chunk_scores[chunk_id]["rrf"] += rrf_score

        for rank, (doc, score) in enumerate(bm25_results):
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:50])
            rrf_score = 1.0 / (rrf_k + rank + 1)
            if chunk_id not in chunk_scores:
                chunk_scores[chunk_id] = {"doc": doc, "rrf": 0.0, "vector_score": 0.0}
            chunk_scores[chunk_id]["rrf"] += rrf_score

        # Sort by fused score and return top-k
        sorted_results = sorted(
            chunk_scores.values(), key=lambda x: x["rrf"], reverse=True
        )[:k]

        # Normalize scores to 0-100 range for display
        if sorted_results:
            max_score = sorted_results[0]["rrf"]
            for r in sorted_results:
                r["normalized_score"] = (
                    round((r["rrf"] / max_score) * 100) if max_score > 0 else 0
                )

        return sorted_results

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
        vectorstore = self._get_user_collection(user_id)

        # 1. Retrieve many child chunks from vector store (up to 200)
        search_results = self._hybrid_search(
            question, vectorstore, filter_files, k=config.retrieve_k
        )

        # 2. Cross-Encoder Reranking on ALL retrieved children
        reranked_children = []
        if self.reranker and search_results:
            candidate_results = []
            filtered_candidates = []
            try:
                import numpy as np

                candidate_limit = int(config.rerank_candidate_k)
                if candidate_limit <= 0:
                    logger.info(
                        "RAG rerank skipped: non-positive candidate limit=%s",
                        candidate_limit,
                    )
                    reranked_children = search_results
                    candidate_results = []
                else:
                    candidate_results = search_results[:candidate_limit]
                blank_candidate_count = 0
                if candidate_results:
                    for result in candidate_results:
                        page_content = getattr(result.get("doc"), "page_content", "") or ""
                        if not page_content.strip():
                            blank_candidate_count += 1
                            continue
                        filtered_candidates.append(result)

                    candidate_lengths = [len(res["doc"].page_content.strip()) for res in filtered_candidates]
                    logger.info(
                        "RAG rerank boundary: total_candidates=%s filtered_candidates=%s blank_candidates=%s min_chars=%s max_chars=%s",
                        len(candidate_results),
                        len(filtered_candidates),
                        blank_candidate_count,
                        min(candidate_lengths) if candidate_lengths else 0,
                        max(candidate_lengths) if candidate_lengths else 0,
                    )

                    if not filtered_candidates:
                        reranked_children = search_results
                        raise ValueError("No non-empty rerank candidates available")

                    query_text = question.strip()
                    # Guard against empty queries — cross-encoder tokenizers
                    # can produce zero-length sequences for these and the
                    # attention layer then fails on a reshape into
                    # [batch, 0, -1, head_dim] (ambiguous -1 with 0 elements).
                    if not query_text:
                        reranked_children = filtered_candidates
                        raise ValueError("Empty query — skipping rerank")

                    # zerank-2 supports instruction-aware reranking:
                    # prepend the domain instruction to each query in the pair
                    if config.reranker_instruction and "zerank" in config.reranker_model.lower():
                        instructed_query = f"{config.reranker_instruction}\n{query_text}"
                    else:
                        instructed_query = query_text

                    # Drop pairs where either side is whitespace-only — cross-encoder
                    # tokenizers can yield zero-length tensors for these.
                    pairs = []
                    valid_candidates = []
                    for res in filtered_candidates:
                        passage = (res["doc"].page_content or "").strip()
                        if not passage:
                            continue
                        pairs.append([instructed_query, passage])
                        valid_candidates.append(res)
                    if not pairs:
                        reranked_children = filtered_candidates
                        raise ValueError("No tokenizable rerank pairs after filtering")
                    filtered_candidates = valid_candidates

                    rerank_batch_size = max(1, int(config.rerank_batch_size))
                    rerank_scores = []
                    for batch_start in range(0, len(pairs), rerank_batch_size):
                        batch_pairs = pairs[batch_start: batch_start + rerank_batch_size]
                        batch_scores = self.reranker.predict(batch_pairs)
                        rerank_scores.extend(list(batch_scores))

                    if len(rerank_scores) != len(filtered_candidates):
                        raise ValueError(
                            f"Reranker score count mismatch: expected {len(filtered_candidates)} got {len(rerank_scores)}"
                        )

                    scores = np.array(rerank_scores, dtype=float)
                    if not np.isfinite(scores).all():
                        raise ValueError("Reranker returned non-finite scores")

                    for i, score in enumerate(scores):
                        filtered_candidates[i]["rerank_score"] = float(score)

                    # Normalize scores to 0-1 range using min-max
                    min_s, max_s = scores.min(), scores.max()
                    if max_s > min_s:
                        normalized = (scores - min_s) / (max_s - min_s)
                    else:
                        normalized = np.ones_like(scores) * 0.5

                    for i, r in enumerate(filtered_candidates):
                        r["normalized_score"] = round(float(normalized[i]) * 100)

                    # Filter by relevance threshold (>= configured normalized threshold)
                    reranked_children = [
                        r for i, r in enumerate(filtered_candidates)
                        if normalized[i] >= config.rerank_threshold
                    ]
                    if not reranked_children:
                        logger.warning(
                            "RAG rerank produced zero survivors after threshold=%s; falling back to retrieval order.",
                            config.rerank_threshold,
                        )
                        reranked_children = filtered_candidates or search_results
                    else:
                        # Sort by rerank score descending
                        reranked_children.sort(key=lambda x: x["rerank_score"], reverse=True)
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                reranked_children = filtered_candidates or search_results
        else:
            reranked_children = search_results

        # 3. Parent-Child Resolution: resolve filtered children to unique parents
        parent_ids_seen: set[str] = set()
        parent_results: List[Dict[str, Any]] = []

        for result in reranked_children:
            d = result["doc"]
            ctype = d.metadata.get("content_type", "text")
            if ctype != "text":
                # Tables pass through directly
                parent_results.append(result)
                continue

            parent_id = d.metadata.get("parent_id")
            if not parent_id or parent_id in parent_ids_seen:
                continue

            parent_ids_seen.add(parent_id)
            ptext = self._get_parent_text(parent_id, user_id)
            if ptext:
                from langchain_core.documents import Document
                # Create a synthetic result with parent text
                parent_results.append({
                    "doc": Document(
                        page_content=ptext,
                        metadata=d.metadata,
                    ),
                    "rerank_score": result.get("rerank_score", 0),
                    "normalized_score": result.get("normalized_score", 0),
                })
            else:
                parent_results.append(result)

            if len(parent_results) >= config.max_parents:
                break

        # 4. Build LLM context from resolved parents
        context_parts = []
        sources = []
        for result in parent_results:
            d = result["doc"]
            score = result.get("normalized_score", 0)
            ctype = d.metadata.get("content_type", "text")
            src = d.metadata.get("source", "")
            sec = d.metadata.get("section_title", "")

            title = d.metadata.get("doc_title", "")
            authors = d.metadata.get("doc_authors", "")

            # Build a rich header for the LLM context
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
                context_parts.append(f"[TABLE | {header_str}]:\n{d.page_content}")
            else:
                context_parts.append(f"[{header_str}]:\n{d.page_content}")

            parser_type = d.metadata.get("parser_type", "docling")
            sources.append(
                {
                    "source": src,
                    "section": sec,
                    "parser_type": parser_type,
                    "score": score,
                    "chunk_text": d.page_content[:500],  # First 500 chars for preview
                }
            )

        context = "\n\n".join(context_parts)

        if not context_parts:
            return {
                "answer": "I couldn't find enough relevant context in the selected sources to answer that question.",
                "sources": [],
            }

        # Build multi-turn messages for conversation memory
        system_msg = {
            "role": "system",
            "content": "You are a scientific research assistant. Answer questions using ONLY the provided context from research papers. Use markdown formatting. If the context doesn't contain enough information, say so clearly.",
        }

        messages = [system_msg]

        # Add conversation history (last 5 turns = 10 messages max)
        if chat_history:
            history_window = chat_history[-10:]  # Last 5 Q&A pairs
            for msg in history_window:
                messages.append(
                    {
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    }
                )

        # Add current question with context
        user_msg = f"""Context from research papers:
{context}

Question: {question}"""
        messages.append({"role": "user", "content": user_msg})

        response = await self._invoke_llm(messages=messages, timeout_seconds=RAG_QUERY_TIMEOUT_SECONDS)
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
            response = await self._invoke_llm(prompt=prompt, max_retries=1, timeout_seconds=RAG_SUMMARY_TIMEOUT_SECONDS)
            return response.content.strip()
        except RAGLLMTimeoutError as e:
            logger.warning(f"Summary generation timed out for {filename}: {e}")
            return ""
        except Exception as e:
            logger.warning(f"Summarization failed for {filename}: {e}")
            return ""

    def list_indexed_files(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """Query Qdrant for all unique indexed source files for a specific user.

        Iterates the user's Qdrant collection via ``scroll`` (paginated)
        and aggregates per ``metadata.source`` value.

        Short-circuits to ``[]`` if the user's collection doesn't exist
        yet (i.e. no PDFs have been uploaded). We do NOT call
        ``_get_user_collection`` here — that would create an empty
        collection just to scroll it, which is wasteful and was also
        producing a noisy ``Error: Collection ... not found`` log line
        on the very first session call before any upload.
        """
        try:
            client = self._get_qdrant_client()
            collection_name = self._get_user_collection_name(user_id)

            # Existence probe. If missing, the user hasn't indexed
            # anything — return [] silently rather than creating + scrolling.
            try:
                client.get_collection(collection_name)
            except Exception:
                return []

            file_map: Dict[str, Dict[str, Any]] = {}
            offset = None
            while True:
                points, offset = client.scroll(
                    collection_name=collection_name,
                    limit=1000,
                    with_payload=True,
                    with_vectors=False,
                    offset=offset,
                )
                for point in points:
                    payload = point.payload or {}
                    meta = payload.get("metadata", {}) or {}
                    src = meta.get("source", "")
                    if not src:
                        continue
                    if src not in file_map:
                        file_map[src] = {
                            "name": src,
                            "file_type": meta.get(
                                "file_type", os.path.splitext(src)[1] or ".pdf"
                            ),
                            "chunk_count": 0,
                            "indexed_at": meta.get("indexed_at", ""),
                            "parser_type": meta.get("parser_type", "docling"),
                            "authors": meta.get("doc_authors", ""),
                            "doi": meta.get("doc_doi", ""),
                            "journal": meta.get("doc_journal", ""),
                        }
                    file_map[src]["chunk_count"] += 1
                if offset is None:
                    break

            return list(file_map.values())

        except Exception as e:
            logger.error(f"Error listing indexed files: {e}")
            return []

    def delete_source(self, filename: str, user_id: str = "default") -> bool:
        """Remove a source completely: delete its chunks from Qdrant +
        prune the parent store + delete the uploaded file.
        """
        from qdrant_client.http import models as qmodels

        try:
            self._get_user_collection(user_id)  # ensure collection exists
            client = self._get_qdrant_client()
            collection_name = self._get_user_collection_name(user_id)

            client.delete(
                collection_name=collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="metadata.source",
                                match=qmodels.MatchValue(value=filename),
                            )
                        ]
                    )
                ),
            )
            logger.info(
                f"Deleted chunks for '{filename}' from user {user_id}'s Qdrant collection"
            )

            self._cleanup_parent_store(user_id)
            delete_user_upload_file(user_id, filename)
            self._invalidate_user_collection(user_id)
            return True
        except Exception as e:
            self._invalidate_user_collection(user_id)
            logger.error(f"Error deleting source '{filename}': {e}")
            return False

    def reset_rag(self, user_id: str = "default") -> bool:
        """Permanently delete all indexed data for a user.

        Drops the Qdrant collection (one atomic call — no per-id delete
        loop needed), clears the parent-store JSON, and removes uploads.
        """
        try:
            client = self._get_qdrant_client()
            collection_name = self._get_user_collection_name(user_id)
            try:
                client.delete_collection(collection_name=collection_name)
                logger.info(f"Deleted Qdrant collection {collection_name} for user {user_id}")
            except Exception as exc:
                # Collection might not exist yet — benign.
                logger.info(
                    f"Qdrant collection delete for {user_id} skipped: {exc!r}"
                )

            parent_path = self._get_parent_store_path(user_id)
            if os.path.exists(parent_path):
                os.remove(parent_path)
                logger.info(f"Deleted parent store for user {user_id}")

            delete_user_uploads(user_id)
            self._invalidate_user_collection(user_id)
            return True
        except Exception as e:
            self._invalidate_user_collection(user_id)
            logger.error(f"Error resetting RAG data for user {user_id}: {e}")
            return False

    def cleanup_user(self, user_id: str) -> bool:
        """Delete all data for a user when they close their browser.

        Drops the Qdrant collection, the parent-store JSON, and the
        uploads folder. Idempotent and safe to call repeatedly.
        """
        success = True

        try:
            self.reset_rag(user_id)
        except Exception as e:
            logger.warning(f"Could not reset user data cleanly: {e}")
            success = False

        self._invalidate_user_collection(user_id)

        # Belt-and-suspenders: ensure the parent store is gone even if
        # reset_rag's path wasn't reached.
        parent_path = self._get_parent_store_path(user_id)
        try:
            if os.path.exists(parent_path):
                os.remove(parent_path)
                logger.info(f"Deleted parent store: {parent_path}")
        except Exception as e:
            logger.warning(f"Could not delete parent store: {e}")

        try:
            delete_user_uploads(user_id)
        except Exception as e:
            logger.error(f"Error deleting uploads folder: {e}")
            success = False

        if success:
            logger.info(f"Cleaned up all data for user: {user_id}")
        return success

    # --- Helper Methods (Logic preserved from notebook) ---

    def _process_with_docling_skill(self, pdf_path: str, source: str, pdf_metadata: Dict, user_id: str = "default"):
        """Advanced Docling processing using HybridChunker and parent-child chunking.

        Creates parent chunks (contextualized by HybridChunker) and child chunks
        (small chunks with contextual headers for embedding) consistent with PyMuPDF path.
        """
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
        from docling.datamodel.base_models import InputFormat

        if self._docling_converter is None:
            logger.info("Initializing Docling DocumentConverter for Agent Skill...")
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options.do_cell_matching = False
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            pipeline_options.do_ocr = False
            pipeline_options.do_code_enrichment = False
            pipeline_options.do_formula_enrichment = False
            
            self._docling_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )

        abs_path = os.path.abspath(pdf_path)
        result = self._docling_converter.convert(
            abs_path,
            max_num_pages=config.max_num_pages,
            max_file_size=config.max_file_size,
        )
        
        if not result or not result.document:
            raise ValueError("Docling returned empty result")

        extracted_text = result.document.export_to_markdown()

        # Initialize HybridChunker (respects headers and structure)
        try:
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
            _hybrid_available = True
        except ImportError:
            _hybrid_available = False
        if _hybrid_available:
            logger.info("Using Docling HybridChunker for semantic splitting...")
            from transformers import AutoTokenizer
            tokenizer = HuggingFaceTokenizer(
                tokenizer=AutoTokenizer.from_pretrained(config.embedding_model),
                max_tokens=min(config.parent_chunk_size, 512),  # 512 is the sweet spot for standard embedding models
            )
            chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
            doc_chunks = list(chunker.chunk(result.document))
            
            file_ext = os.path.splitext(source)[1].lower() or ".pdf"
            indexed_at = datetime.now(timezone.utc).isoformat()
            
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            safety_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.parent_chunk_size,
                chunk_overlap=config.parent_chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            doc_title = pdf_metadata.get("title", "")
            parent_chunks: List[Dict[str, str]] = []
            all_child_chunks: List[Dict[str, Any]] = []
            
            for i, chunk in enumerate(doc_chunks):
                # Contextualize adds breadcrumbs (section headers) to the text
                chunk_text = chunker.contextualize(chunk)
                
                # Extract section title from headings
                headings = chunk.meta.headings or []
                section_title = headings[0] if headings else ""
                
                # Contextualize() already prepends section breadcrumbs.
                # Only prepend doc_title here to avoid double section headers.
                doc_header = f"{doc_title}\n\n" if doc_title else ""
                
                # Stable parent ID: include source to avoid collisions across docs
                parent_id = hashlib.md5(
                    f"docling::{source}::{section_title}::{chunk_text[:200]}".encode()
                ).hexdigest()
                
                # --- PARENT CHUNK ---
                # Safety: if HybridChunker produced oversized chunk, split it
                if len(chunk_text) > config.parent_chunk_size * 2:
                    parent_texts = safety_splitter.split_text(chunk_text)
                else:
                    parent_texts = [chunk_text]
                
                for p_idx, p_text in enumerate(parent_texts):
                    pid = f"{parent_id}_p{p_idx}"
                    parent_with_header = doc_header + p_text
                    parent_chunks.append({
                        "parent_id": pid,
                        "text": parent_with_header,
                        "section_title": section_title,
                    })
                    
                    # --- CHILD CHUNKS from this parent ---
                    child_texts = self._split_semantic_children(p_text)
                    for c_idx, c_text in enumerate(child_texts):
                        child_with_header = doc_header + c_text
                        all_child_chunks.append({
                            "text": child_with_header,
                            "metadata": {
                                "source": source,
                                "chunk_id": f"{source}_{i}_p{p_idx}_c{c_idx}",
                                "parser_type": "docling_skill",
                                "file_type": file_ext,
                                "indexed_at": indexed_at,
                                "content_type": "text",
                                "doc_title": doc_title,
                                "doc_authors": pdf_metadata.get("authors", ""),
                                "doc_doi": pdf_metadata.get("doi", ""),
                                "doc_journal": pdf_metadata.get("journal", ""),
                                "page": getattr(chunk.meta.origin, "page_no", 0),
                                "section_title": section_title,
                                "headings": headings,
                                "parent_id": pid,
                                "child_index": c_idx,
                            },
                        })
            
            # Store parents and return children for indexing
            self._add_parents(user_id, parent_chunks)
            logger.info(f"Docling skill created {len(parent_chunks)} parents, {len(all_child_chunks)} children for {source}")
            
            # Deduplicate children
            unique_children = self._deduplicate_chunks(all_child_chunks)
            total = len(unique_children)
            for c in unique_children:
                c["metadata"]["total_chunks"] = total
            from langchain_core.documents import Document
            return [Document(page_content=c["text"], metadata=c["metadata"]) for c in unique_children], extracted_text
        else:
            logger.warning("Docling chunking components missing, falling back to basic extraction")
            raise ImportError("Docling chunking components missing")

    def _extract_with_docling(self, pdf_path):
        logger.info(f"Starting detailed extraction with Docling for: {pdf_path}")
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
            from docling.datamodel.base_models import InputFormat

            if self._docling_converter is None:
                logger.info("Initializing Docling DocumentConverter (this may take a moment)...")
                pipeline_options = PdfPipelineOptions()
                pipeline_options.do_table_structure = True
                pipeline_options.table_structure_options.do_cell_matching = False
                pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
                pipeline_options.do_ocr = False
                
                # Disable heavy enrichment features for now to prevent background VLM model downloads
                pipeline_options.do_code_enrichment = False
                pipeline_options.do_formula_enrichment = False
                
                self._docling_converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                    }
                )
            
            # Use absolute path to avoid any confusion
            abs_path = os.path.abspath(pdf_path)
            result = self._docling_converter.convert(
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
        downstream MarkdownTextSplitter handles them naturally without a
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

    def _chunk_by_sections(
        self,
        sections,
        tables,
        doc_metadata: Dict[str, str] = None,
        source: str = "",
        use_semantic_children: bool = True,
    ):
        """Create parent-child hierarchical chunks with Markdown splitting and contextual headers.

        Flow:
        1. Convert each section to Markdown with headers
        2. Split into parent chunks (~2500 chars) using MarkdownTextSplitter
        3. Split into child chunks using SemanticChunker (meaning-based boundaries)
        4. Prepend contextual chunk headers (CCH) to each child before embedding:
           [Document Title] > [Section Title] > [chunk text]

        Tables are indexed directly without parent-child split.
        Returns: (parent_chunks, all_chunks_for_indexing)
        """
        doc_metadata = doc_metadata or {}
        doc_title = doc_metadata.get("title", "")

        # Build contextual header prefix for this document
        def build_header(section_title: str) -> str:
            parts = []
            if doc_title:
                parts.append(doc_title)
            if section_title and section_title != "Start":
                parts.append(section_title)
            if parts:
                return " > ".join(parts) + "\n\n"
            return ""

        # Markdown splitters
        from langchain_text_splitters import MarkdownTextSplitter
        parent_splitter = MarkdownTextSplitter(
            chunk_size=config.parent_chunk_size,
            chunk_overlap=config.parent_chunk_overlap,
        )
        table_splitter = MarkdownTextSplitter(
            chunk_size=config.parent_chunk_size,
            chunk_overlap=config.parent_chunk_overlap,
        )

        parent_chunks: List[Dict[str, str]] = []
        all_chunks: List[Dict[str, Any]] = []

        # Tables: index directly, no parent-child
        for table in tables:
            content = table.get("content", "")
            if content.strip():
                table_md = f"## Table\n\n{content}"
                table_chunks = table_splitter.split_text(table_md)
                for i, tc in enumerate(table_chunks):
                    all_chunks.append(
                        {
                            "text": tc,
                            "metadata": {
                                "content_type": "table",
                                "page": table.get("page", 0),
                                "chunk_part": i + 1,
                            },
                        }
                    )

        # Sections: parent-child hierarchical chunking with Markdown
        for section in sections:
            text = section.get("text", "")
            if not text.strip():
                continue

            section_title = section.get("title", "")
            header = build_header(section_title)

            # Convert section to Markdown with header for the splitter
            # Skip "## Start" as it's not a real section
            if section_title and section_title != "Start":
                section_md = f"## {section_title}\n\n{text}"
                header_prefix = f"## {section_title}\n\n"
            else:
                section_md = text
                header_prefix = ""

            # Stable parent_id: include source to avoid collisions across docs
            parent_id = hashlib.md5(
                f"{source}::{section_title}::{text[:200]}".encode()
            ).hexdigest()

            # --- PARENT CHUNKS ---
            # Split section into parent-sized markdown chunks
            parent_texts = parent_splitter.split_text(section_md)
            for p_idx, p_text in enumerate(parent_texts):
                # Strip the splitter-injected section header to avoid doubles
                if header_prefix and p_text.startswith(header_prefix):
                    p_text = p_text[len(header_prefix):]
                parent_with_header = header + p_text
                pid = f"{parent_id}_p{p_idx}"
                parent_chunks.append(
                    {
                        "parent_id": pid,
                        "text": parent_with_header,
                        "section_title": section_title,
                    }
                )

                # --- CHILD CHUNKS (from this parent) ---
                # PyMuPDF uses a simpler fast child split; Docling keeps semantic splitting.
                if use_semantic_children:
                    child_texts = self._split_semantic_children(p_text)
                else:
                    from langchain_text_splitters import MarkdownTextSplitter
                    child_splitter = MarkdownTextSplitter(
                        chunk_size=config.child_chunk_size,
                        chunk_overlap=config.child_chunk_overlap,
                    )
                    child_texts = child_splitter.split_text(p_text)
                for c_idx, c_text in enumerate(child_texts):
                    # Prepend contextual header to child for embedding
                    child_with_header = header + c_text
                    all_chunks.append(
                        {
                            "text": child_with_header,
                            "metadata": {
                                "section_title": section_title,
                                "content_type": "text",
                                "parent_id": pid,
                                "child_index": c_idx,
                            },
                        }
                    )

        return parent_chunks, all_chunks

    def _deduplicate_chunks(self, chunks):
        unique = []
        seen_hashes = set()
        for chunk in chunks:
            text = chunk.get("text", "")
            if not text.strip():
                continue
            norm = re.sub(r"\s+", " ", text.lower().strip())
            h = hashlib.md5(norm.encode()).hexdigest()
            if h not in seen_hashes:
                unique.append(chunk)
                seen_hashes.add(h)
        return unique

_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def peek_rag_service() -> Optional[RAGService]:
    return _rag_service
