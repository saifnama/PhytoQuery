import os
# Suppress transformers tokenizer sequence length warnings globally
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
import re
import gc
import time
import hashlib
import json
import pickle
import asyncio
import logging
import tempfile
import threading
from importlib import metadata as importlib_metadata
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from backend.core.http_client import HttpClientManager
from backend.core.rag_storage import (
    delete_user_upload_file,
    delete_user_uploads,
    get_user_markdown_file_path,
)

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
    # Parallel-upload tuning (rag_engine.py:process_and_index_pdfs_with_texts)
    # ``upload_workers`` — concurrent parser threads. PyMuPDF and
    # Docling both release the GIL during their hot paths so threads
    # scale linearly up to ~4-8 cores. Set to 1 to force sequential
    # parsing.
    # ``index_flush_size`` — embed+insert is flushed every N parsed
    # files so a 1000-PDF upload doesn't have to hold every chunk in
    # RAM and a mid-job crash leaves earlier files indexed.
    upload_workers: int = 4
    index_flush_size: int = 50
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
        response_format: Optional[Dict[str, Any]] = None,
    ):
        """Invoke the LLM with either a simple prompt or a full messages list.

        Args:
            prompt: Simple string prompt (converted to single user message).
            messages: Full messages list for multi-turn conversations.
            response_format: When set to ``{"type": "json_object"}`` the
                LLM is forced into JSON mode. Used by the citation
                extraction pass (Pydantic-validated downstream).
                Translated transparently per provider:
                  - Ollama: payload.format = "json"
                  - OpenAI-compatible (Groq/OpenRouter): payload.response_format
                Default ``None`` preserves the prior free-text behavior.
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
            if response_format is not None:
                # Ollama uses a top-level ``format`` field; "json" enables
                # grammar-constrained JSON output.
                payload["format"] = "json"
        else:  # OpenAI-compatible: openrouter, groq
            payload = {
                "model": self.model,
                "messages": msg_list,
                "temperature": self.temperature,
            }
            if response_format is not None:
                payload["response_format"] = response_format
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
        # Explicit utf-8 — payloads carry scientific Unicode (U+2212
        # minus sign, U+00B1 ±, Greek letters, em-dashes) that the
        # Windows-default cp1252 codec cannot encode. Without this,
        # the file write itself raises UnicodeEncodeError which then
        # propagates back up to the caller and looks like an LLM
        # failure instead of a logging-side issue.
        with open("rag_error.log", "w", encoding="utf-8") as f:
            f.write(err_msg)
        logger.error(f"All retries failed for RAG LLM: {last_exception}")
        raise last_exception

    async def astream(
        self,
        prompt: str = None,
        messages: list = None,
    ):
        """Async-iterate streamed text chunks from the LLM.

        Yields plain string deltas as they arrive. Caller accumulates.

        Format handling:
          - Ollama streams newline-delimited JSON; each line carries
            ``message.content`` and a final ``done: true`` marker.
          - OpenAI-compatible (OpenRouter / Groq) streams SSE; each
            ``data: {...}`` line has ``choices[0].delta.content`` and
            ends with ``data: [DONE]``.
        """
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
                "stream": True,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            }
        else:  # OpenAI-compatible: openrouter, groq
            payload = {
                "model": self.model,
                "messages": msg_list,
                "temperature": self.temperature,
                "stream": True,
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}

        client = await HttpClientManager.get_client()
        kwargs = {"json": payload, "timeout": None}
        if self.provider != "ollama":
            kwargs["headers"] = headers

        async with client.stream("POST", self.url, **kwargs) as response:
            if self.provider != "ollama" and response.status_code == 401:
                env_var = {
                    "groq": "RAG_GROQ_API_KEY",
                    "openrouter": "RAG_OPENROUTER_API_KEY",
                }.get(self.provider, "the provider's API key")
                raise RAGProviderAuthError(
                    f"{self.provider.title()} authentication failed. "
                    f"Check {env_var} or configure RAG_OLLAMA_URL as a fallback."
                )

            # Surface upstream error bodies. Without aread() the body
            # is still a streaming iterator, so the default
            # raise_for_status() message hides the actual provider
            # error JSON (e.g. Groq's "messages must alternate roles"
            # or context-window overflow), making 400s impossible to
            # diagnose from logs alone.
            if response.status_code >= 400:
                await response.aread()
                body_text = response.text or "<empty body>"
                logger.error(
                    "LLM stream %s from %s: %s",
                    response.status_code,
                    self.url,
                    body_text[:2000],
                )
                raise RuntimeError(
                    f"{self.provider} streaming failed "
                    f"({response.status_code}): {body_text[:500]}"
                )

            async for raw_line in response.aiter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line:
                    continue

                if self.provider == "ollama":
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg = data.get("message") or {}
                    chunk = msg.get("content", "")
                    if chunk:
                        yield chunk
                    if data.get("done"):
                        break
                else:
                    if not line.startswith("data:"):
                        continue
                    payload_str = line[len("data:"):].strip()
                    if not payload_str:
                        continue
                    if payload_str == "[DONE]":
                        break
                    try:
                        data = json.loads(payload_str)
                    except json.JSONDecodeError:
                        continue
                    try:
                        choices = data.get("choices") or []
                        if not choices:
                            continue
                        delta = choices[0].get("delta") or {}
                        chunk = delta.get("content") or ""
                        if chunk:
                            yield chunk
                    except (KeyError, IndexError, TypeError):
                        continue


def _quote_matches_chunk(quote: str, chunk_text: str) -> bool:
    """Loose verbatim check used in citation validation.

    Returns True iff ``quote`` appears in ``chunk_text`` after
    whitespace + case normalization, OR the longest common substring
    covers ≥80% of the quote length. The looser fallback handles
    LLMs that paraphrase one or two words while quoting.
    Pure stdlib (``difflib``) — no extra dependency.
    """
    if not quote or not chunk_text:
        return False
    norm_quote = re.sub(r"\s+", " ", quote.lower()).strip()
    norm_chunk = re.sub(r"\s+", " ", chunk_text.lower())
    if not norm_quote or len(norm_quote) > len(norm_chunk):
        return False
    if norm_quote in norm_chunk:
        return True
    from difflib import SequenceMatcher

    matcher = SequenceMatcher(None, norm_quote, norm_chunk, autojunk=False)
    longest = matcher.find_longest_match(0, len(norm_quote), 0, len(norm_chunk))
    return longest.size >= len(norm_quote) * 0.8


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

    async def _invoke_llm(
        self,
        *,
        prompt: str = None,
        messages: list = None,
        timeout_seconds: Optional[float] = None,
        max_retries: int = 3,
        response_format: Optional[Dict[str, Any]] = None,
    ):
        # Progressive-drop fallback for invokers (notably test fakes
        # and older shims) that don't accept every kwarg. We try the
        # richest call first and on each ``TypeError: got an
        # unexpected keyword argument`` retry without that kwarg.
        attempts = [
            dict(
                prompt=prompt,
                messages=messages,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
                response_format=response_format,
            ),
            dict(
                prompt=prompt,
                messages=messages,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            ),
            dict(
                prompt=prompt,
                messages=messages,
                max_retries=max_retries,
            ),
            dict(prompt=prompt, messages=messages),
        ]
        last_type_error: Optional[TypeError] = None
        for kwargs in attempts:
            try:
                return await self.llm.invoke(**kwargs)
            except TypeError as exc:
                if "unexpected keyword argument" not in str(exc):
                    raise
                last_type_error = exc
                continue
        # All progressive drops still raised TypeError — surface the
        # last one so the failure is visible.
        raise last_type_error  # type: ignore[misc]

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
            try:
                self._invalidate_bm25(user_id)
            except Exception as e:
                logger.warning(
                    f"BM25 cache invalidation after collection reset failed for {user_id}: {e}"
                )

    # --- Parent-Child Chunking: Parent Store ---

    def _get_parent_store_path(self, user_id: str) -> str:
        safe_user_id = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        return os.path.join(config.qdrant_dir, f"{safe_user_id}_parents.json")

    def _load_parent_store(self, user_id: str) -> Dict[str, Any]:
        """Load the per-user parent store from disk.

        Values can be either a bare ``str`` (legacy entries from
        before offset-tracking shipped) or a ``dict`` carrying
        ``{text, body_start, body_end, page, section_title}`` (current
        format). Callers that need normalization should go through
        ``_get_parent_data``.
        """
        path = self._get_parent_store_path(user_id)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_parent_store(self, user_id: str, store: Dict[str, Any]) -> None:
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

    # --- BM25 sparse-index cache --------------------------------
    # Hybrid retrieval combines dense vectors + BM25 keyword
    # ranking. The dense side scales fine to 100K+ chunks via
    # Qdrant HNSW; the BM25 side does not — naively rebuilding
    # ``BM25Okapi`` on every query means scrolling all chunks out
    # of Qdrant, tokenizing, and indexing them per-call (5-30s on
    # 1000 papers). We instead build BM25 ONCE after each upload
    # and cache it on disk + in memory. The cache is invalidated
    # on every add/delete so it can never be stale.

    def _get_bm25_cache_path(self, user_id: str) -> str:
        safe_user_id = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        suffix = self._get_collection_suffix()
        return os.path.join(
            config.qdrant_dir, f"{safe_user_id}_{suffix}_bm25.pkl"
        )

    def _invalidate_bm25(self, user_id: str) -> None:
        """Drop the BM25 cache (in-memory + on-disk) for ``user_id``.
        Called after every upload commit and every source delete."""
        if not hasattr(self, "_bm25_cache"):
            self._bm25_cache: Dict[str, Any] = {}
        self._bm25_cache.pop(user_id, None)
        path = self._get_bm25_cache_path(user_id)
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            logger.warning(f"BM25 cache file delete failed for {user_id}: {e}")

    def _build_bm25_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Scroll the user's Qdrant collection, tokenize every
        chunk's ``page_content``, build a ``BM25Okapi`` index, and
        persist the (texts, metas, tokenized) trio to disk so we
        never have to re-scroll for keyword search.

        Returns the in-memory cache entry on success or ``None`` if
        the collection is empty / unavailable.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed; BM25 disabled")
            return None
        try:
            from qdrant_client.http import models as qmodels  # noqa: F401
        except ImportError:
            return None

        client = self._get_qdrant_client()
        collection_name = self._get_user_collection_name(user_id)

        all_texts: List[str] = []
        all_metas: List[Dict[str, Any]] = []
        offset = None
        try:
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
                    all_texts.append(payload.get("page_content", "") or "")
                    all_metas.append(payload.get("metadata", {}) or {})
                if offset is None:
                    break
        except Exception as e:
            logger.warning(f"BM25 scroll failed for {user_id}: {e}")
            return None

        if not all_texts:
            return None

        tokenized = [t.lower().split() for t in all_texts]
        bm25 = BM25Okapi(tokenized)

        entry = {
            "bm25": bm25,
            "tokenized": tokenized,
            "texts": all_texts,
            "metas": all_metas,
            "size": len(all_texts),
        }

        # Persist tokenized + texts + metas (NOT the BM25Okapi
        # object itself — pickling rank_bm25's internals is not
        # version-stable across releases). Reload rebuilds BM25Okapi
        # from tokenized in O(N).
        path = self._get_bm25_cache_path(user_id)
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            fd, temp_path = tempfile.mkstemp(
                dir=os.path.dirname(path),
                prefix=os.path.basename(path),
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    pickle.dump(
                        {
                            "tokenized": tokenized,
                            "texts": all_texts,
                            "metas": all_metas,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                os.replace(temp_path, path)
            finally:
                if os.path.exists(temp_path):
                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"BM25 cache persist failed for {user_id}: {e}")

        if not hasattr(self, "_bm25_cache"):
            self._bm25_cache = {}
        self._bm25_cache[user_id] = entry
        logger.info(
            f"Built BM25 index for {user_id}: {len(all_texts)} chunks (cached on disk)"
        )
        return entry

    def _load_bm25_for_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Return the cached BM25 entry for ``user_id``.

        Layered lookup:
          1. In-memory cache (process lifetime).
          2. On-disk pickle (persists across restarts).
          3. Build from Qdrant scroll (slow, only on first query
             after upload or after a process restart with no pickle).
        """
        if not hasattr(self, "_bm25_cache"):
            self._bm25_cache = {}
        if user_id in self._bm25_cache:
            return self._bm25_cache[user_id]

        path = self._get_bm25_cache_path(user_id)
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    raw = pickle.load(f)
                from rank_bm25 import BM25Okapi
                tokenized = raw.get("tokenized") or []
                if tokenized:
                    entry = {
                        "bm25": BM25Okapi(tokenized),
                        "tokenized": tokenized,
                        "texts": raw.get("texts") or [],
                        "metas": raw.get("metas") or [],
                        "size": len(tokenized),
                    }
                    self._bm25_cache[user_id] = entry
                    return entry
            except Exception as e:
                logger.warning(
                    f"BM25 cache load failed for {user_id}: {e}; rebuilding"
                )
                # Fall through to build path
                try:
                    os.remove(path)
                except Exception:
                    pass

        return self._build_bm25_for_user(user_id)

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
            # Any source deletion makes the cached BM25 stale —
            # drop it. Next query rebuilds + re-persists. The
            # post-upload path also invalidates, but covering both
            # entry points means one-off deletes (without a
            # follow-on upload) can't leave stale BM25 state.
            try:
                self._invalidate_bm25(user_id)
            except Exception as e:
                logger.warning(
                    f"BM25 cache invalidation after delete failed for {user_id}: {e}"
                )
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
        """Extract, chunk, and index a batch of PDFs at scale.

        At-scale design (1000+ PDF uploads):
          1. **Parallel parse** — files are parsed/chunked
             concurrently via a ``ThreadPoolExecutor``. Both
             PyMuPDF (C extension) and Docling (releases GIL on
             heavy ML work) benefit from threading without needing
             process pools.
          2. **Per-file isolation** — a single corrupt or
             unparseable PDF cannot kill the whole job. Each file
             runs inside try/except; failures are logged and
             collected in ``failed_files`` but the batch continues.
          3. **Batched flush** — child documents and parents are
             flushed every ``index_flush_size`` files (default 50)
             rather than accumulating the full batch in RAM. Caps
             peak memory and means partial work survives a crash.
          4. **Single delete pass** — all source names are pre-
             deleted up front so the per-batch ``add_documents``
             calls can be straight inserts.
          5. **BM25 cache rebuild** — once the upload commits, the
             persistent BM25 index is invalidated so the next
             query rebuilds it from the fresh Qdrant state. Cached
             on disk after first build so subsequent queries skip
             the scroll+tokenize cost.

        Returns ``(source_names, extracted_texts)``. Sources that
        failed to parse are still listed in ``source_names`` (they
        were uploaded), but they will not appear in
        ``extracted_texts``.
        """
        total_started = time.perf_counter()
        extracted_texts: Dict[str, str] = {}
        source_names = [os.path.basename(path) for path in pdf_paths]
        failed_files: List[Dict[str, str]] = []

        # Up-front: clear any prior copies of these sources so the
        # batched inserts below are pure additions.
        self._delete_existing_sources(user_id, source_names)

        # Pre-init lazy resources that ``_process_pdf`` would
        # otherwise initialize inside a thread (would race).
        if parser_type == "docling":
            try:
                _ = self._docling_converter  # property/lazy attr — touch under lock
            except Exception:
                pass
        # Semantic splitter is lazy-init inside _split_semantic_children;
        # touch it once on the main thread to win the race.
        try:
            self._get_semantic_splitter()
        except Exception:
            # Splitter is optional; pymupdf path doesn't use it.
            pass

        # Parallelism — bounded by config.upload_workers (default 4).
        # PyMuPDF and Docling both release the GIL on their hot
        # paths so threads scale ~linearly with cores up to ~4-8.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        workers = max(1, int(getattr(config, "upload_workers", 4)))
        flush_size = max(1, int(getattr(config, "index_flush_size", 50)))

        def _parse_one(path: str) -> Dict[str, Any]:
            file_started = time.perf_counter()
            source = os.path.basename(path)
            try:
                docs, extracted_text, parent_chunks = self._process_pdf(
                    path, user_id=user_id, parser_type=parser_type,
                )
                return {
                    "ok": True,
                    "path": path,
                    "source": source,
                    "docs": docs,
                    "parents": parent_chunks,
                    "text": extracted_text or "",
                    "ms": (time.perf_counter() - file_started) * 1000,
                }
            except Exception as e:
                logger.exception(f"Parse failed for {source}: {e}")
                return {
                    "ok": False,
                    "path": path,
                    "source": source,
                    "error": str(e)[:300],
                    "ms": (time.perf_counter() - file_started) * 1000,
                }

        parse_and_chunk_ms = 0.0
        embed_ms = 0.0
        embed_calls = 0
        embed_texts = 0
        index_total_ms = 0.0
        total_chunk_count = 0

        # Buffers flushed every ``flush_size`` files.
        buf_docs: List[Any] = []
        buf_parents: List[Dict[str, Any]] = []
        buf_files = 0

        def _flush() -> None:
            nonlocal buf_docs, buf_parents, buf_files
            nonlocal embed_ms, embed_calls, embed_texts, index_total_ms
            nonlocal total_chunk_count
            if not buf_docs and not buf_parents:
                buf_docs, buf_parents, buf_files = [], [], 0
                return
            # 1. Persist parents for THIS batch (read JSON once,
            # mutate, write once).
            if buf_parents:
                self._add_parents(user_id, buf_parents)
            # 2. Embed + insert children for THIS batch.
            if buf_docs:
                sanitized = _sanitize_documents_for_qdrant(buf_docs)
                self.embeddings.begin_timing_session()
                index_started = time.perf_counter()
                try:
                    vectorstore = self._get_user_collection(user_id)
                    vectorstore.add_documents(sanitized)
                except Exception as e:
                    # Same recovery path as before — Qdrant
                    # dimension/collection errors recoverable by
                    # invalidating cache + (if corrupt) wiping the
                    # collection.
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
                    vectorstore = self._get_user_collection(user_id)
                    vectorstore.add_documents(sanitized)
                finally:
                    embed_stats = self.embeddings.consume_timing_session()
                index_total_ms += (time.perf_counter() - index_started) * 1000
                embed_ms += float(embed_stats.get("total_ms", 0.0))
                embed_calls += int(embed_stats.get("calls", 0))
                embed_texts += int(embed_stats.get("texts", 0))
                total_chunk_count += len(buf_docs)
            buf_docs, buf_parents, buf_files = [], [], 0

        # Drive parse in parallel; consume results in COMPLETION
        # order (not submission order) so straggling Docling files
        # don't stall the flush pipeline.
        if workers > 1 and len(pdf_paths) > 1:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_path = {pool.submit(_parse_one, p): p for p in pdf_paths}
                for fut in as_completed(future_to_path):
                    res = fut.result()
                    parse_and_chunk_ms += float(res.get("ms", 0.0))
                    if not res["ok"]:
                        failed_files.append(
                            {"source": res["source"], "error": res["error"]}
                        )
                        continue
                    docs = res["docs"]
                    parents = res["parents"]
                    if res.get("text"):
                        extracted_texts[res["source"]] = res["text"]
                    logger.info(
                        "RAG upload phase: parser=%s file=%s parse_and_chunk=%.1fms chunks=%s extracted_chars=%s",
                        parser_type,
                        res["source"],
                        res["ms"],
                        len(docs),
                        len(res.get("text") or ""),
                    )
                    buf_docs.extend(docs)
                    buf_parents.extend(parents)
                    buf_files += 1
                    if buf_files >= flush_size:
                        _flush()
        else:
            # Sequential path — single thread, single file, or
            # explicit ``upload_workers=1``.
            for path in pdf_paths:
                res = _parse_one(path)
                parse_and_chunk_ms += float(res.get("ms", 0.0))
                if not res["ok"]:
                    failed_files.append(
                        {"source": res["source"], "error": res["error"]}
                    )
                    continue
                docs = res["docs"]
                parents = res["parents"]
                if res.get("text"):
                    extracted_texts[res["source"]] = res["text"]
                logger.info(
                    "RAG upload phase: parser=%s file=%s parse_and_chunk=%.1fms chunks=%s extracted_chars=%s",
                    parser_type, res["source"], res["ms"],
                    len(docs), len(res.get("text") or ""),
                )
                buf_docs.extend(docs)
                buf_parents.extend(parents)
                buf_files += 1
                if buf_files >= flush_size:
                    _flush()

        # Final flush — anything still in buffers.
        _flush()

        # BM25 cache: drop the stale persisted index now that
        # Qdrant has new content. Next query will rebuild + persist.
        try:
            self._invalidate_bm25(user_id)
        except Exception as e:
            logger.warning(f"BM25 cache invalidation failed for {user_id}: {e}")

        store_overhead_ms = max(index_total_ms - embed_ms, 0.0)
        total_elapsed_ms = (time.perf_counter() - total_started) * 1000
        if total_chunk_count:
            logger.info(
                "RAG upload timings: parser=%s user=%s files=%s ok=%s failed=%s chunks=%s %s embed_calls=%s embed_texts=%s total=%.1fms",
                parser_type,
                user_id,
                len(pdf_paths),
                len(pdf_paths) - len(failed_files),
                len(failed_files),
                total_chunk_count,
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
                "RAG upload timings: parser=%s user=%s files=%s ok=%s failed=%s chunks=0 parse_and_chunk=%.1fms total=%.1fms",
                parser_type,
                user_id,
                len(pdf_paths),
                len(pdf_paths) - len(failed_files),
                len(failed_files),
                parse_and_chunk_ms,
                total_elapsed_ms,
            )
        if failed_files:
            logger.warning(
                "RAG upload completed with %d failures: %s",
                len(failed_files),
                [f["source"] for f in failed_files],
            )

        return source_names, extracted_texts

    def _add_parents(self, user_id: str, parent_chunks: List[Dict[str, Any]]) -> None:
        """Persist parents to disk.

        Each parent is stored as a dict carrying the chunk text plus
        optional offset metadata. The offsets pin the parent's body
        text to a precise ``[body_start, body_end)`` char range in
        the saved paper markdown — used at render time to anchor
        citation highlights without fuzzy matching.

        Backwards compatible: writes the new dict shape, but readers
        in this module also accept the legacy bare-string shape from
        parent stores written by older builds.
        """
        store = self._load_parent_store(user_id)
        for p in parent_chunks:
            entry: Dict[str, Any] = {"text": p["text"]}
            for key in ("body_start", "body_end", "page", "section_title"):
                if p.get(key) is not None:
                    entry[key] = p[key]
            store[p["parent_id"]] = entry
        self._save_parent_store(user_id, store)

    def _get_parent_text(self, parent_id: str, user_id: str) -> str:
        """Backwards-compat shim: return just the text body. Used by
        callers that don't need the offsets."""
        data = self._get_parent_data(parent_id, user_id)
        return data.get("text", "") if data else ""

    def _get_parent_data(self, parent_id: str, user_id: str) -> Dict[str, Any]:
        """Return the parent entry as a dict regardless of which
        on-disk shape produced it. Legacy bare-string entries are
        normalized to ``{"text": <str>}`` so callers don't branch."""
        store = self._load_parent_store(user_id)
        raw = store.get(parent_id)
        if raw is None:
            return {}
        if isinstance(raw, str):
            return {"text": raw}
        if isinstance(raw, dict):
            return raw
        # Defensive: unknown shape — coerce to text only
        return {"text": str(raw)}

    @staticmethod
    def _find_page_for_offset(full_text: str, offset: int) -> Optional[int]:
        """Recover the 1-based page number for a char offset inside
        ``full_text`` by scanning ``<!-- Page N -->`` markers (the
        per-page boundaries pymupdf4llm emits at extraction time).

        Returns the page of the LAST marker preceding ``offset``, or
        ``None`` if the offset precedes any marker / no markers exist
        / offset is invalid. Used to enrich chunk metadata so the
        citation panel can show 'p. N' without re-scanning the
        whole document at query time.
        """
        if offset is None or offset < 0 or not full_text:
            return None
        page_pattern = re.compile(r"<!--\s*Page\s+(\d+)\s*-->")
        last_page: Optional[int] = None
        scan_to = min(max(offset, 1), len(full_text))
        for match in page_pattern.finditer(full_text, 0, scan_to):
            try:
                last_page = int(match.group(1))
            except (TypeError, ValueError):
                continue
        return last_page

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
            return [], "", []

        # Persist the extracted markdown so the citation preview
        # panel can serve it via /api/chat/files/{name}/markdown.
        # Best-effort: failures are logged inside the helper.
        self._save_paper_markdown(user_id, source, full_text)

        # 2. Section detection & Chunking (Regex-based fallback)
        sections = self._detect_sections(full_text)
        use_semantic_children = parser_type != "pymupdf"
        parent_chunks, chunks = self._chunk_by_sections(
            sections,
            tables,
            pdf_metadata,
            source,
            use_semantic_children=use_semantic_children,
            # Pass the full markdown so each parent can record an
            # exact char offset for the citation-highlight panel.
            full_text=full_text,
        )

        # Parents are returned to the caller (not written here) so
        # the upload orchestrator can batch them at flush time. This
        # is critical for thread-safe parallel parsing — each thread
        # produces its own parent_chunks; the orchestrator merges
        # them once per flush, avoiding the read-mutate-write race
        # over the per-user parent JSON.

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

        return documents, full_text, parent_chunks

    def _hybrid_search(
        self,
        question: str,
        vectorstore,  # langchain_qdrant.QdrantVectorStore (lazy import)
        filter_files: Optional[List[str]] = None,
        k: int = None,
        user_id: Optional[str] = None,
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

        # --- 2. BM25 keyword search (cached) ---
        # The BM25 index is built once after each upload and cached
        # on disk + in memory; per-query cost is just ``get_scores``
        # over the pre-tokenized corpus, no Qdrant scroll. The cache
        # is invalidated by ``_invalidate_bm25`` whenever sources
        # are added or removed (see process_and_index_pdfs_with_texts
        # and _delete_existing_sources), so it can never be stale.
        # ``filter_files`` is applied as a post-filter on the
        # pre-built corpus rather than re-scrolling — at scale this
        # is the difference between sub-millisecond and many-second
        # query latency.
        bm25_results = []
        try:
            from langchain_core.documents import Document
            entry = self._load_bm25_for_user(user_id) if user_id else None
            if entry is not None:
                bm25 = entry["bm25"]
                texts = entry["texts"]
                metas = entry["metas"]
                query_tokens = question.lower().split()
                bm25_scores = bm25.get_scores(query_tokens)

                # Apply optional source filter as a post-filter mask.
                allow_idx: Optional[set] = None
                if filter_files:
                    allowed = set(filter_files)
                    allow_idx = {
                        i for i, m in enumerate(metas)
                        if (m or {}).get("source") in allowed
                    }

                # Top-k by BM25 score, dropping zero-score chunks.
                if allow_idx is not None:
                    candidate_indices = [i for i in allow_idx if bm25_scores[i] > 0]
                else:
                    candidate_indices = [
                        i for i in range(len(bm25_scores)) if bm25_scores[i] > 0
                    ]
                candidate_indices.sort(
                    key=lambda i: bm25_scores[i], reverse=True
                )
                top_indices = candidate_indices[:k]

                for idx in top_indices:
                    doc = Document(
                        page_content=texts[idx],
                        metadata=metas[idx] if idx < len(metas) else {},
                    )
                    bm25_results.append((doc, float(bm25_scores[idx])))
        except Exception as e:
            logger.warning(f"BM25 cached search failed, using vector-only: {e}")

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

    def _get_reranker_max_tokens(self) -> int:
        """Return the loaded reranker model's *actual* max input
        token count.

        Critical: this is NOT ``config.reranker_max_length``. That
        config value is what the model is *constructed* with and
        can be set higher than the underlying transformer's
        ``max_position_embeddings``. When that happens, the
        tokenizer happily produces up to the configured length but
        the model raises ``tensor a (N) must match tensor b (512)``
        at predict time. We need the smaller of:

          - the tokenizer's ``model_max_length`` (often the right
            value, but sometimes a sentinel like 1e18)
          - the model's ``config.max_position_embeddings``
            (the hard architectural limit)
          - the user-set ``config.reranker_max_length`` (if smaller
            than the architectural limit, honor it)

        Returns 512 as a defensive default if introspection fails.
        """
        if self.reranker is None:
            return 512

        candidates = []
        try:
            tok_max = getattr(
                self.reranker.tokenizer, "model_max_length", None
            )
            # Tokenizers without an enforced limit set this to a
            # huge sentinel (~1e18). Filter values that are clearly
            # out of range for any real cross-encoder.
            if tok_max and 0 < tok_max < 8192:
                candidates.append(int(tok_max))
        except Exception:
            pass

        try:
            mdl = getattr(self.reranker, "model", None)
            mdl_cfg = getattr(mdl, "config", None) if mdl else None
            if mdl_cfg is not None:
                mpe = getattr(mdl_cfg, "max_position_embeddings", None)
                if mpe and 0 < mpe < 8192:
                    candidates.append(int(mpe))
        except Exception:
            pass

        try:
            cfg_max = int(config.reranker_max_length)
            if 0 < cfg_max < 8192:
                candidates.append(cfg_max)
        except Exception:
            pass

        if candidates:
            return min(candidates)
        return 512

    def _truncate_for_reranker(self, text: str, max_tokens: int) -> str:
        """Truncate ``text`` so that the reranker's tokenizer
        produces at most ``max_tokens`` tokens.

        Cross-encoders (the kind we use) have a fixed
        max-position-embedding (typically 512). When a (claim,
        chunk) pair tokenizes longer than that, ``predict()``
        raises a tensor-shape mismatch. Existing retrieval rerank
        avoids this because children are small; our re-attribution
        scores parents which can be 2500+ chars (~700 tokens) and
        overflow.

        Token-precise truncation via the reranker's own tokenizer —
        no character heuristics, so this works regardless of the
        underlying model. Falls back to a 4-char-per-token estimate
        only if the tokenizer call itself errors.
        """
        if not text or self.reranker is None or max_tokens <= 0:
            return text
        try:
            tokenizer = self.reranker.tokenizer
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) <= max_tokens:
                return text
            return tokenizer.decode(
                ids[:max_tokens], skip_special_tokens=True
            )
        except Exception as e:
            logger.warning(
                f"Reranker tokenizer truncation failed ({e}); "
                f"falling back to char estimate"
            )
            # Conservative char fallback (~4 chars per token for
            # English; we deliberately under-estimate to stay safe).
            return text[: max_tokens * 4]

    def _find_sentence_start(self, text: str, position: int) -> int:
        """Locate the start index of the sentence ending at or before
        ``position`` in ``text``. Used by re-attribution to extract
        the claim a citation marker is attached to. Falls back to
        the start of the text if no sentence boundary is found.

        Boundaries: ``. ``, ``! ``, ``? `` (with the trailing space
        to avoid abbreviations), or paragraph break ``\\n\\n``. The
        start of the matched delimiter is past, so the returned
        index points at the FIRST char of the sentence body.
        """
        best = 0
        for delim in (". ", "! ", "? ", "\n\n", ".\n", "!\n", "?\n"):
            idx = text.rfind(delim, 0, position)
            if idx != -1:
                candidate = idx + len(delim)
                if candidate > best:
                    best = candidate
        return best

    def _find_best_sentence(self, claim: str, chunk_text: str) -> str:
        """Return the sentence in ``chunk_text`` most relevant to
        ``claim`` according to the cross-encoder reranker.

        Falls back to the first non-trivial sentence if the reranker
        is unavailable, the chunk has only one sentence, or scoring
        raises. This is the langroid principle applied at sentence
        granularity — the LLM gives the chunk number, deterministic
        scoring picks the supporting sentence.
        """
        if not chunk_text:
            return ""
        sentences = [
            s.strip()
            for s in re.split(r"(?<=[.!?])\s+", chunk_text)
            if s.strip()
        ]
        # Drop very short fragments (likely artifacts of bullet
        # points, abbreviations like "et al.", etc.) so we don't
        # rank noise above real sentences.
        sentences = [s for s in sentences if len(s) >= 20]
        if not sentences:
            # Fallback to the chunk start if sentence-splitting yielded
            # nothing usable.
            return chunk_text[:300].strip()
        if len(sentences) == 1 or self.reranker is None:
            return sentences[0]
        try:
            import numpy as np
            # Use the MODEL's actual max token count, not
            # ``config.reranker_max_length`` — the latter is the
            # construction-time setting and can exceed what the
            # transformer's ``max_position_embeddings`` allows. ``-8``
            # budgets for special tokens ([CLS], [SEP], etc.).
            max_total = max(64, self._get_reranker_max_tokens() - 8)
            half = max_total // 2
            t_claim = self._truncate_for_reranker(claim, half)
            t_sentences = [
                self._truncate_for_reranker(s, half) for s in sentences
            ]
            pairs = [[t_claim, s] for s in t_sentences]
            scores = self.reranker.predict(pairs)
            best_idx = int(np.argmax(scores))
            return sentences[best_idx]
        except Exception as e:
            logger.warning(f"Sentence reranker scoring failed: {e}")
            return sentences[0]

    async def _select_used_chunks(
        self,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]],
        timeout_seconds: float = 25.0,
    ) -> List[str]:
        """Structured-output pass: ask the LLM which chunk_ids it
        used to write the answer.

        Replaces the inline ``[cN]`` marker scheme that depended on
        the LLM voluntarily following a citation rule (which broke
        on list-style answers, smaller models, and multi-turn
        drift). Schema-mode JSON forces the model to commit to a
        list of chunk_ids — the validator rejects empty/malformed
        responses, and we additionally constrain the result to ids
        that were actually retrieved.

        Returns a list of validated chunk_ids in priority order
        (LLM's stated order). Empty list on any failure — caller
        should fall back to "use top-N reranked chunks for the
        whole answer."
        """
        if not answer.strip() or not sources:
            return []

        retrieved_ids = [
            s["chunk_id"] for s in sources
            if s.get("chunk_id") and s.get("chunk_text", "").strip()
        ]
        if not retrieved_ids:
            return []

        # Compact chunk catalog — id + first ~400 chars per chunk so
        # the LLM can identify each one without re-streaming the full
        # body it already saw during the answering pass.
        chunks_block = "\n\n".join(
            f"[{s['chunk_id']}] {(s.get('chunk_text') or '')[:400]}"
            for s in sources
            if s.get("chunk_id")
        )

        prompt = (
            "You just wrote the ANSWER below using the source CHUNKS. "
            "List the chunk_ids you actually drew on to write that "
            "answer.\n\n"
            "Output ONLY a JSON object of the form:\n"
            "  {\"chunk_ids\": [\"c1\", \"c3\", ...]}\n\n"
            "Rules:\n"
            f"- Use ONLY ids from this list: {retrieved_ids}.\n"
            "- Order ids by how central they are to the answer (most "
            "central first).\n"
            "- Include every chunk that contributed a fact, number, or "
            "definition you used. Skip chunks you did not use.\n"
            "- The list MUST contain at least one chunk_id.\n"
            "- Do NOT include any prose, explanation, or extra fields.\n\n"
            f"QUESTION:\n{question}\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"CHUNKS:\n{chunks_block}\n\n"
            "JSON:"
        )

        try:
            response = await self._invoke_llm(
                prompt=prompt,
                max_retries=1,
                timeout_seconds=timeout_seconds,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"chunk-id selection LLM call failed: {e}")
            return []

        raw = (response.content or "").strip()
        if not raw:
            return []
        # Permissive JSON parsing — small models in JSON mode commonly
        # emit malformations that strict ``json.loads`` rejects:
        # trailing commas, code fences, single quotes, unclosed
        # braces, smart quotes, prose preamble before the object.
        # ``json_repair`` recovers the intended structure from these
        # without an extra LLM call. The downstream isinstance guards
        # below still gate on shape, so a "repair" that lands on the
        # wrong shape (e.g., a string) is rejected exactly like a
        # parse failure was — strict regression-safety: we never
        # accept anything we wouldn't have accepted before.
        try:
            from json_repair import repair_json
            parsed = repair_json(raw, return_objects=True)
        except Exception as e:
            logger.warning(f"chunk-id selection JSON parse failed: {e}")
            return []

        ids_raw = parsed.get("chunk_ids") if isinstance(parsed, dict) else None
        if not isinstance(ids_raw, list):
            return []

        valid_set = set(retrieved_ids)
        out: List[str] = []
        seen: set = set()
        for cid in ids_raw:
            if not isinstance(cid, str):
                continue
            cid = cid.strip()
            if cid in valid_set and cid not in seen:
                out.append(cid)
                seen.add(cid)
        return out

    def _reattribute_and_extract(
        self,
        answer: str,
        sources: List[Dict[str, Any]],
        used_chunk_ids: Optional[List[str]] = None,
    ) -> tuple:
        """Inject ``[cN]`` markers into the answer based on the set
        of chunk_ids the LLM (via JSON-mode follow-up) said it
        used, then build the citations list.

        Algorithm:
          1. Split the answer into sentences.
          2. For each chunk_id in ``used_chunk_ids``, run the
             cross-encoder reranker over (chunk_text, sentence)
             pairs and pick the sentence with the highest score —
             that's the sentence this chunk best supports.
          3. Insert ``[cN]`` immediately after the chosen sentence.
             If two chunks land on the same sentence, both markers
             stack at that boundary.
          4. Build the citations list with the best sentence FROM
             the chunk (used by the highlight panel as ``quote``).

        Falls back gracefully:
          * No reranker → attach all used_chunk_ids as a final
            citation block at the end of the answer.
          * No used_chunk_ids → uses top-2 reranker-scored chunks
            against the whole answer (so we never return uncited).
          * Empty answer or sources → return as-is.

        Returns ``(answer_with_markers, citations)`` — citations is
        a list of ``{chunk_id, quote}`` dicts.
        """
        if not answer or not sources:
            return answer, []

        # Belt-and-braces: the system prompt tells the LLM not to
        # emit ``[cN]``/``[N]`` markers itself, but small models
        # occasionally still do. Strip them BEFORE we inject — the
        # reranker chooses placement, so any pre-existing markers
        # would just produce double-citations on a single sentence.
        marker_pattern = re.compile(r"\[\s*[Cc]?\s*\d+\s*\]")
        answer = marker_pattern.sub("", answer)
        # Collapse any double-spaces left behind so the prose reads
        # cleanly when the corrected frame replaces the streamed text.
        answer = re.sub(r"  +", " ", answer)

        # Build candidate pool of (id, text).
        chunk_text_by_id = {
            s["chunk_id"]: s.get("chunk_text", "")
            for s in sources
            if s.get("chunk_id") and s.get("chunk_text", "").strip()
        }
        if not chunk_text_by_id:
            return answer, []

        # Split answer into sentences with their byte ranges. Markdown
        # bullets and headings count as their own "sentence" so list
        # answers attribute one citation per item rather than the
        # whole list collapsing onto one chunk.
        sentences = self._split_into_sentences(answer)
        if not sentences:
            return answer, []

        # Resolve the working set of chunk_ids. If the structured
        # call returned nothing, fall back to "top-2 chunks against
        # the whole answer" using the reranker so we never serve an
        # uncited answer.
        working_ids: List[str] = []
        if used_chunk_ids:
            for cid in used_chunk_ids:
                if cid in chunk_text_by_id:
                    working_ids.append(cid)

        if not working_ids:
            working_ids = self._fallback_top_chunks_for_answer(
                answer, chunk_text_by_id, top_n=2
            )

        if not working_ids:
            return answer, []

        # Reranker-driven sentence-to-chunk attribution. If the
        # reranker can't load, we degrade to "all citations land on
        # the final sentence" — still cited, just less precise.
        rk = self.reranker
        try:
            import numpy as np
        except ImportError:
            np = None

        if rk is None or np is None:
            tail_idx = len(sentences) - 1
            citations = [
                {
                    "chunk_id": cid,
                    "quote": (chunk_text_by_id[cid] or "")[:300].strip(),
                }
                for cid in working_ids
            ]
            answer_with_markers = self._inject_markers_into_sentences(
                answer, sentences, {tail_idx: working_ids}
            )
            return answer_with_markers, citations

        # Token budgets — same trick as before. Pre-truncate the
        # chunk text ONCE per query and reuse across every chunk_id
        # we resolve.
        max_total = max(64, self._get_reranker_max_tokens() - 8)
        sentence_budget = min(96, max_total // 4)
        chunk_budget = max(64, max_total - sentence_budget)
        sentence_truncated = [
            self._truncate_for_reranker(s["text"], sentence_budget)
            for s in sentences
        ]

        # sentence_idx -> [chunk_id, ...]
        attach_map: Dict[int, List[str]] = {}
        citations: List[Dict[str, Any]] = []

        for cid in working_ids:
            chunk_text = chunk_text_by_id.get(cid, "")
            if not chunk_text.strip():
                continue
            t_chunk = self._truncate_for_reranker(chunk_text, chunk_budget)
            try:
                pairs = [[t_chunk, s] for s in sentence_truncated]
                scores = rk.predict(pairs)
            except Exception as e:
                logger.warning(
                    f"Reranker sentence-attribution failed for {cid}: {e}"
                )
                continue
            best_idx = int(np.argmax(scores))
            attach_map.setdefault(best_idx, []).append(cid)
            citations.append({
                "chunk_id": cid,
                "quote": self._find_best_sentence(
                    sentences[best_idx]["text"], chunk_text
                ),
            })

        if not attach_map:
            return answer, []

        answer_with_markers = self._inject_markers_into_sentences(
            answer, sentences, attach_map
        )
        return answer_with_markers, citations

    @staticmethod
    def _split_into_sentences(text: str) -> List[Dict[str, Any]]:
        """Split ``text`` into sentence-like spans for citation
        attribution. Returns dicts with ``text``, ``start``, ``end``
        — char offsets into the original string, end-exclusive.

        Markdown awareness:
          * Each non-blank line is treated as its own unit.
          * Inside paragraph lines, sentence boundaries split on
            ``. ! ?`` followed by whitespace.
          * Bullet points and headings stay as one unit (don't try
            to sub-split a single list item).
          * Trailing whitespace and any markers we left in are
            tolerated — markers can land just after the visible
            text.
        """
        if not text:
            return []
        spans: List[Dict[str, Any]] = []
        # Walk the text line by line, tracking absolute offsets.
        i = 0
        n = len(text)
        while i < n:
            j = text.find("\n", i)
            if j == -1:
                j = n
            line = text[i:j]
            stripped = line.strip()
            if not stripped:
                i = j + 1
                continue

            # Bullets/headings — keep as one span to preserve list
            # boundaries (citation lands right after the item).
            if re.match(r"^\s*(?:[-*+]\s|\d+[.)]\s|#{1,6}\s)", line):
                spans.append({"text": stripped, "start": i, "end": j})
                i = j + 1
                continue

            # Paragraph line — sub-split on sentence delimiters.
            local = 0
            for m in re.finditer(r"[.!?](?:\s+|$)", line):
                end_local = m.end()
                seg = line[local:end_local].strip()
                if seg:
                    seg_start = i + local
                    seg_end = i + end_local
                    spans.append({
                        "text": seg, "start": seg_start, "end": seg_end,
                    })
                local = end_local
            if local < len(line):
                tail = line[local:].strip()
                if tail:
                    spans.append({
                        "text": tail,
                        "start": i + local,
                        "end": j,
                    })
            i = j + 1
        return spans

    def _fallback_top_chunks_for_answer(
        self,
        answer: str,
        chunk_text_by_id: Dict[str, str],
        top_n: int = 2,
    ) -> List[str]:
        """Used when the structured-output call returned nothing.
        Score every chunk against the full answer with the reranker
        and pick the top-N. Guarantees an answer is never uncited.

        Falls back further to "first N chunks in retrieval order"
        when the reranker is unavailable.
        """
        ids = list(chunk_text_by_id.keys())
        if not ids:
            return []
        if self.reranker is None:
            return ids[:top_n]
        try:
            import numpy as np
        except ImportError:
            return ids[:top_n]
        max_total = max(64, self._get_reranker_max_tokens() - 8)
        a_budget = min(160, max_total // 3)
        c_budget = max(64, max_total - a_budget)
        a_t = self._truncate_for_reranker(answer, a_budget)
        try:
            pairs = [
                [a_t, self._truncate_for_reranker(chunk_text_by_id[cid], c_budget)]
                for cid in ids
            ]
            scores = self.reranker.predict(pairs)
        except Exception as e:
            logger.warning(f"Fallback chunk-rerank failed: {e}")
            return ids[:top_n]
        order = np.argsort(scores)[::-1]
        return [ids[int(i)] for i in order[:top_n]]

    @staticmethod
    def _inject_markers_into_sentences(
        answer: str,
        sentences: List[Dict[str, Any]],
        attach_map: Dict[int, List[str]],
    ) -> str:
        """Splice ``[cN]`` markers into ``answer`` at each sentence's
        end offset. Markers for the same sentence are concatenated
        in stable order. Edits are applied in REVERSE so earlier
        offsets remain valid as we splice.
        """
        if not attach_map:
            return answer
        edits: List[tuple] = []
        for idx, ids in attach_map.items():
            if idx < 0 or idx >= len(sentences):
                continue
            insert_pos = sentences[idx]["end"]
            marker_str = "".join(f"[{cid}]" for cid in ids)
            # Strip any trailing whitespace/punctuation already
            # present at insert_pos so the marker sits flush.
            edits.append((insert_pos, marker_str))
        if not edits:
            return answer
        out = answer
        for pos, marker in sorted(edits, key=lambda x: -x[0]):
            out = out[:pos] + marker + out[pos:]
        return out

    @staticmethod
    def _strip_to_body(text: str, title: str = "") -> str:
        """Strip context-header lines added at chunking time so what
        remains is the verbatim body that appears in the saved paper
        markdown.

        Handles every prefix shape the chunking code currently
        produces:
          - PyMuPDF:  ``"<doc_title> > <section>\\n\\n<body>"``
          - PyMuPDF:  ``"<section>\\n\\n<body>"`` (no doc_title)
          - PyMuPDF:  ``"## <section>\\n\\n<body>"``
          - Docling:  ``"<doc_title>\\n\\n<contextualized chunk>\\n\\n<body>"``
          - Docling:  ``"<doc_title>\\n\\n## <section>\\n\\n<body>"``

        Walks lines from the top peeling off anything that looks
        like a header (markdown heading, doc title, breadcrumb,
        short non-prose line). Stops at the first line that looks
        like body prose. The returned text is what
        ``MarkdownPreviewPanel.findFlexibleSpan`` will look for in
        the paper markdown — keeping it as a verbatim substring
        means the exact-match strategy succeeds and the highlight
        lands precisely.
        """
        if not text:
            return ""
        lines = text.splitlines()
        body_start = 0
        title_norm = title.strip() if title else ""

        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                # Blank line — keep walking; final lstrip cleans up.
                body_start = i + 1
                continue
            # Markdown heading line (#, ##, …)
            if re.match(r"^#{1,6}\s+\S", stripped):
                body_start = i + 1
                continue
            # Exact doc title alone
            if title_norm and stripped == title_norm:
                body_start = i + 1
                continue
            # Doc title plus breadcrumb continuation
            if title_norm and stripped.startswith(title_norm + " >"):
                body_start = i + 1
                continue
            # Generic breadcrumb-style header: contains " > ", short,
            # no terminal sentence punctuation. Body prose wouldn't
            # match this shape.
            if (
                " > " in stripped
                and len(stripped) < 200
                and not re.search(r"[.!?:]\s*$", stripped)
            ):
                body_start = i + 1
                continue
            # Otherwise — looks like body. Stop peeling.
            break

        body = "\n".join(lines[body_start:]).lstrip("\n").strip()
        # Defensive: if our walker stripped EVERYTHING (over-eager
        # heuristic on a short chunk), fall back to the original
        # text minus leading whitespace. Better fuzz than nothing.
        return body if body else text.lstrip()

    @staticmethod
    def _diversify_chunks(
        chunks: List[Dict[str, Any]],
        max_keep: int,
        similarity_threshold: float = 0.60,
    ) -> List[Dict[str, Any]]:
        """MMR-lite greedy diversity filter.

        Walks the rerank-ordered candidate list and keeps a chunk
        only if its lexical bigram Jaccard with EVERY already-kept
        chunk is below ``similarity_threshold``. Bigrams (rather
        than unigrams) catch paragraph-level near-duplicates that
        share most function words but differ in phrasing.

        ``similarity_threshold=0.60`` is empirically the right knob
        for scientific paper chunks: same-section parents typically
        score 0.70+, distinct-section chunks score <0.40. Adjust if
        you have unusually short or unusually similar chunks.

        Pure stdlib — no new dependency.
        """
        if not chunks:
            return []

        def bigrams(text: str) -> set:
            words = re.findall(r"\w+", text.lower())
            if len(words) < 2:
                return set(words)
            return set(zip(words, words[1:]))

        selected: List[Dict[str, Any]] = []
        selected_grams: List[set] = []

        for cand in chunks:
            if len(selected) >= max_keep:
                break
            cand_text = (cand.get("doc").page_content if cand.get("doc") else "") or ""
            cand_grams = bigrams(cand_text)
            if not cand_grams:
                # Tiny / empty chunk — keep it (likely a table or
                # heading) since it can't dominate the LLM context.
                selected.append(cand)
                selected_grams.append(cand_grams)
                continue

            too_similar = False
            for sel_grams in selected_grams:
                if not sel_grams:
                    continue
                inter = len(cand_grams & sel_grams)
                denom = min(len(cand_grams), len(sel_grams))
                if denom and inter / denom >= similarity_threshold:
                    too_similar = True
                    break
            if not too_similar:
                selected.append(cand)
                selected_grams.append(cand_grams)

        return selected

    async def _prepare_query(
        self,
        question: str,
        filter_files: Optional[List[str]] = None,
        user_id: str = "default",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Run retrieval, rerank, parent-resolution and build the LLM
        prompt. Returns one of two shapes:

          * ``{"answer": str, "sources": []}`` when no context was
            found — caller can short-circuit and return this directly.
          * ``{"messages": list, "sources": list}`` when the LLM
            should be invoked. ``messages`` is ready for either
            ``OllamaLLM.invoke`` (full answer) or ``OllamaLLM.astream``
            (token streaming); ``sources`` is the citation list that
            should accompany the answer.

        Both ``query()`` (non-streaming) and ``query_stream()`` use
        this so retrieval logic stays in one place.
        """
        vectorstore = self._get_user_collection(user_id)

        # 1. Retrieve many child chunks from vector store (up to 200)
        search_results = self._hybrid_search(
            question, vectorstore, filter_files, k=config.retrieve_k,
            user_id=user_id,
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

        # 3. Parent-Child Resolution: resolve filtered children to
        # unique parents. We oversample up to 2x ``max_parents`` first
        # so the diversity filter in step 3.5 has a richer candidate
        # pool to pick from. Without oversampling, a homogeneous top
        # of the rerank list would force all our chunks to come from
        # the same paragraph, which is exactly what produces the
        # "many citations on one line, all the same content" UX issue.
        candidate_pool_size = max(config.max_parents * 2, 4)
        parent_ids_seen: set[str] = set()
        candidate_parents: List[Dict[str, Any]] = []

        for result in reranked_children:
            d = result["doc"]
            ctype = d.metadata.get("content_type", "text")
            if ctype != "text":
                # Tables pass through directly
                candidate_parents.append(result)
                continue

            parent_id = d.metadata.get("parent_id")
            if not parent_id or parent_id in parent_ids_seen:
                continue

            parent_ids_seen.add(parent_id)
            parent_data = self._get_parent_data(parent_id, user_id)
            ptext = parent_data.get("text", "")
            if ptext:
                from langchain_core.documents import Document
                # Create a synthetic result with parent text. Carry
                # the parent's body_start/body_end/page offsets onto
                # the result dict so they survive into the source
                # output without us re-reading the parent store.
                candidate_parents.append({
                    "doc": Document(
                        page_content=ptext,
                        metadata=d.metadata,
                    ),
                    "rerank_score": result.get("rerank_score", 0),
                    "normalized_score": result.get("normalized_score", 0),
                    "body_start": parent_data.get("body_start"),
                    "body_end": parent_data.get("body_end"),
                    "page": parent_data.get("page"),
                })
            else:
                candidate_parents.append(result)

            if len(candidate_parents) >= candidate_pool_size:
                break

        # 3.5. Diversity filter — MMR-style greedy selection that
        # drops candidates whose lexical bigram overlap with already-
        # selected chunks exceeds the configured threshold. Same-
        # paragraph parents from one paper section often share 60-80%
        # of their bigrams; without this filter the LLM gets shown
        # the same content under different chunk_ids and ends up
        # citing all of them.
        parent_results = self._diversify_chunks(
            candidate_parents, max_keep=config.max_parents
        )

        # 4. Build LLM context from resolved parents.
        #
        # Citation markers are assigned **positionally per turn**
        # (``c1``, ``c2``, ``c3``, …) — the LlamaIndex
        # CitationQueryEngine pattern. Critical property: numbering
        # is scoped to THIS query, not persistent across the
        # conversation. Earlier we used a deterministic md5-hash id
        # (same chunk → same id across every turn), which caused a
        # bug where the LLM would emit a chunk_id from a prior turn
        # and the frontend would happily resolve it to the same
        # physical chunk because the hash matched. Positional ids
        # cannot carry across turns by construction: ``c1`` only
        # means anything inside the prompt that defined it.
        # The ``c`` prefix avoids collisions with literal reference
        # numbers like ``[1]`` that appear naturally in scientific
        # papers.
        context_parts = []
        sources = []
        for chunk_index, result in enumerate(parent_results):
            d = result["doc"]
            score = result.get("normalized_score", 0)
            ctype = d.metadata.get("content_type", "text")
            src = d.metadata.get("source", "")
            sec = d.metadata.get("section_title", "")

            title = d.metadata.get("doc_title", "")
            authors = d.metadata.get("doc_authors", "")

            chunk_id = f"c{chunk_index + 1}"

            # Build a rich header for the LLM context
            header_elements = [f"chunk_id={chunk_id}"]
            if title:
                header_elements.append(f"Title: {title}")
            if authors:
                header_elements.append(f"Authors: {authors}")
            header_elements.append(f"File: {src}")
            if sec:
                header_elements.append(f"Section: {sec}")

            header_str = " | ".join(header_elements)

            # Each chunk's body is preceded by a literal ``[chunk_id]``
            # marker on its own line so the LLM can reference it back
            # using the same syntax. The header on the next line is
            # informational only.
            if ctype == "table":
                context_parts.append(
                    f"[{chunk_id}] [TABLE | {header_str}]:\n{d.page_content}"
                )
            else:
                context_parts.append(
                    f"[{chunk_id}] [{header_str}]:\n{d.page_content}"
                )

            parser_type = d.metadata.get("parser_type", "docling")
            # Strip every flavor of ingest-time context header
            # (markdown headings, doc title, breadcrumbs like
            # "Title > Section") so what we expose as chunk_text is
            # the verbatim body that lives in the saved paper
            # markdown. The frontend's exact-substring search then
            # finds it precisely; without this strip, partial
            # prefixes prevent exact matching and the highlight
            # falls back to fuzzy approximation.
            citable_text = self._strip_to_body(d.page_content, title=title)
            # Pull the offset metadata recorded at ingest time onto the
            # source dict. When body_start/body_end are present, the
            # frontend slices the saved markdown directly — byte-exact
            # highlight, no fuzzy matching needed. When absent (legacy
            # chunks indexed before this feature, or chunks where the
            # ingest-time substring search failed), the frontend falls
            # back to its existing fuzzy strategies.
            source_record: Dict[str, Any] = {
                "chunk_id": chunk_id,
                "source": src,
                "section": sec,
                "parser_type": parser_type,
                "score": score,
                "chunk_text": citable_text,
            }
            body_start = result.get("body_start")
            body_end = result.get("body_end")
            page = result.get("page") or d.metadata.get("page") or None
            if body_start is not None and body_end is not None:
                source_record["body_start"] = body_start
                source_record["body_end"] = body_end
            if page:
                source_record["page"] = page
            sources.append(source_record)

        context = "\n\n".join(context_parts)

        if not context_parts:
            return {
                "answer": "I couldn't find enough relevant context in the selected sources to answer that question.",
                "sources": [],
            }

        # Build multi-turn messages for conversation memory.
        #
        # NOTE: the LLM is no longer asked to emit ``[cN]`` markers
        # inline — that approach was fragile (small models, list-
        # style answers, and multi-turn drift all caused the model
        # to silently drop markers, leaving answers uncited). We now
        # let the LLM answer freely in markdown prose; a separate
        # post-stream JSON-mode call selects which chunk_ids were
        # used, and the reranker then inserts ``[cN]`` markers at
        # the sentence boundaries each chunk best supports. This is
        # the structured-output pattern (LangChain ``with_structured_
        # output``, LlamaIndex ``CitationQueryEngine``) — schema-
        # enforced, so an answer can never be uncited.
        system_msg = {
            "role": "system",
            "content": (
                "You are a scientific research assistant. Answer the "
                "question using ONLY the provided context from research "
                "papers above. Use clear markdown formatting (headings, "
                "lists, bold for key terms).\n\n"
                "Rules:\n"
                "1. Ground every factual claim in the supplied context. "
                "If the context does not contain enough information, say "
                "so explicitly rather than guessing.\n"
                "2. Be precise: prefer concrete numbers, dataset names, "
                "and quoted terminology from the context over vague "
                "summaries.\n"
                "3. Do NOT add bracketed reference markers like [1], "
                "[c1], (Smith 2020), or footnote-style citations of any "
                "kind. The application attaches citations automatically "
                "after you finish — your job is only to write the "
                "answer."
            ),
        }

        messages = [system_msg]

        # Add conversation history (last 5 turns = 10 messages max).
        #
        # Rewrite citation markers in prior assistant turns. We
        # accept all the variants LLMs naturally emit: ``[cN]``
        # (canonical, what the prompt asks for), bare ``[N]``
        # (most common — bare numeric brackets dominate training
        # data), and any of those with leading/trailing whitespace
        # inside the brackets like ``[ c1]`` or ``[ 1 ]`` (some
        # models pad for visual separation). Uppercase ``[C1]`` is
        # also accepted defensively.
        #
        # Per-turn positional ids mean ``c1`` in turn 1 may point at
        # a totally different chunk than ``c1`` in turn 2, so we
        # MUST NOT leave the literal ``[cN]`` form in history — that
        # would mis-attribute. But fully *erasing* markers caused a
        # different bug: the LLM, looking at its own marker-free
        # prior turn, drifted into a "this assistant doesn't cite"
        # style and stopped emitting markers on follow-up questions
        # (few-shot mimicry over the system prompt). Logs showed
        # ``total_markers=1`` on Q1 and ``total_markers=0`` on Q2/Q3
        # within the same conversation.
        #
        # Compromise: replace each marker with ``[†]`` (the academic
        # footnote dagger). It preserves the inline-citation pattern
        # the LLM should imitate without leaking any turn-specific
        # chunk id. Rule 5 of the system prompt ("Use ONLY [cN]
        # labels that appear verbatim in the context above") still
        # forces fresh, valid citations on the current turn.
        marker_pattern = re.compile(r"\[\s*[Cc]?\s*\d+\s*\]")
        if chat_history:
            history_window = chat_history[-10:]  # Last 5 Q&A pairs
            for msg in history_window:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "assistant" and content:
                    content = marker_pattern.sub("[†]", content)
                messages.append({"role": role, "content": content})

        # Add current question with context
        user_msg = f"""Context from research papers:
{context}

Question: {question}"""
        messages.append({"role": "user", "content": user_msg})

        return {"messages": messages, "sources": sources}

    async def query(
        self,
        question: str,
        filter_files: Optional[List[str]] = None,
        user_id: str = "default",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Non-streaming RAG query — returns the full answer in one
        shot. Frontend uses this as a fallback when the streaming
        endpoint is unavailable."""
        prepared = await self._prepare_query(
            question, filter_files, user_id, chat_history
        )
        if "answer" in prepared:
            return prepared

        response = await self._invoke_llm(
            messages=prepared["messages"], timeout_seconds=RAG_QUERY_TIMEOUT_SECONDS
        )
        # Non-streaming path is a fallback only; the structured-
        # output citation pipeline lives in ``query_stream`` (which
        # is what the frontend uses). Returning the plain answer +
        # sources here keeps this path lightweight and matches the
        # ``QueryResponse`` schema (no ``citations`` field).
        return {
            "answer": (response.content or "").strip(),
            "sources": prepared["sources"],
        }

    async def query_stream(
        self,
        question: str,
        filter_files: Optional[List[str]] = None,
        user_id: str = "default",
        chat_history: Optional[List[Dict[str, str]]] = None,
    ):
        """Streaming RAG query — yields NDJSON-shaped frames suitable
        for the ``/api/chat/query/stream`` endpoint to forward to the
        client. Frame types match the frontend's ``StreamFrame`` union:

          * ``{"type": "text_delta", "text": "..."}`` — additive token
          * ``{"type": "sources",   "sources": [...]}`` — citation list
          * ``{"type": "error",     "error": "..."}``  — fatal mid-stream
          * ``{"type": "done"}``                       — clean end
        """
        try:
            prepared = await self._prepare_query(
                question, filter_files, user_id, chat_history
            )
        except Exception as e:
            yield {"type": "error", "error": str(e)}
            return

        # No-context short-circuit — emit the canned answer as a single
        # text_delta so the UI behaves identically to the streaming
        # path with empty sources.
        if "answer" in prepared:
            yield {"type": "text_delta", "text": prepared["answer"]}
            yield {"type": "sources", "sources": prepared["sources"]}
            yield {"type": "done"}
            return

        # Yield the sources frame BEFORE streaming text so the frontend
        # can populate its valid chunk_id set before any [chunk_id]
        # markers arrive in the stream. Without this, markers render as
        # literal "[xxxxxxxx]" text mid-stream and only snap into
        # superscript badges after the stream completes.
        yield {"type": "sources", "sources": prepared["sources"]}

        # Stream LLM tokens. We accumulate the full answer text so we
        # can run a follow-up citation-extraction pass (Pass 2) once
        # streaming is done.
        accumulated = ""
        try:
            async for chunk in self.llm.astream(messages=prepared["messages"]):
                if chunk:
                    accumulated += chunk
                    yield {"type": "text_delta", "text": chunk}
        except RAGProviderAuthError as e:
            yield {"type": "error", "error": f"Auth failed: {e}"}
            return
        except Exception as e:
            logger.exception("query_stream LLM call failed")
            yield {"type": "error", "error": f"LLM stream failed: {e}"}
            return

        # Diagnostic: log what the LLM actually cited vs what was
        # retrieved. Lets us spot at-a-glance whether bad citations
        # are coming from (a) LLM lazily citing only c1 — narrow set
        # vs many retrieved, (b) retrieval returning few/homogeneous
        # chunks — small retrieved set, (c) prompt drift — markers
        # not following the [cN] format.
        if accumulated.strip():
            # Accept both ``[cN]`` and bare ``[N]``; normalize bare
            # numeric markers to canonical ``cN`` form. Bounds-check
            # against retrieved chunk_ids so reference numbers
            # quoted from source text don't masquerade as citations.
            retrieved_id_set = {
                s["chunk_id"] for s in prepared.get("sources", [])
            }
            # Same permissive pattern as the strip + extract sites
            # — accepts ``[c1]``, ``[1]``, ``[C1]``, ``[ c1]``,
            # ``[ 1 ]``, etc. Whitespace inside the brackets is
            # observed in practice from real LLM outputs.
            raw_markers = re.findall(
                r"\[\s*[Cc]?\s*(\d+)\s*\]", accumulated
            )
            found_markers = [
                f"c{num}" for num in raw_markers
                if f"c{num}" in retrieved_id_set
            ]
            unique_cited = sorted(set(found_markers))
            retrieved_ids = sorted(
                s["chunk_id"] for s in prepared.get("sources", [])
            )
            # Build per-chunk diagnostic mapping. We also OFFSET-VALIDATE
            # each chunk: reload the saved paper markdown once per
            # source, slice ``markdown[body_start:body_end]``, and
            # compare to the ``chunk_text`` field. If they diverge,
            # the offset is stale OR ``_strip_to_body`` is producing
            # a different body than what was stored at ingest.
            sources_for_log = prepared.get("sources", [])
            md_cache: Dict[str, Optional[str]] = {}

            def _load_md_once(filename: str) -> Optional[str]:
                if filename in md_cache:
                    return md_cache[filename]
                try:
                    md_path = get_user_markdown_file_path(user_id, filename)
                    if md_path.is_file():
                        md_cache[filename] = md_path.read_text(encoding="utf-8")
                    else:
                        md_cache[filename] = None
                except Exception:
                    md_cache[filename] = None
                return md_cache[filename]

            def _validate(s: Dict[str, Any]) -> str:
                bs = s.get("body_start")
                be = s.get("body_end")
                if bs is None or be is None:
                    return "no-offset"
                md = _load_md_once(s.get("source", ""))
                if md is None:
                    return f"off{bs}-md-missing"
                if not (0 <= bs < be <= len(md)):
                    return f"off{bs}-OOB(md_len={len(md)})"
                slice_text = md[bs:be]
                ctext = s.get("chunk_text", "")
                if slice_text == ctext:
                    return f"off{bs}-OK"
                # Compute first divergence position for actionable
                # diagnostics.
                limit = min(len(slice_text), len(ctext))
                first_diff = next(
                    (
                        i for i in range(limit)
                        if slice_text[i] != ctext[i]
                    ),
                    limit,
                )
                return (
                    f"off{bs}-MISMATCH("
                    f"slice_len={len(slice_text)},"
                    f"ctext_len={len(ctext)},"
                    f"first_diff={first_diff})"
                )

            id_to_source = {
                s["chunk_id"]: (
                    f"{s.get('source', '?')}"
                    + (f":p{s['page']}" if s.get("page") else "")
                    + f":{_validate(s)}"
                )
                for s in sources_for_log
            }
            # Snippet of the actual LLM output so we can diagnose
            # zero-marker cases. Tells us whether the LLM emitted
            # NO citations at all (prompt/model compliance issue) or
            # markers in an unexpected format the regex doesn't
            # catch (parentheses (1), angle brackets <1>, prefixed
            # like [Source 1], unicode superscripts ¹², etc.).
            # Whitespace collapsed for readability; truncated to
            # keep the log line manageable.
            snippet_raw = accumulated[:280]
            snippet = re.sub(r"\s+", " ", snippet_raw).strip()
            # Also surface ALL bracketed substrings (any content),
            # so non-numeric markers like [Source 1] stand out at a
            # glance even though they don't match the citation
            # regex. Limited to first 8 to avoid log spam.
            any_brackets = re.findall(r"\[[^\]\n]{1,40}\]", snippet_raw)[:8]
            logger.warning(
                "[CITATION DIAG] total_markers=%d unique_cited=%d/%d "
                "retrieved=%s cited=%s mapping=%s "
                "any_brackets=%s snippet=%r",
                len(found_markers),
                len(unique_cited),
                len(retrieved_ids),
                retrieved_ids,
                unique_cited,
                id_to_source,
                any_brackets,
                snippet,
            )

        # Structured-output citation pass. Two stages:
        #   1. Ask the LLM (JSON mode) which chunk_ids it used to
        #      write the streamed answer. Schema-constrained, so it
        #      can't return an empty list and can't hallucinate ids
        #      outside the retrieved set.
        #   2. Reranker assigns each of those chunks to the answer
        #      sentence it best supports, then we splice ``[cN]``
        #      markers in at those sentence boundaries.
        # If stage 1 fails, stage 2 falls back to "top-2 reranker-
        # scored chunks against the whole answer" so an answer is
        # never uncited.
        citations: List[Dict[str, Any]] = []
        rewrite_error: Optional[str] = None
        corrected_answer = accumulated
        used_chunk_ids: List[str] = []
        if accumulated.strip():
            try:
                used_chunk_ids = await self._select_used_chunks(
                    question=question,
                    answer=accumulated,
                    sources=prepared["sources"],
                )
            except Exception as e:
                logger.warning(f"chunk-id selection failed: {e}")
                used_chunk_ids = []
            try:
                corrected_answer, citations = self._reattribute_and_extract(
                    accumulated,
                    prepared["sources"],
                    used_chunk_ids=used_chunk_ids,
                )
            except Exception as e:
                logger.warning(f"Citation injection failed: {e}")
                citations = []
                corrected_answer = accumulated
                rewrite_error = str(e)[:200]

        # Diagnostic — fires unconditionally so we always see the
        # outcome of the structured-output citation pass. Logs the
        # ids the LLM said it used (stage 1) and the citations the
        # reranker actually attached to sentences (stage 2). When
        # ``llm_selected`` is empty but ``citations`` is not, the
        # fallback "top-N chunks against the whole answer" path
        # ran — that's expected behavior, not an error.
        if accumulated.strip():
            answer_rewritten = corrected_answer != accumulated
            citations_summary = [
                {
                    "chunk_id": c.get("chunk_id"),
                    "quote_preview": (c.get("quote") or "")[:140],
                }
                for c in citations
            ]
            logger.warning(
                "[CITATION DIAG] llm_selected=%s injected=%s rewritten=%s "
                "citations=%s%s",
                used_chunk_ids,
                len(citations),
                answer_rewritten,
                citations_summary,
                f" error={rewrite_error!r}" if rewrite_error else "",
            )

        # If re-attribution rewrote any markers, push the corrected
        # answer text to the frontend so subsequent renders use the
        # right [cN] → chunk mapping. The displayed superscript
        # numbers stay stable because the frontend numbers by order
        # of first appearance — only the underlying chunk_id link
        # changes, so users don't see the answer text "flicker".
        if corrected_answer != accumulated:
            yield {"type": "answer_corrected", "text": corrected_answer}

        yield {"type": "citations", "citations": citations}

        yield {"type": "done"}

    async def _extract_citations(
        self,
        answer: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Pass 2 of the citation pipeline — extract verbatim quotes
        per cited chunk_id from the streamed answer.

        Pipeline:
          1. Regex-find all ``[hex_id]`` markers actually present in
             the answer (the LLM may have skipped some chunks).
          2. Filter ``chunks`` down to just the cited ones.
          3. Ask the LLM (JSON mode) to map each cited chunk_id to
             a verbatim quote from that chunk's text.
          4. Validate the response with the ``Citations`` Pydantic
             schema, then drop any citation whose ``chunk_id`` isn't
             in the cited set OR whose ``quote`` doesn't fuzzy-match
             the chunk text (>=80% via difflib SequenceMatcher).
          5. Return validated citations as plain dicts.

        Returns ``[]`` on any failure — citations are nice-to-have,
        the streamed answer is the primary deliverable.
        """
        from backend.schemas.schemas import Citations as _CitationsSchema

        # 1. Find chunk_ids actually mentioned in the answer.
        # Tolerates every marker variant LLMs commonly emit:
        # ``[c1]``, ``[1]``, ``[C1]``, ``[ c1]``, ``[ 1 ]``, etc.
        # All normalize to ``cN`` for downstream lookup.
        # Bounds-checked below by intersecting with retrieved
        # chunk_ids so bare numeric brackets quoted from paper
        # body text don't false-positive.
        marker_pattern = re.compile(r"\[\s*[Cc]?\s*(\d+)\s*\]")
        cited_ids = {f"c{num}" for num in marker_pattern.findall(answer)}
        if not cited_ids:
            return []

        # 2. Filter chunks down to cited ones.
        cited_chunks = [c for c in chunks if c.get("chunk_id") in cited_ids]
        if not cited_chunks:
            return []

        # 3. Build the JSON-mode prompt. We hand the LLM only the
        # chunks it actually cited so the prompt stays small and
        # fits even Ollama's modest 4-8K default context windows.
        chunks_block = "\n\n".join(
            f"[{c['chunk_id']}]\n{c.get('chunk_text', '')[:2000]}"
            for c in cited_chunks
        )

        prompt = (
            "You are a citation extractor. Given an ANSWER and the source "
            "CHUNKS it cites, return a JSON object listing the verbatim "
            "quote from each chunk that supports the answer's claim about "
            "that chunk.\n\n"
            "Rules:\n"
            "- Output ONLY a JSON object matching: "
            "{\"citations\": [{\"chunk_id\": \"...\", \"quote\": \"...\"}]}\n"
            "- Each `quote` MUST be a verbatim substring (or near-verbatim "
            "phrase) from the matching chunk's text.\n"
            "- Maximum 300 characters per quote.\n"
            "- One citation entry per cited chunk_id.\n"
            "- Do NOT include chunk_ids that do not appear in CHUNKS.\n"
            "- Do NOT include any prose or explanation outside the JSON.\n\n"
            f"ANSWER:\n{answer}\n\n"
            f"CHUNKS:\n{chunks_block}\n\n"
            "JSON:"
        )

        try:
            response = await self._invoke_llm(
                prompt=prompt,
                max_retries=1,
                timeout_seconds=30.0,
                response_format={"type": "json_object"},
            )
        except Exception as e:
            logger.warning(f"Citation pass-2 LLM call failed: {e}")
            return []

        raw = (response.content or "").strip()
        if not raw:
            return []

        # 4. Pydantic-validate. Strip code fences a few small models
        # add even in JSON mode.
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.MULTILINE)
        try:
            parsed = _CitationsSchema.model_validate_json(raw)
        except Exception as e:
            logger.warning(f"Citation JSON failed Pydantic validation: {e}")
            return []

        # 5. Validate each citation: chunk_id must be cited, quote
        # must fuzzy-match the chunk text.
        chunk_text_by_id = {
            c["chunk_id"]: (c.get("chunk_text") or "") for c in cited_chunks
        }
        validated: List[Dict[str, Any]] = []
        seen_ids: set = set()
        for cit in parsed.citations:
            cid = cit.chunk_id.strip()
            quote = cit.quote.strip()
            if not cid or not quote:
                continue
            if cid not in cited_ids:
                continue
            if cid in seen_ids:
                continue
            chunk_text = chunk_text_by_id.get(cid, "")
            if not _quote_matches_chunk(quote, chunk_text):
                logger.debug(
                    f"Dropping hallucinated citation for {cid}: "
                    f"quote not in chunk text"
                )
                continue
            seen_ids.add(cid)
            # ``verified=True`` is redundant given the chunk_id +
            # fuzzy-match checks above already passed; we set it
            # explicitly so the frontend (or future analytics) can
            # filter on it without relying on list membership alone.
            validated.append(
                {"chunk_id": cid, "quote": quote, "verified": True}
            )

        # Marker completeness check (industry-standard verification step).
        # Track which [chunk_id] markers in the answer didn't get a
        # validated quote back — these are markers the LLM emitted but
        # Pass 2 couldn't substantiate. They surface as a log warning
        # so operators can spot prompt drift or weak Pass-2 models.
        unverified_ids = cited_ids - seen_ids
        total_markers = len(cited_ids)
        if unverified_ids:
            logger.warning(
                "[CITATION DIAG] completeness=%s/%s markers verified; "
                "unverified=%s",
                total_markers - len(unverified_ids),
                total_markers,
                sorted(unverified_ids),
            )
        else:
            logger.warning(
                "[CITATION DIAG] completeness=%s/%s markers verified",
                total_markers,
                total_markers,
            )

        return validated

    def _save_paper_markdown(
        self,
        user_id: str,
        source: str,
        markdown: str,
    ) -> None:
        """Persist the extracted markdown view of a paper alongside
        its PDF so the citation preview panel can serve it later.
        Best-effort: a write failure is logged but never breaks
        ingest. Content is utf-8."""
        if not markdown or not source:
            return
        try:
            md_path = get_user_markdown_file_path(user_id, source)
            md_path.parent.mkdir(parents=True, exist_ok=True)
            md_path.write_text(markdown, encoding="utf-8")
        except Exception as e:
            logger.warning(
                f"Failed to save extracted markdown for {source} "
                f"(user {user_id}): {e}"
            )

    async def suggest_followups(
        self,
        chat_history: List[Dict[str, str]],
        max_suggestions: int = 3,
    ) -> List[str]:
        """Return short follow-up question prompts based on the recent
        conversation. Used by the assistant-ui SuggestionAdapter to
        populate "you might also ask…" clickable chips. Returns an
        empty list on any failure — the UI treats suggestions as
        nice-to-have."""
        if not chat_history:
            return []

        # Take the last 3 turns (6 messages max) to keep latency low.
        window = chat_history[-6:]
        transcript_lines = []
        for m in window:
            role = m.get("role", "user")
            content = (m.get("content") or "").strip()
            if not content:
                continue
            label = "User" if role == "user" else "Assistant"
            transcript_lines.append(f"{label}: {content[:600]}")
        transcript = "\n".join(transcript_lines)
        if not transcript:
            return []

        prompt = (
            "Given the conversation below between a researcher and an "
            "assistant about scientific papers, propose "
            f"{max_suggestions} short follow-up questions the user "
            "might ask next. Each question must:\n"
            "- be self-contained and clearly worded\n"
            "- be 12 words or fewer\n"
            "- not repeat anything already asked\n\n"
            f"Conversation:\n{transcript}\n\n"
            "Return ONLY the questions, one per line, with no "
            "numbering, bullets, or extra prose."
        )

        try:
            response = await self._invoke_llm(
                prompt=prompt,
                max_retries=1,
                timeout_seconds=15.0,
            )
        except Exception as e:
            logger.warning(f"suggest_followups LLM call failed: {e}")
            return []

        raw = (response.content or "").strip()
        if not raw:
            return []

        # Split on newlines, strip leading list markers / numbering.
        lines = []
        for line in raw.splitlines():
            cleaned = line.strip()
            if not cleaned:
                continue
            cleaned = re.sub(r"^[\-\*•]\s*", "", cleaned)
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", cleaned)
            if not cleaned:
                continue
            # Drop any line that is suspiciously not a question
            # (model occasionally adds preamble like "Here are…").
            if len(cleaned) < 6:
                continue
            lines.append(cleaned)
            if len(lines) >= max_suggestions:
                break
        return lines

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

            # Existence probe. ONLY treat 404/"not found" as "no
            # uploads yet" → []. Any other exception (auth, transport,
            # local-mode quirks) falls through to the outer try/except
            # which logs visibly — masking those was hiding the PDF
            # the user just uploaded.
            try:
                client.get_collection(collection_name)
            except Exception as exc:
                msg = str(exc).lower()
                if "not found" in msg or "404" in msg or "doesn't exist" in msg or "does not exist" in msg:
                    return []
                # Real error — let it surface, but try to scroll anyway
                # since the collection might actually exist; some local
                # Qdrant versions raise on get_collection but scroll fine.
                logger.warning(
                    f"get_collection probe raised non-404 for {collection_name}: {exc!r}; attempting scroll anyway"
                )

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

        # Persist the extracted markdown so the citation preview
        # panel can serve it via /api/chat/files/{name}/markdown.
        self._save_paper_markdown(user_id, source, extracted_text)

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
                    page_from_origin = getattr(chunk.meta.origin, "page_no", 0)

                    # Resolve body offset in the exported markdown.
                    # HybridChunker can reassemble across structural
                    # boundaries, so the contextualized chunk text
                    # may not be a contiguous substring — we fall back
                    # to searching for the chunk body without the
                    # contextualization breadcrumbs before giving up.
                    # ``matched_text`` tracks WHICH string was actually
                    # found so ``body_end`` reflects the real matched
                    # span (the previous version used ``len(p_text)``
                    # even when the raw_text fallback fired, which
                    # overshot the highlight when raw_text was
                    # shorter than the contextualized chunk).
                    body_start: Optional[int] = None
                    body_end: Optional[int] = None
                    if extracted_text and p_text.strip():
                        matched_text: Optional[str] = None
                        pos = extracted_text.find(p_text)
                        if pos != -1:
                            matched_text = p_text
                        else:
                            raw_text = getattr(chunk, "text", "") or ""
                            if raw_text.strip():
                                raw_pos = extracted_text.find(raw_text)
                                if raw_pos != -1:
                                    pos = raw_pos
                                    matched_text = raw_text
                        if pos != -1 and matched_text:
                            # Same trimmed-body alignment as the
                            # PyMuPDF path. Without this, leading or
                            # trailing whitespace in ``matched_text``
                            # causes the recorded offset to disagree
                            # with the ``chunk_text`` field exposed
                            # to the frontend (which comes from
                            # ``_strip_to_body`` and is .strip()-ed).
                            leading_ws = len(matched_text) - len(matched_text.lstrip())
                            trailing_ws = len(matched_text) - len(matched_text.rstrip())
                            body_start = pos + leading_ws
                            body_end = pos + len(matched_text) - trailing_ws

                    parent_chunks.append({
                        "parent_id": pid,
                        "text": parent_with_header,
                        "section_title": section_title,
                        "body_start": body_start,
                        "body_end": body_end,
                        "page": page_from_origin or None,
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
                                "page": page_from_origin,
                                "section_title": section_title,
                                "headings": headings,
                                "parent_id": pid,
                                "child_index": c_idx,
                                "char_count": len(c_text),
                                "word_count": len(c_text.split()),
                            },
                        })
            
            # Parents are returned (not persisted here) so the
            # upload orchestrator can batch-write them once per
            # flush — see ``_process_pdf`` for the full rationale.
            logger.info(f"Docling skill created {len(parent_chunks)} parents, {len(all_child_chunks)} children for {source}")

            # Deduplicate children
            unique_children = self._deduplicate_chunks(all_child_chunks)
            total = len(unique_children)
            for c in unique_children:
                c["metadata"]["total_chunks"] = total
            from langchain_core.documents import Document
            return (
                [Document(page_content=c["text"], metadata=c["metadata"]) for c in unique_children],
                extracted_text,
                parent_chunks,
            )
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
        full_text: str = "",
    ):
        """Create parent-child hierarchical chunks with Markdown splitting and contextual headers.

        Flow:
        1. Convert each section to Markdown with headers
        2. Split into parent chunks (~2500 chars) using MarkdownTextSplitter
        3. Split into child chunks using SemanticChunker (meaning-based boundaries)
        4. Prepend contextual chunk headers (CCH) to each child before embedding:
           [Document Title] > [Section Title] > [chunk text]

        Tables are indexed directly without parent-child split.

        When ``full_text`` is provided (the saved paper markdown),
        each parent gets a ``body_start``/``body_end`` offset pinning
        its body to a precise char range in the source — used at
        render time to anchor citation highlights without fuzzy
        matching. Pages are also recovered from ``<!-- Page N -->``
        markers when present. ``full_text=""`` (default) preserves
        the prior behavior with no offset metadata.

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
                # Strip the splitter-injected section header to avoid doubles.
                # The remaining ``p_text`` is the parent's BODY — the same
                # bytes we expect to find verbatim in the saved paper
                # markdown (modulo header lines elided during section
                # detection).
                if header_prefix and p_text.startswith(header_prefix):
                    p_text = p_text[len(header_prefix):]
                parent_with_header = header + p_text
                pid = f"{parent_id}_p{p_idx}"

                # Resolve the parent's body offset in the original
                # markdown. ``find()`` is byte-exact so this either
                # returns a precise [start, end) span or -1 (we
                # record nothing in that case and the frontend falls
                # back to fuzzy matching for this chunk only).
                #
                # We align the recorded span to the *trimmed* body —
                # advance past leading whitespace and pull back from
                # trailing whitespace. Without this, ``p_text`` from
                # the splitter often carries a leading newline that
                # ``_strip_to_body`` (the function exposing
                # ``chunk_text`` to Pass 2 and the frontend) removes
                # via ``.strip()``. The recorded offsets would then
                # disagree with ``chunk_text`` by 1-2 whitespace
                # chars at the boundaries, breaking byte-exact
                # comparison and causing Pass 2 quote-verification
                # drift + frontend slice/chunk_text mismatch.
                body_start: Optional[int] = None
                body_end: Optional[int] = None
                page: Optional[int] = None
                if full_text and p_text.strip():
                    pos = full_text.find(p_text)
                    if pos != -1:
                        leading_ws = len(p_text) - len(p_text.lstrip())
                        trailing_ws = len(p_text) - len(p_text.rstrip())
                        body_start = pos + leading_ws
                        body_end = pos + len(p_text) - trailing_ws
                        page = self._find_page_for_offset(full_text, body_start)

                parent_chunks.append(
                    {
                        "parent_id": pid,
                        "text": parent_with_header,
                        "section_title": section_title,
                        "body_start": body_start,
                        "body_end": body_end,
                        "page": page,
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
                    child_meta: Dict[str, Any] = {
                        "section_title": section_title,
                        "content_type": "text",
                        "parent_id": pid,
                        "child_index": c_idx,
                        # Cheap quantitative metadata (Qdrant payload).
                        # Useful for filtering and analytics; doesn't
                        # affect retrieval scoring.
                        "char_count": len(c_text),
                        "word_count": len(c_text.split()),
                    }
                    if page is not None:
                        child_meta["page"] = page
                    all_chunks.append(
                        {
                            "text": child_with_header,
                            "metadata": child_meta,
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
