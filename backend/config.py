"""
PhytoQuery — Unified Configuration
====================================
Usage:
    from backend.config import env, env_int, env_float, resolve_llm_settings
"""

import os
from pathlib import Path

# Auto-load env files. Precedence (highest → lowest):
#   1. Real OS env vars (always win — Slurm's CUDA_VISIBLE_DEVICES,
#      ``docker run -e``, systemd ``Environment=``).
#   2. ``.env.<PHYTOQUERY_PROFILE>`` — profile-specific overrides.
#      Profile = a single env var that switches the whole webapp's
#      config in one flip. Examples:
#        PHYTOQUERY_PROFILE=macbook  → loads .env.macbook
#        PHYTOQUERY_PROFILE=server   → loads .env.server
#        PHYTOQUERY_PROFILE=demo     → loads .env.demo
#      Unset/empty = no profile, just the base ``.env``. Replaces
#      the old "cp .env.macbook .env" shuffle — set the profile
#      once per environment (shell, systemd, Slurm batch) and the
#      right values load automatically.
#   3. ``.env`` — base / shared defaults.
#   4. Defaults declared in this module.
#
# ``override=False`` everywhere: a higher-priority source already
# in env stays put; each file only fills in values still unset.
try:
    from dotenv import load_dotenv
    _project_root = Path(__file__).resolve().parent.parent

    _profile = os.environ.get("PHYTOQUERY_PROFILE", "").strip().lower()
    if _profile:
        _profile_file = _project_root / f".env.{_profile}"
        if _profile_file.exists():
            load_dotenv(_profile_file, override=False)

    load_dotenv(_project_root / ".env", override=False)
except ImportError:
    pass  # python-dotenv not installed; rely on system env vars


# ---------------------------------------------------------------------------
# Helpers — thin wrappers around os.getenv with type casting
# ---------------------------------------------------------------------------

def env(key: str, default: str = "") -> str:
    """Read a string from the environment."""
    return os.getenv(key, default).strip()


def env_int(key: str, default: int = 0) -> int:
    """Read an integer from the environment (empty/garbage → default)."""
    try:
        val = os.getenv(key, "").strip()
        return int(val) if val else default
    except (TypeError, ValueError):
        return default


def env_float(key: str, default: float = 0.0) -> float:
    """Read a float from the environment (empty/garbage → default)."""
    try:
        val = os.getenv(key, "").strip()
        return float(val) if val else default
    except (TypeError, ValueError):
        return default


def env_bool(key: str, default: bool = False) -> bool:
    """Read a boolean from the environment (accepts 1/true/yes)."""
    val = os.getenv(key, str(default)).strip().lower()
    return val in {"1", "true", "yes"}


def env_optional(key: str):
    """Read a value that may be intentionally unset (returns None if missing/empty)."""
    val = os.getenv(key, "").strip()
    return val if val else None


# ---------------------------------------------------------------------------
# RAG Settings (non-LLM; the LLM connection is LLM_API_* above)
# ---------------------------------------------------------------------------

# --- Qdrant: server mode vs embedded mode ---
# QDRANT_URL: when set (e.g. ``http://localhost:6333``), connect to
# a Qdrant Server (Docker, native binary, or Qdrant Cloud). This is the
# recommended mode for production and any Linux/HPC environment where
# filesystem ``flock()`` is unreliable (Lustre, some NFS configs). Server
# mode supports concurrent clients and ``uvicorn --workers N > 1``.
#
# Spin one up next to the backend with the bundled helper:
#   ./scripts/qdrant.sh start        # Linux / macOS
#   .\scripts\qdrant.ps1 start       # Windows
#
# Or directly via Docker:
#   docker run -d --name phytoquery-qdrant \
#     -p 6333:6333 -p 6334:6334 \
#     -v "$HOME/.local/share/phytoquery/qdrant_storage:/qdrant/storage" \
#     --restart unless-stopped \
#     qdrant/qdrant:v1.18.0
QDRANT_URL = env("QDRANT_URL")

# QDRANT_API_KEY: optional bearer token for Qdrant Cloud or any
# server started with ``--service.api_key=...``. Only consulted when
# QDRANT_URL is set (server mode).
QDRANT_API_KEY = env("QDRANT_API_KEY")

# QDRANT_DIR: storage path for the *embedded* local client. Only used
# when QDRANT_URL is empty. Leave empty for the default
# (``<repo>/data/qdrant/``). Set to a writable LOCAL-DISK path on systems
# where the default lives on a filesystem with broken ``flock()`` support.
# ``~`` is expanded; relative paths are resolved to absolute at startup.
QDRANT_DIR = env("QDRANT_DIR")

RAG_TEMPERATURE = env_float("RAG_TEMPERATURE", 0.1)


def _safe_int(key: str, default: int) -> int:
    """Env int that never crashes startup on garbage input."""
    try:
        return int(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


# Canonical context window lives under LLM_* (it describes the shared
# target model, not the RAG pipeline). RAG_CONTEXT_WINDOW remains as a
# deprecated fallback; LLM_CONTEXT_WINDOW wins when set.
_llm_ctx = _safe_int("LLM_CONTEXT_WINDOW", 0)
LLM_CONTEXT_WINDOW = _llm_ctx or None
RAG_CONTEXT_WINDOW = LLM_CONTEXT_WINDOW or _safe_int("RAG_CONTEXT_WINDOW", 8192)
RAG_EMBEDDING_MODEL = env("RAG_EMBEDDING_MODEL", "BAAI/bge-m3")
_dim = env_optional("RAG_EMBEDDING_DIM")
RAG_EMBEDDING_DIM = int(_dim) if _dim else None
RAG_EMBEDDING_INSTRUCTION = env("RAG_EMBEDDING_INSTRUCTION")
RAG_TOP_K = env_int("RAG_TOP_K", 10)
RAG_SIMILARITY_THRESHOLD = env_float("RAG_SIMILARITY_THRESHOLD", 0.85)
RAG_RERANKER_MODEL = env("RAG_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
# Citation attribution: "fast" (default) skips the post-stream LLM
# chunk-selection call and attributes each answer sentence
# deterministically (verbatim fast-path, then one batched
# cross-encoder pass). Any other value keeps the legacy
# "llm_select" path unchanged.
RAG_CITATION_MODE = env("RAG_CITATION_MODE", "fast")


def _safe_float(key: str, default: float) -> float:
    """Env float that never crashes startup on garbage input."""
    try:
        return float(os.getenv(key, str(default)))
    except (TypeError, ValueError):
        return default


# A sentence cites its best chunk when EITHER holds:
# - absolute certainty: best score >= FLOOR, or
# - distinctiveness: best score beats that sentence's mean chunk
#   score by >= MARGIN. The margin rule is scale-invariant — it
#   survives domain logit shift (e.g. an ms-marco scorer going
#   all-negative on biomedical text), where a fixed threshold
#   would silently leave everything uncited.
RAG_CITATION_SUPPORT_FLOOR = _safe_float("RAG_CITATION_SUPPORT_FLOOR", 0.0)
RAG_CITATION_SUPPORT_MARGIN = _safe_float("RAG_CITATION_SUPPORT_MARGIN", 1.0)

# Tokens reserved out of the context window for the system prompt,
# conversation history, and the answer itself. The remainder is the
# retrieval budget: sources past it are marked omitted_budget, kept
# in the frame as an honest signal, but excluded from both the LLM
# context and the citation pool. Char estimate uses 4/token.
RAG_CONTEXT_RESERVE_TOKENS = int(_safe_float("RAG_CONTEXT_RESERVE_TOKENS", 2000))
RAG_MULTI_GPU = env_bool("RAG_MULTI_GPU", False)
RAG_FLASH_ATTENTION = env_bool("RAG_FLASH_ATTENTION", True)

# ---------------------------------------------------------------------------
# NER Settings (non-LLM; the LLM connection is LLM_API_* above)
# ---------------------------------------------------------------------------

# NER_HYBRID=false disables the LLM phase entirely (dictionary-only
# extraction, zero LLM calls/cost). Default true preserves hybrid mode.
NER_HYBRID = env_bool("NER_HYBRID", True)
NER_CONFIDENCE_THRESHOLD = env_float("NER_CONFIDENCE_THRESHOLD", 0.7)
NER_CHUNK_WORDS = env_int("NER_CHUNK_WORDS", 250)
NER_MAX_CHUNKS = env_int("NER_MAX_CHUNKS", 3)
# Word-chunk size for Analyse PDF uploads, which run the full
# dictionary+LLM pipeline (NERService.process_sections) over the whole
# document. Larger chunks mean fewer LLM calls per paper.
NER_UPLOAD_CHUNK_WORDS = env_int("NER_UPLOAD_CHUNK_WORDS", 600)

# Validation retries per section before falling back to
# dictionary-only entities for that section.
NER_MAX_ATTEMPTS = env_int("NER_MAX_ATTEMPTS", 2)
# Wall-clock budget for the whole LLM phase of a paper extraction.
# When exceeded, remaining sections skip the LLM and keep dictionary
# entities, so /paper/json always returns well inside the frontend's
# 10-minute request timeout. 0 disables the budget (not recommended).
NER_BUDGET_SECONDS = env_float("NER_BUDGET_SECONDS", 240.0)


# ---------------------------------------------------------------------------
# Unified LLM Settings (OpenAI Python SDK, Chat Completions)
# ---------------------------------------------------------------------------
# One application-wide connection for RAG, NER, and RAGAS evaluation.
# The target server changes only through these three variables:
#
#   LLM_API_BASE_URL=https://api.openai.com/v1
#   LLM_API_KEY=...
#   LLM_MODEL=...
#
# LLM_API_BASE_URL must be a /v1 API root — never an operation path
# (/v1/chat/completions) or the native Ollama protocol (/api/chat).

LLM_API_BASE_URL = env("LLM_API_BASE_URL")
LLM_API_KEY = env("LLM_API_KEY")
LLM_MODEL = env("LLM_MODEL")
LLM_TIMEOUT_SECONDS = _safe_float("LLM_TIMEOUT_SECONDS", 300.0)
LLM_MAX_RETRIES = _safe_int("LLM_MAX_RETRIES", 2)
# Thinking control. LLM_THINKING=false (default) appends
# LLM_NO_THINK_DIRECTIVE to the last user message — a Qwen-template
# convention, inert text elsewhere. Set LLM_THINKING=true to allow
# chain-of-thought, or empty the directive to append nothing.
LLM_THINKING = env_bool("LLM_THINKING", False)
LLM_NO_THINK_DIRECTIVE = env("LLM_NO_THINK_DIRECTIVE", "/no_think")
# Restore of the pre-unification compat-path behavior: send
# chat_template_kwargs.enable_thinking via SDK extra_body. The old code
# sent this on every OpenRouter/llama.cpp call. Default off — official
# OpenAI rejects unknown top-level fields, so enable only for
# compatible servers (OpenRouter, llama.cpp, vLLM) that tolerate it.
LLM_CHAT_TEMPLATE_KWARGS = env_bool("LLM_CHAT_TEMPLATE_KWARGS", False)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"


class LLMConfigError(ValueError):
    """Raised when the unified LLM connection is missing or malformed."""


def _normalize_llm_base_url(url: str) -> str:
    """Normalize to a /v1 API root. Forgives a trailing
    /v1/chat/completions operation path; rejects native /api/chat."""
    url = url.strip().rstrip("/")
    if "/api/chat" in url or url.rstrip("/").endswith("/api/tags"):
        raise LLMConfigError(
            "LLM_API_BASE_URL must be an OpenAI-compatible /v1 root, "
            f"not a native Ollama path: {url!r}. "
            "Ollama serves an OpenAI-compatible API at <host>/v1 — "
            "point LLM_API_BASE_URL there instead."
        )
    if url.endswith("/v1/chat/completions"):
        url = url[: -len("/chat/completions")]
    if not url.rstrip("/").endswith("/v1"):
        raise LLMConfigError(
            "LLM_API_BASE_URL must be a /v1 API root "
            f"(e.g. https://api.openai.com/v1), got: {url!r}."
        )
    return url


def _legacy_llm_fallback() -> "dict | None":
    """Deprecated RAG_* > NER_* mapping, used only when NO LLM_API_*
    variable is set. Lets pre-unification .env files keep working
    untouched; ignored the moment any LLM_API_* variable is present."""
    import os as _os
    import warnings

    def _openrouter(prefix: str):
        api_key = _os.getenv(f"{prefix}_OPENROUTER_API_KEY", "").strip()
        if api_key and api_key not in {"sk-", "sk"}:
            return {
                "base_url": "https://openrouter.ai/api/v1",
                "api_key": api_key,
                "model": _os.getenv(f"{prefix}_OPENROUTER_MODEL", "").strip(),
            }
        return None

    def _compat(url_key: str, model_key: str, key_key: str):
        url = _os.getenv(url_key, "").strip().rstrip("/")
        if not url:
            return None
        # Accept any server-base form, normalize to the /v1 root.
        if url.endswith("/v1/chat/completions"):
            url = url[: -len("/chat/completions")]
        elif "/api/chat" in url:  # native Ollama host → its OpenAI-compat root
            url = url.split("/api/chat")[0].rstrip("/") + "/v1"
        elif not url.endswith("/v1"):
            url = url + "/v1"
        return {
            "base_url": url,
            "api_key": _os.getenv(key_key, "").strip(),
            "model": _os.getenv(model_key, "default").strip() or "default",
        }

    def _ollama(prefix: str):
        host = _os.getenv(f"{prefix}_OLLAMA_URL", "").strip().rstrip("/")
        if not host:
            return None
        return {
            "base_url": host + "/v1",
            "api_key": "",
            "model": _os.getenv(f"{prefix}_OLLAMA_MODEL", "").strip(),
        }

    for prefix in ("RAG", "NER"):
        hit = (
            _compat(
                f"{prefix}_LLAMACPP_URL",
                f"{prefix}_LLAMACPP_MODEL",
                f"{prefix}_LLAMACPP_API_KEY",
            )
            or _openrouter(prefix)
            or _ollama(prefix)
        )
        if hit and hit.get("model"):
            warnings.warn(
                f"No LLM_API_* variables set; using legacy {prefix}_* "
                "configuration. Set LLM_API_BASE_URL/LLM_API_KEY/LLM_MODEL "
                "instead — the legacy keys are deprecated.",
                DeprecationWarning,
                stacklevel=3,
            )
            return hit
    return None


class LLMSettings:
    """Immutable application-wide LLM connection."""

    __slots__ = ("base_url", "api_key", "model", "timeout", "max_retries",
                 "thinking", "no_think_directive", "chat_template_kwargs")

    def __init__(self, base_url: str, api_key: str, model: str,
                 timeout: float = 300.0, max_retries: int = 2,
                 thinking: bool = False, no_think_directive: str = "/no_think",
                 chat_template_kwargs: bool = False):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.thinking = thinking
        self.no_think_directive = no_think_directive
        self.chat_template_kwargs = chat_template_kwargs

    def __repr__(self) -> str:  # never leak the key
        return (
            f"LLMSettings(base_url={self.base_url!r}, "
            f"model={self.model!r}, timeout={self.timeout}, "
            f"max_retries={self.max_retries})"
        )


def resolve_llm_settings() -> LLMSettings:
    """Resolve the one shared LLM connection.

    Precedence: LLM_API_* env > legacy RAG_*/NER_* fallback (deprecated)
    > official OpenAI default (requires LLM_API_KEY + LLM_MODEL).
    Raises LLMConfigError with an actionable message when incomplete.
    """
    import os as _os

    has_unified = any(
        _os.getenv(k, "").strip()
        for k in ("LLM_API_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
    )
    if has_unified:
        base_url = _os.getenv("LLM_API_BASE_URL", "").strip() or DEFAULT_OPENAI_BASE_URL
        base_url = _normalize_llm_base_url(base_url)
        api_key = _os.getenv("LLM_API_KEY", "").strip()
        model = _os.getenv("LLM_MODEL", "").strip()
        if not model:
            raise LLMConfigError("LLM_MODEL is not set. Set LLM_MODEL to a model ID understood by LLM_API_BASE_URL.")
        if not api_key:
            if "api.openai.com" in base_url:
                raise LLMConfigError("LLM_API_KEY is not set. Set LLM_API_KEY for the official OpenAI API.")
            api_key = "not-needed"  # ponytail: placeholder for keyless compat servers
    else:
        legacy = _legacy_llm_fallback()
        if legacy is None:
            raise LLMConfigError(
                "No LLM configured. Set LLM_API_BASE_URL, LLM_API_KEY, "
                "and LLM_MODEL (see .env.example)."
            )
        base_url = _normalize_llm_base_url(legacy["base_url"])
        api_key = legacy["api_key"] or "not-needed"  # ponytail: placeholder for keyless compat servers
        model = legacy["model"]
    # Transport + thinking knobs read live (import-time module constants
    # would freeze profile-switched values).
    return LLMSettings(
        base_url=base_url, api_key=api_key, model=model,
        timeout=_safe_float("LLM_TIMEOUT_SECONDS", LLM_TIMEOUT_SECONDS),
        max_retries=_safe_int("LLM_MAX_RETRIES", LLM_MAX_RETRIES),
        thinking=_os.getenv("LLM_THINKING", "").strip().lower() in {"1", "true", "yes"},
        no_think_directive=_os.getenv("LLM_NO_THINK_DIRECTIVE", "/no_think"),
        chat_template_kwargs=_os.getenv("LLM_CHAT_TEMPLATE_KWARGS", "").strip().lower() in {"1", "true", "yes"},
    )


def llm_status() -> dict:
    """Non-billable configuration check for readiness probes."""
    try:
        settings = resolve_llm_settings()
    except LLMConfigError as exc:
        return {"llm": f"unconfigured: {exc}"}
    return {"llm": "configured", "model": settings.model}
