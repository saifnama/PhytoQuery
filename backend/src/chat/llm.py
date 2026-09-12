"""Split of services/rag_engine.py (Phase 2b-2). Verbatim moves — no logic changes."""
import asyncio
import gc
import hashlib
import json
import logging
import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Optional

from backend.src.settings import (
    LLM_CONTEXT_WINDOW,
    QDRANT_API_KEY,
    QDRANT_DIR,
    QDRANT_URL,
    RAG_CITATION_MODE,
    RAG_CITATION_SUPPORT_FLOOR,
    RAG_CITATION_SUPPORT_MARGIN,
    RAG_CONTEXT_RESERVE_TOKENS,
    RAG_EMBEDDING_DIM,
    RAG_EMBEDDING_INSTRUCTION,
    RAG_EMBEDDING_MODEL,
    RAG_FLASH_ATTENTION,
    RAG_MULTI_GPU,
    RAG_RERANKER_MODEL,
    RAG_SIMILARITY_THRESHOLD,
    RAG_TEMPERATURE,
    RAG_TOP_K,
    _safe_float,
    _safe_int,
)
from backend.src.common.llm_client import (
    LLMAuthError,
    LLMRateLimitError,
    LLMTimeoutError,
    SharedLLMClient,
    get_llm_client,
)
from backend.src.common.uploads import (
    delete_user_upload_file,
    delete_user_uploads,
    extract_paper_markdown,
    get_user_markdown_file_path,
)

logger = logging.getLogger(__name__)

RAGProviderAuthError = LLMAuthError


RAGLLMTimeoutError = LLMTimeoutError


class SDKLLMAdapter:
    """Provider-neutral LLM adapter over the shared OpenAI SDK client.

    Keeps the historical ``invoke`` / ``astream`` contract so RAGService
    call sites are unchanged. Legacy constructor kwargs (base_url,
    model, provider, api_key, num_ctx, ...) are accepted and ignored.
    """

    def __init__(self, temperature=0.1, _client=None, **_ignored):
        self.temperature = temperature
        self._client = _client  # injectable for tests

    def _shared(self) -> SharedLLMClient:
        if self._client is not None:
            return self._client
        return get_llm_client()

    async def invoke(
        self,
        prompt: str = None,
        messages: list = None,
        max_retries: int = 3,
        base_delay: float = 2.0,
        timeout_seconds=None,
        response_format=None,
        thinking: Optional[bool] = None,
        **_ignored,
    ):
        """Invoke the LLM with either a simple prompt or messages list.

        Args:
            response_format: When set to ``{"type": "json_object"}`` the
                LLM is asked for JSON mode (citation extraction pass).
        """
        if messages is None and prompt is None:
            raise ValueError("Either prompt or messages must be provided")
        last_error = None
        for attempt in range(max(1, max_retries)):
            try:
                return await self._shared().invoke(
                    prompt=prompt,
                    messages=messages,
                    temperature=self.temperature,
                    response_format=response_format,
                    timeout_seconds=timeout_seconds,
                    thinking=thinking,
                )
            except LLMRateLimitError as exc:
                last_error = exc
                delay = base_delay * (2**attempt)
                logger.warning(
                    f"LLM rate limited, retrying after {delay}s "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(delay)
        raise last_error

    async def astream(
        self,
        prompt: str = None,
        messages: list = None,
        thinking: Optional[bool] = None,
        **_ignored,
    ):
        """Async-iterate streamed text chunks from the LLM."""
        if messages is None and prompt is None:
            raise ValueError("Either prompt or messages must be provided")
        async for chunk in self._shared().astream(
            prompt=prompt, messages=messages, temperature=self.temperature,
            thinking=thinking,
        ):
            yield chunk


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


