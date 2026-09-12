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

from backend.src.chat.citations import _CitationsMixin
from backend.src.chat.config import (
    LLM_TEMPERATURE,
    RAG_QUERY_TIMEOUT_SECONDS,
    RAG_SUMMARY_TIMEOUT_SECONDS,
)
from backend.src.chat.embeddings import RAGConfig, BloomIndexEmbeddings, config, get_optimal_device
from backend.src.chat.ingest import _IngestMixin
from backend.src.chat.llm import RAGLLMTimeoutError, RAGProviderAuthError, SDKLLMAdapter
from backend.src.chat.retrieval import _RetrievalMixin

class RAGService(_IngestMixin, _RetrievalMixin, _CitationsMixin):
    """Chat RAG engine. Logic lives in the _*Mixin splits; orchestration here."""

    _atexit_registered: bool = False


    def __init__(self, llm=None):
        self._device = get_optimal_device()
        self.embeddings = BloomIndexEmbeddings(
            model=config.embedding_model,
            device=self._device,
            mrl_dim=config.embedding_dim,
            query_instruction=config.embedding_instruction,
        )
        # Sync service device with embeddings (embeddings may have fallen back to cpu)
        self._device = self.embeddings.device

        # Defer reranker loading to first use so service construction is lightweight
        self._reranker = ...  # sentinel: not loaded yet
        self._reranker_lock = threading.Lock()

        # SDK-backed adapter over the one shared process-wide client.
        # Construction is lightweight (no I/O); unconfigured LLM raises
        # only when a query actually invokes the model.
        self.llm = llm or SDKLLMAdapter(temperature=LLM_TEMPERATURE)
        # Cache for per-user vectorstores
        self._vectorstore_cache: Dict[str, Any] = {}
        # Shared Qdrant local client (one DB, many per-user collections).
        self._qdrant_client = None
        self._qdrant_lock = threading.Lock()
        # One-shot guard so ``_get_qdrant_client`` registers the
        # ``atexit`` close-handler exactly once, even though the
        # method is called concurrently from many code paths.
        self._atexit_registered = False
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


    KB_COLLECTION_NAME = "kb_papers"


    _kb_vectorstore_cache: Optional[Any] = None


    _KB_DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


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
        answer_text = (response.content or "").strip()

        if answer_text and not prepared.get("is_kb_mode"):
            # Normal chat: LLM inline [cN] → References from parsed IDs.
            # No second LLM call, whole chunks, clickable.
            try:
                citable = prepared.get("citable_sources", prepared.get("sources", []))
                valid_ids = {s["chunk_id"] for s in citable if s.get("chunk_id")}
                nums = re.findall(r"\[\s*[Cc]?\s*(\d+)\s*\]", answer_text)
                used_ids = []
                seen: set = set()
                for n in nums:
                    cid = f"c{n}"
                    if cid in valid_ids and cid not in seen:
                        seen.add(cid)
                        used_ids.append(cid)
                # Clean visible text (no inline badges, just References)
                cleaned = re.sub(r"\[†\]", "", answer_text)
                cleaned = re.sub(r"\[\s*[Cc]?\s*\d+\s*\]", "", cleaned)
                cleaned = re.sub(r"  +", " ", cleaned)
                answer_text = cleaned
                # Fallback to all citable when LLM cites nothing
                ref_ids = used_ids if used_ids else [s["chunk_id"] for s in citable if s.get("chunk_id")]
                if ref_ids:
                    pseudo = [
                        {
                            "chunk_id": cid,
                            "page": next((s.get("page") for s in citable if s.get("chunk_id") == cid), None),
                            "title": next((s.get("doc_title") or s.get("title") or "" for s in citable if s.get("chunk_id") == cid), ""),
                            "source": next((s.get("source") or "" for s in citable if s.get("chunk_id") == cid), ""),
                        }
                        for cid in ref_ids
                    ]
                    refs = self._build_references_block(pseudo, citable)
                    if refs and "**References**" not in answer_text:
                        answer_text = answer_text + refs
            except Exception as e:
                logger.warning(f"Fallback citation build failed: {e}")

        if prepared.get("is_kb_mode") and answer_text:
            seen_papers: Dict[str, Dict[str, str]] = {}
            for s in prepared.get("citable_sources", prepared.get("sources", [])):
                key = s.get("source", "")
                if key and key not in seen_papers:
                    seen_papers[key] = {
                        "title": s.get("doc_title", ""),
                        "doi": s.get("doc_doi", ""),
                    }
            if seen_papers:
                ref_lines = ["\n\n---\n\n**References**\n"]
                for idx, (fname, meta) in enumerate(seen_papers.items(), 1):
                    title = meta["title"] or fname
                    doi = meta["doi"]
                    line = f"{idx}. {title}"
                    if doi:
                        line += f". DOI: {doi}"
                    ref_lines.append(line)
                answer_text += "\n" + "\n".join(ref_lines)

        return {
            "answer": answer_text,
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

        # Attribution + reference pool: only sources the model actually
        # saw in context. The full list (incl. omitted_budget records)
        # goes to the frames as an honest signal.
        citable_sources = prepared.get("citable_sources", prepared.get("sources", []))

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
                s["chunk_id"] for s in citable_sources
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
                s["chunk_id"] for s in citable_sources
            )
            # Build per-chunk diagnostic mapping. We also OFFSET-VALIDATE
            # each chunk: reload the saved paper markdown once per
            # source, slice ``markdown[body_start:body_end]``, and
            # compare to the ``chunk_text`` field. If they diverge,
            # the offset is stale OR ``_strip_to_body`` is producing
            # a different body than what was stored at ingest.
            sources_for_log = citable_sources
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

        is_kb_mode = prepared.get("is_kb_mode", False)

        if is_kb_mode:
            corrected_answer = accumulated
            citations: List[Dict[str, Any]] = []
            if accumulated.strip():
                seen_papers: Dict[str, Dict[str, str]] = {}
                for s in citable_sources:
                    key = s.get("source", "")
                    if key and key not in seen_papers:
                        seen_papers[key] = {
                            "title": s.get("doc_title", ""),
                            "doi": s.get("doc_doi", ""),
                        }
                if seen_papers:
                    ref_lines = ["\n\n---\n\n**References**\n"]
                    for idx, (fname, meta) in enumerate(seen_papers.items(), 1):
                        title = meta["title"] or fname
                        doi = meta["doi"]
                        line = f"{idx}. {title}"
                        if doi:
                            line += f". DOI: {doi}"
                        ref_lines.append(line)
                    corrected_answer = accumulated + "\n" + "\n".join(ref_lines)
        else:
            # Inline self-report: LLM appends [cN] per sentence it used.
            # We parse those IDs (the chunk_ids we put in headers) to
            # build References — whole chunks, section + page, clickable.
            # No second LLM call. Visible answer stays clean (no inline
            # badges) per user request — References below is the list.
            citations: List[Dict[str, Any]] = []
            rewrite_error: Optional[str] = None
            attribution_mode = "llm_inline"
            # Parse *before* stripping — these are the LLM's report.
            valid_ids = {s["chunk_id"] for s in citable_sources if s.get("chunk_id")}
            nums = re.findall(r"\[\s*[Cc]?\s*(\d+)\s*\]", accumulated)
            used_ids: List[str] = []
            seen: set = set()
            for n in nums:
                cid = f"c{n}"
                if cid in valid_ids and cid not in seen:
                    seen.add(cid)
                    used_ids.append(cid)
            # Clean visible text: strip leaked daggers + the [cN] we
            # just parsed (so no inline badges, just References).
            cleaned = re.sub(r"\[†\]", "", accumulated)
            cleaned = re.sub(r"\[\s*[Cc]?\s*\d+\s*\]", "", cleaned)
            cleaned = re.sub(r"  +", " ", cleaned)
            corrected_answer = cleaned
            # Fallback when LLM sends 0 markers (non-compliance) —
            # scorer guess so References is never empty.
            fallback_used = False
            if not used_ids and accumulated.strip():
                try:
                    _, diag_cites = self._attribute_sentences_to_sources(
                        accumulated,
                        citable_sources,
                        floor=RAG_CITATION_SUPPORT_FLOOR,
                        margin=RAG_CITATION_SUPPORT_MARGIN,
                    )
                    # Use scorer's picks as used_ids
                    for c in diag_cites:
                        cid = c.get("chunk_id")
                        if cid in valid_ids and cid not in seen:
                            seen.add(cid)
                            used_ids.append(cid)
                    citations = diag_cites
                    fallback_used = True
                except Exception as e:
                    logger.warning(f"Fallback attribution failed: {e}")
                    citations = []

            if accumulated.strip():
                citations_summary = [
                    {
                        "chunk_id": cid,
                        "quote_preview": next(
                            (c.get("quote") or "")[:40] for c in citations if c.get("chunk_id") == cid
                        ) if citations else "",
                    }
                    for cid in used_ids
                ]
                logger.warning(
                    "[CITATION DIAG] mode=%s omitted=%d parsed=%s fallback=%s citations=%s%s",
                    attribution_mode,
                    len(prepared.get("sources", [])) - len(citable_sources),
                    used_ids,
                    fallback_used,
                    citations_summary,
                    f" error={rewrite_error!r}" if rewrite_error else "",
                )

        # References = the chunks the LLM said it used (parsed [cN]),
        # or the fallback set when it said none. Whole chunks,
        # section + page + title, clickable to markdown preview.
        if not is_kb_mode and corrected_answer.strip():
            # Prefer parsed LLM ids; if none, show all citable as honest
            # "available to LLM" list (never empty).
            ref_ids = used_ids if used_ids else [s["chunk_id"] for s in citable_sources if s.get("chunk_id")]
            pseudo = [
                {
                    "chunk_id": cid,
                    "page": next((s.get("page") for s in citable_sources if s.get("chunk_id") == cid), None),
                    "title": next((s.get("doc_title") or s.get("title") or "" for s in citable_sources if s.get("chunk_id") == cid), ""),
                    "source": next((s.get("source") or "" for s in citable_sources if s.get("chunk_id") == cid), ""),
                }
                for cid in ref_ids
            ]
            refs = self._build_references_block(pseudo, citable_sources)
            if refs:
                if "**References**" not in corrected_answer:
                    corrected_answer = corrected_answer + refs

        if corrected_answer != accumulated:
            yield {"type": "answer_corrected", "text": corrected_answer}

        yield {"type": "citations", "citations": citations}

        yield {"type": "done"}


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


_rag_service: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def peek_rag_service() -> Optional[RAGService]:
    return _rag_service


