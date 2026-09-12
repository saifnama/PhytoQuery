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

from backend.src.chat.embeddings import BloomIndexEmbeddings, RAGConfig, config
from backend.src.common.paths import kb_dir
from backend.src.chat.citations import _is_chrome_sentence
from backend.src.chat.embeddings import _build_cuda_model_kwargs
from backend.src.settings import RAG_CONTEXT_WINDOW

# Re-exported for tests that reference these by module-attr.
from langchain_core.documents import Document  # noqa: F401

try:
    from sentence_transformers import CrossEncoder  # noqa: F401
except Exception:  # allow tests / minimal envs to inject a stub later
    CrossEncoder = None  # type: ignore[assignment]

class _RetrievalMixin:
    pass  # methods attached below (verbatim moves)

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

                if self._device.startswith("cuda"):
                    cuda_kwargs = _build_cuda_model_kwargs(
                        enable_flash_attn=RAG_FLASH_ATTENTION,
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


    def _hybrid_search(
        self,
        question: str,
        vectorstore,  # langchain_qdrant.QdrantVectorStore (lazy import)
        filter_files: Optional[List[str]] = None,
        k: int = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search via Qdrant's native dense + BM25 sparse fusion.

        The ``vectorstore`` is configured with
        ``RetrievalMode.HYBRID``, so ``similarity_search_with_score``
        delegates to Qdrant's ``query_points(prefetch=[dense, sparse],
        query=FusionQuery(Fusion.RRF))`` — server-side scoring, no
        Python BM25 cache, no manual RRF merge.

        Returns list of dicts with ``doc``, ``rrf``, ``vector_score``,
        and ``normalized_score`` keys (the rest of the pipeline reads
        these names; renaming would ripple).
        """
        from qdrant_client.http import models as qmodels

        if k is None:
            k = config.top_k

        # Build optional source filter (any-of over selected files).
        # langchain-qdrant accepts a qdrant_client Filter directly.
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
            fused = vectorstore.similarity_search_with_score(
                question, **search_kwargs
            )
        except Exception as e:
            logger.warning(
                f"Hybrid search failed ({e}); returning empty result list"
            )
            fused = []

        # Wrap (doc, score) tuples in the dict shape downstream
        # callers read. The score IS the fused RRF score from Qdrant
        # — we keep both ``rrf`` and ``vector_score`` so existing
        # call sites that reference either name keep working.
        result_dicts: List[Dict[str, Any]] = []
        for doc, score in fused:
            result_dicts.append({
                "doc": doc,
                "rrf": float(score),
                "vector_score": float(score),
            })

        # Normalize scores to 0-100 range for display in the UI.
        if result_dicts:
            max_score = result_dicts[0]["rrf"]
            for r in result_dicts:
                r["normalized_score"] = (
                    round((r["rrf"] / max_score) * 100) if max_score > 0 else 0
                )

        return result_dicts


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


    def _find_best_sentence(self, claim: str, chunk_text: str,
                              exclude: tuple = ()) -> str:
        """Return the sentence in ``chunk_text`` most relevant to
        ``claim`` according to the cross-encoder reranker.

        ``exclude`` holds already-used quote texts (normalized
        comparison): when several claims cite one chunk, each gets a
        distinct quote instead of all pointing at the same line.
        Falls back to the overall best when everything is excluded.

        Falls back to the first non-trivial sentence if the reranker
        is unavailable, the chunk has only one sentence, or scoring
        raises. This is the langroid principle applied at sentence
        granularity — the LLM gives the chunk number, deterministic
        scoring picks the supporting sentence.
        """
        if not chunk_text:
            return ""
        # Split paragraphs first (and strip markdown heading markers)
        # so a section heading can never glue itself to the following
        # paragraph and win as one long "sentence".
        paras = [
            p.strip()
            for p in re.split(r"\n\s*\n", chunk_text)
            if p.strip()
        ]
        sentences = []
        for p in paras:
            p = re.sub(r"(?m)^#{1,6}\s*", "", p).strip()
            sentences.extend(
                s.strip()
                for s in re.split(r"(?<=[.!?])\s+", p)
                if s.strip()
            )
        # Drop very short fragments (likely artifacts of bullet
        # points, abbreviations like "et al.", etc.) so we don't
        # rank noise above real sentences.
        sentences = [s for s in sentences if len(s) >= 20]
        # Drop publisher chrome (journal bars, DOI/URL lines,
        # all-caps headings) so quotes never anchor on footers
        # when body sentences score low. Fall back to the unfiltered
        # list if everything was chrome — a quote beats no quote.
        filtered = [s for s in sentences if not _is_chrome_sentence(s)]
        if filtered:
            sentences = filtered
        if not sentences:
            # Fallback to the chunk start if sentence-splitting yielded
            # nothing usable.
            return chunk_text[:300].strip()
        if len(sentences) == 1 or self.reranker is None:
            return sentences[0]
        try:
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
            ranked = sorted(
                range(len(scores)), key=lambda i: float(scores[i]),
                reverse=True,
            )
            excluded = {
                self._normalize_for_match(q) for q in exclude
            }
            for i in ranked:
                if self._normalize_for_match(sentences[i]) not in excluded:
                    return sentences[i]
            return sentences[ranked[0]]
        except Exception as e:
            logger.warning(f"Sentence reranker scoring failed: {e}")
            return sentences[0]


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


    @staticmethod
    def _apply_context_budget(
        items: List[Dict[str, Any]],
        budget_chars: int,
    ) -> tuple:
        """Split retrieved items into in-context vs over-budget.

        ``items`` are ``{chunk_id, block, record}`` in retrieval order.
        Sets ``record["context_status"]`` to ``"full"`` or
        ``"omitted_budget"`` and returns
        ``(context_parts, sources, citable_sources)`` — sources keeps
        every record (omitted ones stay visible as an honest signal),
        citable holds only what the model actually sees. The top item
        is always included even over budget: some context beats a
        guaranteed empty answer.
        """
        context_parts: List[str] = []
        sources: List[Dict[str, Any]] = []
        citable_sources: List[Dict[str, Any]] = []
        used = 0
        for i, item in enumerate(items):
            record = item["record"]
            block = item["block"]
            if i == 0 or used + len(block) <= budget_chars:
                record["context_status"] = "full"
                context_parts.append(block)
                citable_sources.append(record)
                used += len(block)
            else:
                record["context_status"] = "omitted_budget"
            sources.append(record)
        return context_parts, sources, citable_sources


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
            ``SDKLLMAdapter.invoke`` (full answer) or ``SDKLLMAdapter.astream``
            (token streaming); ``sources`` is the citation list that
            should accompany the answer.

        Both ``query()`` (non-streaming) and ``query_stream()`` use
        this so retrieval logic stays in one place.
        """
        user_files = self.list_indexed_files(user_id)
        # KB mode is active if user has no uploaded files OR explicitly unchecked all files
        is_kb_mode = (not user_files) or (filter_files is not None and len(filter_files) == 0)
        if not is_kb_mode:
            vectorstore = self._get_user_collection(user_id)
            effective_filter = filter_files
            _resolve_parent = lambda pid: self._get_parent_data(pid, user_id)
        else:
            kb_store = self._get_kb_collection()
            if kb_store is None:
                return {
                    "answer": "I couldn't find enough relevant context in the knowledge base to answer that question.",
                    "sources": [],
                    "is_kb_mode": True,
                }
            vectorstore = kb_store
            effective_filter = None
            _resolve_parent = self._get_kb_parent_data

        # 1. Retrieve many child chunks from vector store (up to 200)
        search_results = self._hybrid_search(
            question, vectorstore, effective_filter, k=config.retrieve_k,
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

                    # Instruction-aware reranking: prepend the domain
                    # instruction to each query in the pair (honored by
                    # instruction-tuned cross-encoders; inert otherwise).
                    if config.reranker_instruction:
                        instructed_query = f"{config.reranker_instruction}\n{query_text}"
                    else:
                        instructed_query = query_text

                    # Drop pairs where either side is whitespace-only — cross-encoder
                    # tokenizers can yield zero-length tensors for these.
                    pairs = []
                    valid_candidates = []
                    reranker_budget = self._get_reranker_max_tokens() // 2
                    for res in filtered_candidates:
                        passage = (res["doc"].page_content or "").strip()
                        if not passage:
                            continue
                        pairs.append([self._truncate_for_reranker(instructed_query, reranker_budget),
                                      self._truncate_for_reranker(passage, reranker_budget)])
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
        # unique parents. We oversample up to 3x ``max_parents`` first
        # so the diversity filter in step 3.5 has a richer candidate
        # pool to pick from. Without oversampling, a homogeneous top
        # of the rerank list would force all our chunks to come from
        # the same paragraph, which is exactly what produces the
        # "many citations on one line, all the same content" UX issue.
        # (3x, not 2x: generic queries like "summarize the methods"
        # rank the right children mid-list — a 20-wide pool cut them
        # off before diversity ever saw them.)
        candidate_pool_size = max(config.max_parents * 3, 6)
        parent_ids_seen: set[str] = set()
        candidate_parents: List[Dict[str, Any]] = []

        # Batch-fetch parent contexts for KB mode — one query instead of N+1.
        kb_parent_cache: Dict[str, Dict[str, Any]] = {}
        if is_kb_mode:
            import sqlite3 as _sqlite3
            from scripts.ingest_kb import get_parent_contexts

            all_parent_ids = list({
                r["doc"].metadata.get("parent_id")
                for r in reranked_children
                if r["doc"].metadata.get("content_type", "text") == "text"
                and r["doc"].metadata.get("parent_id")
            })
            kb_path = os.fspath(kb_dir() / "kb.sqlite")
            if os.path.exists(kb_path) and all_parent_ids:
                _conn = _sqlite3.connect(kb_path)
                kb_parent_cache = get_parent_contexts(all_parent_ids, _conn)
                _conn.close()

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
            if not is_kb_mode:
                parent_data = _resolve_parent(parent_id)
            else:
                parent_data = kb_parent_cache.get(parent_id, {})
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
        items: List[Dict[str, Any]] = []
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
                block = (
                    f"[{chunk_id}] [TABLE | {header_str}]:\n{d.page_content}"
                )
            else:
                block = (
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
            # For KB mode, carry paper-level metadata so the
            # reference builder can group chunks by paper and
            # emit Perplexity-style numbered references.
            if is_kb_mode:
                source_record["doc_title"] = d.metadata.get("doc_title", "")
                source_record["doc_doi"] = d.metadata.get("doc_doi", "")
            body_start = result.get("body_start")
            body_end = result.get("body_end")
            page = result.get("page") or d.metadata.get("page") or None
            if body_start is not None and body_end is not None:
                source_record["body_start"] = body_start
                source_record["body_end"] = body_end
            if page:
                source_record["page"] = page
            items.append(
                {"chunk_id": chunk_id, "block": block, "record": source_record}
            )

        # Context budget: ~4 chars/token, minus reserve for system
        # prompt + history + answer. Over-budget sources are marked,
        # kept in the frame, but excluded from the LLM context AND
        # the citation pool — never certify text the model never saw.
        budget_chars = max(1000, (RAG_CONTEXT_WINDOW - RAG_CONTEXT_RESERVE_TOKENS) * 4)
        context_parts, sources, citable_sources = self._apply_context_budget(
            items, budget_chars
        )
        context = "\n\n".join(context_parts)

        if not context_parts:
            no_context_msg = (
                "I couldn't find enough relevant context in the knowledge base to answer that question."
                if is_kb_mode
                else "I couldn't find enough relevant context in the selected sources to answer that question."
            )
            return {
                "answer": no_context_msg,
                "sources": [],
                "is_kb_mode": is_kb_mode,
            }

        # Build multi-turn messages for conversation memory.
        #
        # Inline self-report: the LLM cites the chunks it actually used
        # by appending [cN] markers. Every chunk in the prompt is headed
        # "[cN] [Title > Section (p. N)]" with its chunk_id — the ID
        # plus doc_title/source/section/page are all in the header, so
        # the model has the full provenance per chunk. It reports the
        # chunk_id it drew on; we parse those IDs to build References.
        # No second LLM call — the answer streams with its cites.
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
                "3. Cite chunks you used: at the end of each sentence "
                "that draws on the context, append [cN] using the "
                "chunk_id from its header (e.g. [c1], [c2]). Use only "
                "IDs from this prompt, you may list several like "
                "[c1][c3]. Leave a sentence uncited only if it is not "
                "from the context."
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

        return {
            "messages": messages,
            "sources": sources,
            "citable_sources": citable_sources,
            "is_kb_mode": is_kb_mode,
        }


_bm25_probe_done: bool = False


_bm25_probe_ok: bool = False


def _check_bm25() -> bool:
    """Return True if fastembed's BM25 stemmer works in this runtime."""
    global _bm25_probe_done, _bm25_probe_ok
    if _bm25_probe_done:
        return _bm25_probe_ok
    _bm25_probe_done = True
    try:
        import subprocess, sys
        result = subprocess.run(
            [
                sys.executable, "-c",
                "import py_rust_stemmers;"
                "s = py_rust_stemmers.SnowballStemmer('english');"
                "s.stem_word('testing')",
            ],
            capture_output=True,
            timeout=30,
        )
        _bm25_probe_ok = result.returncode == 0
        if not _bm25_probe_ok:
            logger.warning(
                "BM25 sparse encoder disabled: py_rust_stemmers crashed "
                "(exit code %d). Falling back to dense-only retrieval. "
                "Upgrade py_rust_stemmers or Python to re-enable hybrid search.",
                result.returncode,
            )
        else:
            logger.info("BM25 sparse encoder probe passed — hybrid retrieval available.")
    except Exception as exc:
        logger.warning(
            "BM25 probe could not run (%s), disabling sparse retrieval.", exc
        )
        _bm25_probe_ok = False
    return _bm25_probe_ok


