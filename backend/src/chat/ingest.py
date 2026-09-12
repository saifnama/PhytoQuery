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

from backend.src.chat.embeddings import BloomIndexEmbeddings, RAGConfig, config
from backend.src.chat.retrieval import _check_bm25

class _IngestMixin:
    pass  # methods attached below (verbatim moves)

    def _get_semantic_splitter(self):
        """Lazy-init SemanticChunker for child-level semantic splitting."""
        if getattr(self, "_semantic_splitter", None) is None:
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

        This ensures Qdrant collections are versioned by embedding
        config, preventing dimension mismatch when switching models.

        **Idempotency contract**: every call MUST return the same
        hash for the same configured model. ``BloomIndexEmbeddings``
        initializes ``model_dim`` to a placeholder (2560 — Qwen3-4B's
        dim) and only updates it to the real value once
        ``_load_model`` actually loads the SentenceTransformer. If
        ``_get_collection_suffix`` was called before that load, it
        produced a hash over ``"<name>:2560"`` rather than
        ``"<name>:<real_dim>"`` — a different hash for the same
        model, depending on call order. Forcing the load here makes
        the hash a pure function of the configured model: idempotent
        and order-independent. After the first call
        ``_ensure_model_loaded`` is a no-op, so the cost is one-time.
        """
        self.embeddings._ensure_model_loaded()
        model_key = f"{self.embeddings.model_name}:{self.embeddings.model_dim}"
        return hashlib.md5(model_key.encode()).hexdigest()[:8]


    def _get_qdrant_client(self):
        """Lazy-init the shared local QdrantClient.

        One client points at ``tmp/qdrant/``. Per-user isolation is
        achieved via collection names (``user_<safe_user_id>_<8-char-hash>``)
        within that single directory rather than per-user folders. This
        avoids the SQLite-tenant gymnastics that plagued the old Chroma
        path and lets ``delete_collection`` be a single atomic call.

        On first creation, registers an ``atexit`` handler that
        guarantees ``close()`` runs on *any* process-exit path
        (Ctrl+C / SIGINT, ``sys.exit``, test teardown, normal
        completion, FastAPI lifespan). atexit fires while the
        interpreter is still functional — modules are not yet being
        torn down — so the lazy imports inside ``close()`` succeed
        and no "Exception ignored in: __del__" traceback prints.
        This is the universal cleanup; the FastAPI lifespan hook is
        a complementary earlier-fire path for the graceful-shutdown
        case (atexit's idempotent ``close()`` then no-ops).
        """
        if self._qdrant_client is not None:
            return self._qdrant_client
        with self._qdrant_lock:
            if self._qdrant_client is not None:
                return self._qdrant_client
            from qdrant_client import QdrantClient

            # Two modes, env-toggled via ``QDRANT_URL``:
            #
            #   - SET (e.g. ``http://localhost:6333``) → connect to a
            #     Qdrant Server (Docker, native binary, Qdrant Cloud).
            #     Recommended for Linux/HPC and any environment where
            #     filesystem flock() is unreliable (Lustre, some NFS).
            #     Supports concurrent clients + ``uvicorn --workers N``.
            #
            #   - UNSET → embedded local-mode at ``config.qdrant_dir``.
            #     Zero-dependency, works on every OS, but only one
            #     process can hold the storage lock at a time.
            #
            # Call sites only see ``self._qdrant_client`` — they don't
            # care which mode produced it. The Qdrant Python client
            # exposes the same interface for both.
            if QDRANT_URL:
                # ``api_key`` is optional — qdrant-client accepts None
                # gracefully (no Authorization header sent). Qdrant Cloud
                # and any server started with ``--service.api_key=...``
                # require it; plain Docker/local server doesn't.
                self._qdrant_client = QdrantClient(
                    url=QDRANT_URL,
                    api_key=QDRANT_API_KEY or None,
                )
                _auth_note = " (authenticated)" if QDRANT_API_KEY else ""
                logger.info(
                    f"Initialized Qdrant remote client at {QDRANT_URL}{_auth_note}"
                )
            else:
                os.makedirs(config.qdrant_dir, exist_ok=True)
                self._qdrant_client = QdrantClient(path=config.qdrant_dir)
                logger.info(
                    f"Initialized Qdrant local client at {config.qdrant_dir}"
                )

            # Universal cleanup registration — fires on every exit
            # path that runs Python code (everything except SIGKILL
            # / interpreter abort, which wouldn't run __del__ either).
            # Registered exactly once per service instance.
            if not self._atexit_registered:
                import atexit
                atexit.register(self._atexit_close)
                self._atexit_registered = True
            return self._qdrant_client


    def _atexit_close(self) -> None:
        """``atexit``-safe wrapper around ``close()``.

        atexit handlers must never raise — any exception here is
        silently swallowed. Importantly we do *not* log inside this
        method because logger handlers may already be in
        teardown-flush state by the time atexit runs.
        """
        try:
            self.close()
        except Exception:
            pass


    def close(self) -> None:
        """Close the embedded Qdrant client cleanly during app shutdown.

        Without this, the client's ``__del__`` runs during Python
        interpreter teardown, by which point ``sys.meta_path`` is
        already ``None`` — so the lazy import inside
        ``qdrant_local.close()`` raises ``ImportError`` and Python
        prints "Exception ignored in: <function QdrantClient.__del__>".

        The exception is harmless (Python catches it), but it's
        cosmetically alarming in production logs and dissertation-
        demo terminals. Calling ``close()`` from the FastAPI
        ``lifespan`` shutdown handler runs the same cleanup *before*
        the interpreter starts tearing down, so imports succeed and
        no traceback is printed.

        Idempotent — safe to call after the client was never created
        (no-op) and safe to call twice (second call no-ops).
        Best-effort — any exception during close is logged, not
        raised, so it never blocks shutdown.
        """
        with self._qdrant_lock:
            client = self._qdrant_client
            if client is not None:
                if hasattr(client, "close"):
                    try:
                        client.close()
                        logger.info("Closed Qdrant local client cleanly")
                    except Exception as e:
                        logger.warning(
                            f"Qdrant client close raised (ignored): {e}"
                        )
                self._qdrant_client = None
                self._vectorstore_cache.clear()

        self._close_fastembed_models()


    def _close_fastembed_models(self) -> None:
        """Clean up fastembed models and terminate their loky process pool.

        On Python 3.14+ (unsupported — see the ceiling note in main.py),
        the loky reusable executor (used by fastembed's BM25 sparse
        encoder) does not implement __del__, so its IPC semaphore is
        never released during normal garbage collection, and the
        interpreter can segfault in shutdown cleanup as a result.

        We explicitly terminate and release the executor here so the
        semaphore is unlinked before the resource_tracker runs.

        Idempotent: safe to call when the models were never loaded.
        Best-effort: any exception is silently caught.
        """
        try:
            from joblib.externals.loky import reusable_executor as _loky_re

            executor = _loky_re._executor
            if executor is not None:
                executor.shutdown(wait=True)
                _loky_re._executor = None
                try:
                    _loky_re._executor_kwargs = None
                except AttributeError:
                    pass
        except Exception:
            pass

        try:
            if hasattr(self, "_sparse_embeddings_cache"):
                del self._sparse_embeddings_cache
        except Exception:
            pass

        gc.collect()


    def _ensure_qdrant_collection(self, client, collection_name: str) -> None:
        """Create the per-user Qdrant collection if it doesn't exist.

        Uses the native hybrid schema: named ``dense`` field (Qwen3/
        bge embeddings) plus a named ``sparse`` field with
        ``Modifier.IDF`` for BM25-style retrieval. A single
        ``query_points`` call with ``FusionQuery(Fusion.RRF)`` runs
        the hybrid merge server-side at query time — sparse vectors
        live alongside dense vectors per point, no separate Python
        BM25 cache needed.

        ``Modifier.IDF`` is what gives the sparse field BM25-style
        scoring (inverse document frequency weighting on the sparse
        term values). Without it, the sparse field would store raw
        term frequencies and produce naive TF scoring rather than
        BM25.

        Note on payload indexes: embedded Qdrant ignores them silently
        (we tested), so they're not used here. Filter queries still
        work via linear scan.
        """
        from qdrant_client.http import models as qmodels

        try:
            client.get_collection(collection_name)
            return
        except Exception:
            pass  # collection doesn't exist; fall through to create

        # Embedder must be loaded so model_dim is known. Cosine distance
        # matches the metric the LangChain Qdrant integration assumes
        # for normalized vectors.
        self.embeddings._ensure_model_loaded()
        model_dim = self.embeddings.model_dim
        requested_dim = config.embedding_dim
        # ``RAG_EMBEDDING_DIM`` is MRL truncation — it can shrink the
        # output dim, never expand it. If the user configured a dim
        # LARGER than the loaded model can produce (e.g. requested=1024
        # but loaded model is bge-small-en-v1.5 at 384, because the
        # preferred larger model failed to load), we'd otherwise create
        # a collection the model can never fill — langchain-qdrant then
        # rejects every upload with a "dimensions mismatch" error.
        # Clamp + log a clear warning so the misconfig is visible.
        if requested_dim and requested_dim > model_dim:
            logger.warning(
                f"RAG_EMBEDDING_DIM={requested_dim} exceeds the loaded "
                f"model's native output dim {model_dim} "
                f"({self.embeddings.model_name!r}). MRL truncation can "
                f"shrink but never expand — clamping vec_dim to {model_dim}. "
                f"If you want {requested_dim}-dim vectors, load a model "
                f"with model_dim >= {requested_dim} (e.g. BAAI/bge-m3 = 1024, "
                f"Qwen/Qwen3-Embedding-4B = 2560)."
            )
            vec_dim = model_dim
        else:
            vec_dim = requested_dim or model_dim

        client.create_collection(
            collection_name=collection_name,
            vectors_config={
                "dense": qmodels.VectorParams(
                    size=vec_dim, distance=qmodels.Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams(
                    modifier=qmodels.Modifier.IDF,
                ),
            },
        )
        logger.info(
            f"Created Qdrant collection {collection_name} (HYBRID: "
            f"dense size={vec_dim}, sparse IDF-BM25, distance=COSINE)"
        )


    def _get_user_collection(self, user_id: str):
        """Get or create a Qdrant-backed LangChain VectorStore for a user.

        Wraps a per-user Qdrant collection in ``QdrantVectorStore`` so the
        rest of the pipeline (``add_documents``, ``similarity_search_with_score``)
        keeps working unchanged. The collection name encodes the
        embedding-model+dim hash so switching models cannot collide.

        Always uses native hybrid retrieval:
          * ``retrieval_mode=RetrievalMode.HYBRID`` so
            ``add_documents`` writes both dense and sparse vectors
            per point and ``similarity_search`` runs RRF fusion
            server-side via Qdrant's ``Prefetch + FusionQuery``
            API.
          * Sparse vectors are produced by FastEmbed's
            ``Qdrant/bm25`` model (lazy-init via ``_sparse_embeddings``
            so we only download/load the model when a real query
            arrives, not on service construction).
        """
        from langchain_qdrant import QdrantVectorStore, RetrievalMode

        if user_id in self._vectorstore_cache:
            return self._vectorstore_cache[user_id]

        safe_user_id = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        model_suffix = self._get_collection_suffix()
        collection_name = f"user_{safe_user_id}_{model_suffix}"

        client = self._get_qdrant_client()
        self._ensure_qdrant_collection(client, collection_name)

        sparse = self._get_sparse_embeddings()
        if sparse is not None:
            mode = RetrievalMode.HYBRID
            mode_label = "HYBRID"
        else:
            mode = RetrievalMode.DENSE
            mode_label = "DENSE (BM25 unavailable)"

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings,
            sparse_embedding=sparse,
            retrieval_mode=mode,
            vector_name="dense",
            sparse_vector_name="sparse",
        )
        logger.info(
            f"Created Qdrant {mode_label} vectorstore for user: {user_id} "
            f"(collection={collection_name})"
        )
        self._vectorstore_cache[user_id] = vectorstore
        return vectorstore


    def _get_kb_collection(self):
        """Get or create a Qdrant-backed VectorStore for the permanent
        knowledge base (``kb_papers`` collection).

        Mirrors ``_get_user_collection`` but targets the fixed KB
        collection created by ``scripts/ingest.py``. Cached on the
        service instance so repeated queries don't re-create the
        wrapper. Returns ``None`` when the collection doesn't exist
        (no papers ingested yet).

        Reads the embedding model name from ``kb.sqlite`` config table
        so the backend always matches whatever ingest.py used — no
        hardcoded model names.
        """
        from langchain_qdrant import QdrantVectorStore, RetrievalMode

        if type(self)._kb_vectorstore_cache is not None:
            return type(self)._kb_vectorstore_cache

        client = self._get_qdrant_client()
        try:
            client.get_collection(self.KB_COLLECTION_NAME)
        except Exception:
            return None

        kb_model = self._read_kb_config("embedding_model") or self._KB_DEFAULT_EMBEDDING_MODEL
        kb_embeddings = BloomIndexEmbeddings(model=kb_model)

        sparse = self._get_sparse_embeddings()
        if sparse is not None:
            mode = RetrievalMode.HYBRID
            mode_label = "HYBRID"
        else:
            mode = RetrievalMode.DENSE
            mode_label = "DENSE (BM25 unavailable)"

        vectorstore = QdrantVectorStore(
            client=client,
            collection_name=self.KB_COLLECTION_NAME,
            embedding=kb_embeddings,
            sparse_embedding=sparse,
            retrieval_mode=mode,
            vector_name="dense",
            sparse_vector_name="sparse",
            content_payload_key="page_content",
            metadata_payload_key="metadata",
        )
        logger.info(
            f"Created Qdrant {mode_label} vectorstore for KB "
            f"(collection={self.KB_COLLECTION_NAME}, "
            f"embedding={kb_model})"
        )
        type(self)._kb_vectorstore_cache = vectorstore
        return vectorstore


    def _invalidate_kb_collection(self) -> None:
        """Drop the cached KB vectorstore wrapper."""
        type(self)._kb_vectorstore_cache = None


    def _read_kb_config(self, key: str) -> Optional[str]:
        """Read a value from the ``config`` table in ``kb.sqlite``.

        Returns ``None`` when the DB file is missing, the table doesn't
        exist yet, or the key isn't present.
        """
        import sqlite3
        from backend.src.common.paths import kb_dir

        kb_path = os.fspath(kb_dir() / "kb.sqlite")
        if not os.path.exists(kb_path):
            return None
        try:
            conn = sqlite3.connect(kb_path, timeout=2)
            row = conn.execute(
                "select value from config where key = ?", (key,)
            ).fetchone()
            conn.close()
            return row[0] if row else None
        except Exception:
            return None


    def _get_sparse_embeddings(self):
        """Lazy-init FastEmbed's ``Qdrant/bm25`` sparse encoder.

        Cached on the service instance so the model is loaded once
        per process. First call downloads ~50MB of BM25 tokenizer
        assets to FastEmbed's cache dir (typically
        ``~/.cache/fastembed/`` on Linux/Mac, ``%LOCALAPPDATA%/fastembed/``
        on Windows). Subsequent calls return the cached instance.

        Kept lazy so service construction stays cheap and the
        download cost is only paid the first time a user actually
        triggers a hybrid ingest or query.

        Returns ``None`` when the BM25 runtime is broken (e.g.
        ``py_rust_stemmers`` segfaults on Python 3.14+).  Callers
        must handle ``None`` by falling back to dense-only mode.
        """
        if not _check_bm25():
            return None
        if not hasattr(self, "_sparse_embeddings_cache") or self._sparse_embeddings_cache is None:
            from langchain_qdrant import FastEmbedSparse
            logger.info("Loading FastEmbed Qdrant/bm25 sparse encoder…")
            self._sparse_embeddings_cache = FastEmbedSparse(model_name="Qdrant/bm25")
            logger.info("FastEmbed Qdrant/bm25 loaded.")
        return self._sparse_embeddings_cache


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


    def _get_parent_store_path(self, user_id: str) -> str:
        from backend.src.common.uploads import get_parent_store_path

        return os.fspath(get_parent_store_path(user_id))


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
          5. **Native sparse vectors** — ``add_documents`` writes
             both dense and BM25 sparse vectors per point (via
             langchain-qdrant's ``RetrievalMode.HYBRID``). No
             post-upload index build step; sparse vectors are
             ready for query immediately.

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


    def _get_kb_parent_data(self, parent_id: str) -> Dict[str, Any]:
        """Read parent data from the KB's ``kb.sqlite`` parents table.

        Returns the same dict shape as ``_get_parent_data`` so the
        citation pipeline works unchanged.
        """
        import sqlite3

        kb_path = os.fspath(kb_dir() / "kb.sqlite")
        if not os.path.exists(kb_path):
            return {}
        try:
            conn = sqlite3.connect(kb_path)
            row = conn.execute(
                "SELECT text, section_title, body_start, body_end, page "
                "FROM parents WHERE parent_id = ?",
                (parent_id,),
            ).fetchone()
            conn.close()
            if row is None:
                return {}
            return {
                "text": row[0] or "",
                "section_title": row[1] or "",
                "body_start": row[2],
                "body_end": row[3],
                "page": row[4],
            }
        except Exception:
            return {}


    @staticmethod
    def _find_page_for_offset(full_text: str, offset: int) -> Optional[int]:
        """Recover the 1-based page number for a char offset inside
        ``full_text`` by scanning ``<!-- Page N -->`` markers (the
        per-page boundaries the pymupdf extractor emits).

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
            import pymupdf

            doc = pymupdf.open(pdf_path)

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


    def _extract_with_pymupdf(self, pdf_path):
        """Fast PDF extraction using plain PyMuPDF (no OCR, no layout model).

        Thin wrapper over ``backend.core.rag_storage.extract_paper_markdown``
        (shared with the preview regen in api/rag so highlight offsets
        always agree). Returns (full_text, tables); ``tables`` stays []
        for API compat — tables are embedded inline in ``full_text``.
        """
        try:
            full_text = extract_paper_markdown(pdf_path)
            if not full_text.strip():
                logger.warning(f"pymupdf extracted empty text from {pdf_path}")
                return None, []
            return full_text, []
        except Exception as e:
            logger.warning(f"pymupdf extraction failed for {pdf_path}: {e}")
            return None, []


    def _detect_sections(self, text):
        """Detect scientific paper sections from plain text or markdown.

        Handles both Docling markdown output (# headers) and PyMuPDF plain text
        by recognizing standard scientific section names.
        """
        # Top-level section names. Subsections are detected generically
        # below (short Title Case line, no period, isolated) so pymupdf
        # — which has no markdown headings — still splits "Soil
        # preparation / Leaching experiment" without hardcoding every
        # possible subsection name. Docling already has `##` headings
        # and doesn't need this heuristic.
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
                # 5. Generic subsection for pymupdf (no markdown).
                # Any short line (2-5 words, <45 chars, no period/colon,
                # starts uppercase) followed by a real paragraph.
                # Catches "Soil preparation", "Grass & Herb Coverage"
                # without a hardcoded name list. Docling has `##`
                # headings and never hits this branch.
                elif (
                    len(stripped) < 45
                    and 2 <= len(stripped.split()) <= 5
                    and not stripped.endswith(".")
                    and not stripped.endswith(":")
                    and stripped[0].isupper()
                    # Generic running-header filter: pagination, postal
                    # codes, emails, and author-line "et al." — no
                    # journal/university/country name list.
                    and not re.search(r"Page \d+ of|\d{5,}|@|et al\.", stripped)
                ):
                    nxt = ""
                    for k in range(i + 1, min(len(lines), i + 4)):
                        if lines[k].strip():
                            nxt = lines[k].strip()
                            break
                    if nxt and len(nxt) > 40 and not nxt.startswith("#"):
                        is_header = True
                        header_level = 2
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


def _format_phase_timings(timings_ms: Dict[str, float]) -> str:
    ordered = []
    for key, value in timings_ms.items():
        ordered.append(f"{key}={value:.1f}ms")
    return ", ".join(ordered)
