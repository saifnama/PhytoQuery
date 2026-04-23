import os
import re
import gc
import time
import hashlib
import httpx
import json
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
from backend.core.http_client import HttpClientManager

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

# --- Import from RAG config ---
from backend.config_rag import (
    RAG_OLLAMA_URL,
    RAG_OLLAMA_MODEL,
    RAG_OPENROUTER_API_KEY,
    RAG_OPENROUTER_MODEL,
    RAG_TEMPERATURE,
    RAG_CONTEXT_WINDOW,
    RAG_EMBEDDING_MODEL,
    RAG_TOP_K,
    RAG_SIMILARITY_THRESHOLD,
    get_rag_provider,
)


# --- RAG Configuration ---
LLM_PROVIDER = get_rag_provider()
LLM_TEMPERATURE = RAG_TEMPERATURE
LLM_CONTEXT_WINDOW = RAG_CONTEXT_WINDOW


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 100
    min_chunk_size: int = 100
    similarity_threshold: float = RAG_SIMILARITY_THRESHOLD
    embedding_model: str = RAG_EMBEDDING_MODEL
    top_k: int = RAG_TOP_K
    parser_type: str = "pymupdf"  # "pymupdf" (fast) or "docling" (detailed)
    chroma_dir: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        "chroma_db",
    )


config = RAGConfig()

logger = logging.getLogger(__name__)


class RAGProviderAuthError(Exception):
    """Raised when the configured LLM provider rejects authentication/config."""


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
                "RAG is not configured. Set a valid RAG_OPENROUTER_API_KEY or configure RAG_OLLAMA_URL."
            )

        headers = {}
        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": msg_list,
                "stream": False,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            }
        else:  # openrouter
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
                if self.provider == "openrouter":
                    kwargs["headers"] = headers
                response = await client.post(self.url, **kwargs)

                if self.provider == "openrouter" and response.status_code == 401:
                    raise RAGProviderAuthError(
                        "OpenRouter authentication failed. Check RAG_OPENROUTER_API_KEY or configure RAG_OLLAMA_URL as a fallback provider."
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
                else:
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
                logger.error(f"Error calling {self.provider} LLM for RAG: {e}")
                raise

        # All retries exhausted
        import traceback

        err_msg = f"RAG LLM Request URL: {self.url}\nPayload: {payload}\nError: {last_exception}\n{traceback.format_exc()}"
        with open("rag_error.log", "w") as f:
            f.write(err_msg)
        logger.error(f"All retries failed for RAG LLM: {last_exception}")
        raise last_exception


class RAGService:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=config.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        try:
            logger.info("Loading CrossEncoder for reranking...")
            self.reranker = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512, device="cpu"
            )
        except Exception as e:
            logger.warning(f"Failed to load CrossEncoder: {e}")
            self.reranker = None

        self.llm = OllamaLLM(
            base_url=LLM_PROVIDER.get("url", "").replace("/api/chat", ""),
            model=LLM_PROVIDER["model"],
            temperature=LLM_TEMPERATURE,
            num_ctx=LLM_CONTEXT_WINDOW,
            provider=LLM_PROVIDER["provider"],
            api_key=LLM_PROVIDER.get("api_key"),
        )
        # Cache for per-user vectorstores
        self._vectorstore_cache: Dict[str, Chroma] = {}

    def _get_user_collection(self, user_id: str) -> Chroma:
        """Get or create a ChromaDB collection for a specific user."""
        if user_id in self._vectorstore_cache:
            return self._vectorstore_cache[user_id]

        # Create user-specific collection
        # Sanitize user_id for collection name (alphanumeric + underscore only)
        safe_user_id = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        collection_name = f"user_{safe_user_id}"

        # User-specific persist directory
        user_chroma_dir = os.path.join(config.chroma_dir, safe_user_id)

        # Retry logic for handling locked files on Windows
        max_retries = 3
        for attempt in range(max_retries):
            try:
                os.makedirs(user_chroma_dir, exist_ok=True)
                vectorstore = Chroma(
                    persist_directory=user_chroma_dir,
                    embedding_function=self.embeddings,
                    collection_name=collection_name,
                )
                break
            except Exception as e:
                if attempt < max_retries - 1 and "locked" in str(e).lower():
                    logger.warning(
                        f"Chroma folder locked for {user_id}, retry {attempt + 1}/{max_retries}"
                    )
                    time.sleep(0.3)
                    continue
                raise

        self._vectorstore_cache[user_id] = vectorstore
        logger.info(f"Created ChromaDB collection for user: {user_id}")
        return vectorstore

    def _invalidate_user_collection(self, user_id: str) -> None:
        """Drop a cached per-user Chroma client so the next access reopens it.

        This is important after delete/reset/cleanup operations because SQLite/
        Chroma handles can remain stale or locked across requests.
        """
        vectorstore = self._vectorstore_cache.pop(user_id, None)
        if vectorstore is not None:
            vectorstore = None
        gc.collect()
        time.sleep(0.1)

    def process_and_index_pdfs(
        self,
        pdf_paths: List[str],
        parser_type: str = "docling",
        user_id: str = "default",
    ):
        """Extract, chunk, and index PDFs for a specific user.

        Args:
            pdf_paths: List of PDF file paths to process
            parser_type: "pymupdf" for fast extraction, "docling" for detailed
            user_id: Unique identifier for the user (isolates their documents)
        """
        # Store parser type for use in _process_pdf
        self._current_parser = (
            parser_type if parser_type in ("pymupdf", "docling") else "docling"
        )

        all_docs = []
        for path in pdf_paths:
            docs = self._process_pdf(path, user_id=user_id)
            all_docs.extend(docs)

        if all_docs:
            try:
                vectorstore = self._get_user_collection(user_id)
                vectorstore.add_documents(all_docs)
            except Exception:
                logger.warning(
                    f"Indexing failed for user {user_id}; invalidating cached Chroma client and retrying once."
                )
                self._invalidate_user_collection(user_id)
                time.sleep(0.3)  # Wait for SQLite file handles to release on Windows
                vectorstore = self._get_user_collection(user_id)
                try:
                    vectorstore.add_documents(all_docs)
                except Exception as retry_err:
                    logger.error(
                        f"Retry failed even after cache invalidation for {user_id}: {retry_err}"
                    )
                    raise retry_err from retry_err
        return [os.path.basename(p) for p in pdf_paths]

    def _extract_pdf_metadata(self, pdf_path: str) -> Dict[str, str]:
        """Extract DOI, authors, and journal from PDF metadata and first pages."""
        metadata = {"authors": "", "doi": "", "journal": ""}
        try:
            import fitz

            doc = fitz.open(pdf_path)

            # Try PDF document info first
            pdf_info = doc.metadata or {}
            if pdf_info.get("author"):
                metadata["authors"] = pdf_info["author"]

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

    def _process_pdf(self, pdf_path: str, user_id: str = "default") -> List[Document]:
        """PDF processing pipeline (Preserved from notebook)"""
        source = os.path.basename(pdf_path)

        # Get parser type from instance or default to docling
        parser_type = getattr(self, "_current_parser", "docling")

        # 0. Extract metadata (DOI, authors, journal)
        pdf_metadata = self._extract_pdf_metadata(pdf_path)

        # 1. Extract using selected parser
        if parser_type == "pymupdf":
            full_text, tables = self._extract_with_pymupdf(pdf_path)
        else:
            full_text, tables = self._extract_with_docling(pdf_path)

        if not full_text:
            return []

        # Store full text for summarization later
        self._last_extracted_text = full_text

        # 2. Section detection & Chunking
        sections = self._detect_sections(full_text)
        chunks = self._chunk_by_sections(sections, tables)

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
            meta["doc_authors"] = pdf_metadata.get("authors", "")
            meta["doc_doi"] = pdf_metadata.get("doi", "")
            meta["doc_journal"] = pdf_metadata.get("journal", "")
            documents.append(Document(page_content=chunk["text"], metadata=meta))

        return documents

    def _hybrid_search(
        self,
        question: str,
        vectorstore: Chroma,
        filter_files: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Hybrid search combining vector similarity + BM25 keyword matching.

        Uses Reciprocal Rank Fusion (RRF) to merge results from both methods.
        Returns list of dicts with 'doc', 'score' keys.
        """
        k = config.top_k
        rrf_k = 60  # RRF constant

        # --- 1. Vector similarity search ---
        search_kwargs = {"k": k}
        if filter_files:
            search_kwargs["filter"] = {"source": {"$in": filter_files}}

        try:
            vector_results = vectorstore.similarity_search_with_relevance_scores(
                question, **search_kwargs
            )
        except Exception:
            # Fallback to regular search if relevance scores not supported
            docs = vectorstore.similarity_search(question, **search_kwargs)
            vector_results = [(doc, 0.5) for doc in docs]

        # --- 2. BM25 keyword search ---
        bm25_results = []
        try:
            from rank_bm25 import BM25Okapi

            # Get all documents from collection (for the selected sources)
            collection = vectorstore._collection
            get_kwargs = {"include": ["documents", "metadatas"]}
            if filter_files:
                get_kwargs["where"] = {"source": {"$in": filter_files}}
            all_data = collection.get(**get_kwargs)

            all_docs_text = all_data.get("documents", [])
            all_metas = all_data.get("metadatas", [])
            all_ids = all_data.get("ids", [])

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

        # 1. Hybrid search (BM25 + Vector with RRF) - Get top 20 candidates
        config.top_k = 20  # Temporary boost for candidate retrieval
        search_results = self._hybrid_search(question, vectorstore, filter_files)
        config.top_k = RAG_TOP_K  # Reset

        # 2. Cross-Encoder Reranking - Pick top 5
        reranked_results = []
        if self.reranker and search_results:
            try:
                pairs = [[question, res["doc"].page_content] for res in search_results]
                rerank_scores = self.reranker.predict(pairs)

                for i, score in enumerate(rerank_scores):
                    search_results[i]["rerank_score"] = float(score)

                # Sort by rerank score descending
                search_results.sort(key=lambda x: x["rerank_score"], reverse=True)

                # Take top k
                reranked_results = search_results[: config.top_k]

                # Normalize cross-encoder scores (typically unbounded, min/max varies, rough sigmoid approximation for display)
                import math

                for r in reranked_results:
                    # Simple sigmoid to squish to 0-100%
                    prob = 1 / (1 + math.exp(-r["rerank_score"]))
                    r["normalized_score"] = round(prob * 100)
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                reranked_results = search_results[: config.top_k]
        else:
            reranked_results = search_results[: config.top_k]

        context_parts = []
        sources = []
        for result in reranked_results:
            d = result["doc"]
            score = result.get("normalized_score", 0)
            ctype = d.metadata.get("content_type", "text")
            src = d.metadata.get("source", "")
            sec = d.metadata.get("section_title", "")

            if ctype == "table":
                context_parts.append(f"[TABLE from {src}]:\n{d.page_content}")
            else:
                header = f"[{src}" + (f" - {sec}" if sec else "") + "]:"
                context_parts.append(f"{header}\n{d.page_content}")

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

        response = await self.llm.invoke(messages=messages)
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
            response = await self.llm.invoke(prompt)
            return response.content.strip()
        except Exception as e:
            logger.warning(f"Summarization failed for {filename}: {e}")
            return ""

    def list_indexed_files(self, user_id: str = "default") -> List[Dict[str, Any]]:
        """Query ChromaDB for all unique indexed source files for a specific user."""
        try:
            vectorstore = self._get_user_collection(user_id)
            collection = vectorstore._collection
            result = collection.get(include=["metadatas"])
            metadatas = result.get("metadatas", [])

            # Aggregate per unique source
            file_map: Dict[str, Dict[str, Any]] = {}
            for meta in metadatas:
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

            return list(file_map.values())

        except Exception as e:
            logger.error(f"Error listing indexed files: {e}")
            return []

    def delete_source(self, filename: str, user_id: str = "default") -> bool:
        """Remove a source completely for a specific user: delete its chunks from ChromaDB."""
        try:
            vectorstore = self._get_user_collection(user_id)
            collection = vectorstore._collection
            result = collection.get(
                where={"source": filename},
                include=["metadatas"],
            )
            ids_to_delete = result.get("ids", [])

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(
                    f"Deleted {len(ids_to_delete)} chunks for '{filename}' from user {user_id}'s ChromaDB"
                )

            self._invalidate_user_collection(user_id)

            return True
        except Exception as e:
            self._invalidate_user_collection(user_id)
            logger.error(f"Error deleting source '{filename}': {e}")
            return False

    def reset_rag(self, user_id: str = "default") -> bool:
        """Permanently delete all indexed chunks from a specific user's ChromaDB."""
        try:
            vectorstore = self._get_user_collection(user_id)
            collection = vectorstore._collection
            result = collection.get(include=["metadatas"])
            all_ids = result.get("ids", [])

            if all_ids:
                collection.delete(ids=all_ids)
                logger.info(f"Deleted all {len(all_ids)} chunks for user {user_id}")

            self._invalidate_user_collection(user_id)

            return True
        except Exception as e:
            self._invalidate_user_collection(user_id)
            logger.error(f"Error resetting RAG data for user {user_id}: {e}")
            return False

    def cleanup_user(self, user_id: str) -> bool:
        """Clean up all data for a user when they close their browser.
        Deletes: ChromaDB collection folder + uploads folder for this user."""
        success = True

        # 0. Empty Chroma collection first (always works even if folder is locked)
        try:
            self.reset_rag(user_id)
        except Exception as e:
            logger.warning(f"Could not empty collection cleanly: {e}")

        # 1. Remove from cache and force garbage collection to release Windows SQLite file locks
        self._invalidate_user_collection(user_id)
        time.sleep(0.4)

        # 2. Delete user's ChromaDB folder
        safe_user_id = re.sub(r"[^a-zA-Z0-9_]", "_", user_id)
        user_chroma_dir = os.path.join(config.chroma_dir, safe_user_id)

        try:
            if os.path.exists(user_chroma_dir):
                import shutil

                shutil.rmtree(user_chroma_dir)
                logger.info(f"Deleted ChromaDB folder: {user_chroma_dir}")
        except Exception as e:
            logger.warning(f"Could not fully delete ChromaDB folder: {e}")
            # Ensure the SQLite DB is at least emptied via reset_rag

        # 3. Delete user's uploads folder
        upload_dir = os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "data",
            "uploads",
            user_id,
        )
        try:
            if os.path.exists(upload_dir):
                import shutil

                shutil.rmtree(upload_dir)
                logger.info(f"Deleted uploads folder: {upload_dir}")
        except Exception as e:
            logger.error(f"Error deleting uploads folder: {e}")
            success = False

        if success:
            logger.info(f"Cleaned up all data for user: {user_id}")
        return success

    # --- Helper Methods (Logic preserved from notebook) ---

    def _extract_with_docling(self, pdf_path):
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.datamodel.base_models import InputFormat

            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_table_structure = True
            pipeline_options.do_ocr = False

            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
            result = converter.convert(pdf_path)
            md_text = result.document.export_to_markdown()

            docling_tables = []
            if hasattr(result.document, "tables"):
                for table in result.document.tables:
                    try:
                        table_md = table.export_to_markdown(doc=result.document)
                        pnum = 0
                        if table.prov and len(table.prov) > 0:
                            pnum = getattr(table.prov[0], "page_no", 0)
                        docling_tables.append({"content": table_md, "page": pnum})
                    except:
                        continue
            return md_text, docling_tables
        except:
            return None, []

    def _extract_with_pymupdf(self, pdf_path):
        """Fast PDF extraction using PyMuPDF (fitz)

        Uses plain text extraction for maximum compatibility.
        Much faster than Docling but without table structure detection.
        """
        try:
            import fitz  # PyMuPDF

            text_parts = []
            tables = []

            doc = fitz.open(pdf_path)
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    text_parts.append(text)

            doc.close()

            full_text = "\n\n".join(text_parts)
            if not full_text.strip():
                logger.warning(f"PyMuPDF extracted empty text from {pdf_path}")
                return None, []
            return full_text, tables
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            return None, []

    def _detect_sections(self, text):
        lines = text.split("\n")
        sections = []
        current = {"title": "Start", "level": 0, "content": [], "start": 0}
        for i, line in enumerate(lines):
            if line.startswith("#"):
                match = re.match(r"^(#{1,4})\s+(.+)$", line)
                if match:
                    if current["content"]:
                        current["text"] = "\n".join(current["content"])
                        sections.append(current)
                    current = {
                        "title": match.group(2).strip(),
                        "level": len(match.group(1)),
                        "content": [],
                        "start": i,
                    }
                    continue
            if re.match(r"^\d+\.?\s+[A-Z]", line):
                if current["content"]:
                    current["text"] = "\n".join(current["content"])
                    sections.append(current)
                current = {"title": line.strip(), "level": 1, "content": [], "start": i}
                continue
            current["content"].append(line)
        if current["content"]:
            current["text"] = "\n".join(current["content"])
            sections.append(current)
        return sections

    def _chunk_by_sections(self, sections, tables):
        try:
            # Try SemanticChunker first
            semantic_splitter = SemanticChunker(
                self.embeddings, breakpoint_threshold_type="percentile"
            )
        except Exception as e:
            logger.warning(
                f"SemanticChunker init failed, falling back to recursive: {e}"
            )
            semantic_splitter = None

        fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = []
        for table in tables:
            content = table.get("content", "")
            if content.strip():
                chunks.append(
                    {
                        "text": content,
                        "metadata": {
                            "content_type": "table",
                            "page": table.get("page", 0),
                        },
                    }
                )
        for section in sections:
            text = section.get("text", "")
            if not text.strip():
                continue
            meta = {"section_title": section.get("title", ""), "content_type": "text"}

            if len(text) <= config.chunk_size:
                if len(text) >= config.min_chunk_size:
                    chunks.append({"text": text, "metadata": meta})
            else:
                if semantic_splitter:
                    try:
                        # Semantic chunking
                        split_docs = semantic_splitter.create_documents([text])
                        for i, doc in enumerate(split_docs):
                            if len(doc.page_content) >= config.min_chunk_size:
                                m = meta.copy()
                                m["chunk_idx"] = i
                                chunks.append({"text": doc.page_content, "metadata": m})
                        continue
                    except Exception as e:
                        logger.warning(
                            f"Semantic chunking failed for section, falling back: {e}"
                        )

                # Fallback to recursive character splitting
                for i, sub in enumerate(fallback_splitter.split_text(text)):
                    if len(sub) >= config.min_chunk_size:
                        m = meta.copy()
                        m["chunk_idx"] = i
                        chunks.append({"text": sub, "metadata": m})
        return chunks

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


# Singleton instance
rag_service = RAGService()
