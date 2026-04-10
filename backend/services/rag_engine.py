import os
import re
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

    async def invoke(self, prompt: str, max_retries: int = 3, base_delay: float = 2.0):
        headers = {}
        if self.provider == "ollama":
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": self.temperature, "num_ctx": self.num_ctx},
            }
        else:  # openrouter
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
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
        self.vectorstore = Chroma(
            persist_directory=config.chroma_dir,
            embedding_function=self.embeddings,
            collection_name="research_docs",
        )
        self.llm = OllamaLLM(
            base_url=LLM_PROVIDER.get("url", "").replace("/api/chat", ""),
            model=LLM_PROVIDER["model"],
            temperature=LLM_TEMPERATURE,
            num_ctx=LLM_CONTEXT_WINDOW,
            provider=LLM_PROVIDER["provider"],
            api_key=LLM_PROVIDER.get("api_key"),
        )
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": config.top_k}
        )

    def process_and_index_pdfs(
        self, pdf_paths: List[str], parser_type: str = "docling"
    ):
        """Extract, chunk, and index PDFs.

        Args:
            pdf_paths: List of PDF file paths to process
            parser_type: "pymupdf" for fast extraction, "docling" for detailed (default: "docling")
        """
        # Store parser type for use in _process_pdf
        self._current_parser = (
            parser_type if parser_type in ("pymupdf", "docling") else "docling"
        )

        all_docs = []
        for path in pdf_paths:
            docs = self._process_pdf(path)
            all_docs.extend(docs)

        if all_docs:
            self.vectorstore.add_documents(all_docs)
        return [os.path.basename(p) for p in pdf_paths]

    def _process_pdf(self, pdf_path: str) -> List[Document]:
        """PDF processing pipeline (Preserved from notebook)"""
        source = os.path.basename(pdf_path)

        # Get parser type from instance or default to docling
        parser_type = getattr(self, "_current_parser", "docling")

        # 1. Extract using selected parser
        if parser_type == "pymupdf":
            full_text, tables = self._extract_with_pymupdf(pdf_path)
        else:
            full_text, tables = self._extract_with_docling(pdf_path)

        if not full_text:
            return []

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
            documents.append(Document(page_content=chunk["text"], metadata=meta))

        return documents

    async def query(
        self, question: str, filter_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Query the RAG pipeline with optional source filtering.

        Args:
            question: The user's natural language question.
            filter_files: If provided, only search chunks from these source filenames.
        """
        # Use filtered search if specific files are selected
        if filter_files:
            docs = self.vectorstore.similarity_search(
                question,
                k=config.top_k,
                filter={"source": {"$in": filter_files}},
            )
        else:
            docs = self.retriever.invoke(question)

        context_parts = []
        sources = []
        for d in docs:
            ctype = d.metadata.get("content_type", "text")
            src = d.metadata.get("source", "")
            sec = d.metadata.get("section_title", "")

            if ctype == "table":
                context_parts.append(f"[TABLE from {src}]:\n{d.page_content}")
            else:
                header = f"[{src}" + (f" - {sec}" if sec else "") + "]:"
                context_parts.append(f"{header}\n{d.page_content}")

            parser_type = d.metadata.get("parser_type", "docling")
            sources.append({"source": src, "section": sec, "parser_type": parser_type})

        context = "\n\n".join(context_parts)
        prompt = f"""Answer using ONLY the context below. Use markdown.

Context:
{context}

Question: {question}

Answer:"""

        response = await self.llm.invoke(prompt)
        return {"answer": response.content.strip(), "sources": sources}

    def list_indexed_files(self) -> List[Dict[str, Any]]:
        """Query ChromaDB for all unique indexed source files and their metadata."""
        try:
            collection = self.vectorstore._collection
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
                        "file_type": meta.get("file_type", os.path.splitext(src)[1] or ".pdf"),
                        "chunk_count": 0,
                        "indexed_at": meta.get("indexed_at", ""),
                        "parser_type": meta.get("parser_type", "docling"),
                    }
                file_map[src]["chunk_count"] += 1

            return list(file_map.values())
        except Exception as e:
            logger.error(f"Error listing indexed files: {e}")
            return []

    def delete_source(self, filename: str) -> bool:
        """Remove a source completely: delete its chunks from ChromaDB and the PDF from disk."""
        try:
            # 1. Find all chunk IDs belonging to this source
            collection = self.vectorstore._collection
            result = collection.get(
                where={"source": filename},
                include=["metadatas"],
            )
            ids_to_delete = result.get("ids", [])

            if ids_to_delete:
                collection.delete(ids=ids_to_delete)
                logger.info(f"Deleted {len(ids_to_delete)} chunks for '{filename}' from ChromaDB")

            # 2. Delete the PDF file from uploads directory
            upload_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "uploads",
            )
            file_path = os.path.join(upload_dir, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted file: {file_path}")

            return True
        except Exception as e:
            logger.error(f"Error deleting source '{filename}': {e}")
            return False

    def reset_rag(self) -> bool:
        """Permanently delete all indexed chunks from ChromaDB and all uploaded PDFs."""
        try:
            # 1. Clear ChromaDB
            # We can retrieve all IDs and delete them
            collection = self.vectorstore._collection
            result = collection.get(include=["metadatas"])
            all_ids = result.get("ids", [])
            
            if all_ids:
                collection.delete(ids=all_ids)
                logger.info(f"Deleted all {len(all_ids)} chunks from ChromaDB")

            # 2. Clear Uploads directory
            upload_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "data", "uploads",
            )
            if os.path.exists(upload_dir):
                for filename in os.listdir(upload_dir):
                    file_path = os.path.join(upload_dir, filename)
                    try:
                        if os.path.isfile(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logger.error(f"Failed to delete {file_path}: {e}")
                logger.info("Cleared all files from uploads directory.")

            return True
        except Exception as e:
            logger.error(f"Error resetting RAG data: {e}")
            return False

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
        splitter = RecursiveCharacterTextSplitter(
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
                for i, sub in enumerate(splitter.split_text(text)):
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
