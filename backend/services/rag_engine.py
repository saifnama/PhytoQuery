import os
import re
import hashlib
import httpx
import json
import logging
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

    async def invoke(self, prompt: str):
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
        try:
            client = await HttpClientManager.get_client()
            kwargs = {"json": payload, "timeout": 120.0}
            if self.provider == "openrouter":
                kwargs["headers"] = headers
            response = await client.post(self.url, **kwargs)
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
            import traceback

            err_msg = f"RAG LLM Request URL: {self.url}\nPayload: {payload}\nError: {e}\n{traceback.format_exc()}"
            with open("rag_error.log", "w") as f:
                f.write(err_msg)
            logger.error(f"Error calling {self.provider} LLM for RAG: {e}")
            raise


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

    def process_and_index_pdfs(self, pdf_paths: List[str]):
        """Extract, chunk, and index PDFs."""
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
        tables = []

        # 1. Primary: Try Docling
        full_text, tables = self._extract_with_docling(pdf_path)

        # 2. Fallbacks
        if not full_text:
            full_text = self._extract_with_pymupdf4llm(pdf_path)

        if not tables:
            tables = self._extract_tables_pdfplumber(pdf_path)

        if not full_text:
            return []

        # 3. Section detection & Chunking
        sections = self._detect_sections(full_text)
        chunks = self._chunk_by_sections(sections, tables)

        # 4. Deduplication
        unique_chunks = self._deduplicate_chunks(chunks)

        documents = []
        for i, chunk in enumerate(unique_chunks):
            meta = chunk.get("metadata", {})
            meta["source"] = source
            meta["chunk_id"] = f"{source}_{i}"
            documents.append(Document(page_content=chunk["text"], metadata=meta))

        return documents

    async def query(self, question: str) -> Dict[str, Any]:
        """Query the RAG pipeline."""
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

            sources.append({"source": src, "section": sec})

        context = "\n\n".join(context_parts)
        prompt = f"""Answer using ONLY the context below. Use markdown.

Context:
{context}

Question: {question}

Answer:"""

        response = await self.llm.invoke(prompt)
        return {"answer": response.content.strip(), "sources": sources}

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

    def _extract_with_pymupdf4llm(self, pdf_path):
        try:
            import pymupdf4llm

            pages = pymupdf4llm.to_markdown(
                pdf_path, page_chunks=True, force_text=True, show_progress=False
            )
            return "\n\n".join([p.get("text", "") for p in pages])
        except:
            return None

    def _extract_tables_pdfplumber(self, pdf_path):
        try:
            import pdfplumber

            tables = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    for t in page.extract_tables():
                        if t and len(t) > 1:
                            md = (
                                "| " + " | ".join([str(c or "") for c in t[0]]) + " |\n"
                            )
                            md += "| " + " | ".join(["---"] * len(t[0])) + " |\n"
                            for row in t[1:]:
                                md += (
                                    "| "
                                    + " | ".join([str(c or "") for c in row])
                                    + " |\n"
                                )
                            tables.append({"content": md, "page": page_num})
            return tables
        except:
            return []

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
