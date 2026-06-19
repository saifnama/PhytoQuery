#!/usr/bin/env python3
"""
Standalone ingestion pipeline for the permanent paper knowledge base.

Completely separate from the per-user upload pipeline — its own folder,
its own Qdrant collection, its own SQLite state store. Safe to re-run at
any time:
  - Already-indexed papers are skipped in seconds.
  - A revised PDF (same filename, changed bytes) is detected and
    the stale entry is cleaned from everywhere before re-indexing.
  - New PDFs dropped into RAW_DIR are picked up automatically.
  - Failed papers from a previous run are retried.

Usage:
    python ingest.py           # process everything pending
    python ingest.py --status  # print catalog summary, exit

Delete a paper:
    python ingest.py --delete  (interactive)
"""

import argparse
import glob
import hashlib
import logging
import multiprocessing
import os
import re
import sqlite3
import sys
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Ensure the repo root is on sys.path so `backend.*` imports work when run
# from any working directory (e.g. `python scripts/ingest.py`).
# --------------------------------------------------------------------------- #
_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# --------------------------------------------------------------------------- #
# Configuration — edit these to match your setup before running.
# --------------------------------------------------------------------------- #
RAW_DIR = "backend/knowledge_base/papers"  # folder holding the PDFs
MARKDOWN_DIR = "backend/knowledge_base/parsed"  # parsed markdown cache
STATE_DB = "backend/knowledge_base/kb.sqlite"  # papers table + parents table

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "kb_papers"  # bake model+dim into name;
# change if you re-embed with
# a different model later.
DENSE_DIM = 384  # BGE-small-en-v1.5 native dim; change if using a different model
DENSE_VEC = "dense"
SPARSE_VEC = "sparse"
SPARSE_MODEL = "Qdrant/bm25"

# Tokenizer the chunker uses to stay within the model's context window.
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
PARENT_MAX_TOKENS = 500  # ~400-600 is the practical sweet spot
CHILD_CHUNK_CHARS = 900  # characters per child chunk
CHILD_CHUNK_OVERLAP = 120

# GPU settings. One worker process owns the GPU; don't raise this
# above 1 unless you have multiple GPUs — competing for one GPU is slower.
PARSE_WORKERS = 1
GPU_PAGE_BATCH = 8  # pages fed to layout model in one batch;
# tune up if VRAM allows (A100: try 16-32)


def _detect_device() -> str:
    """Return 'cuda', 'mps', or 'cpu' depending on what's available."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# Fixed UUID namespace for deterministic point IDs.
# NEVER change this after first ingestion run — it would generate different
# IDs for already-indexed chunks and cause silent duplicates on re-upsert.
ID_NAMESPACE = uuid.UUID("a4f1b9d2-7e3a-4c8b-9f10-7c5d2e8a1b00")
# --------------------------------------------------------------------------- #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("kb_ingest")

DOI_PATTERN = re.compile(
    r"\b(https?://doi\.org/|doi:\s*)?(10\.\d{4,9}/[^\s\"'<>)]+)",
    re.IGNORECASE,
)
REF_HEADING = re.compile(
    r"^(references?|bibliography|works cited|literature cited|citations?)\s*$",
    re.IGNORECASE,
)

# Worker-process globals — populated once by _init_worker(), never per-PDF.
_converter = None
_chunker = None
_splitter = None


def _init_worker() -> None:
    """Load Docling models once per worker process. Expensive — do not call per PDF."""
    global _converter, _chunker, _splitter

    from docling.chunking import HybridChunker
    from docling.datamodel.accelerator_options import (
        AcceleratorDevice,
        AcceleratorOptions,
    )
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
    from docling.datamodel.settings import settings as docling_settings
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.transforms.chunker.tokenizer.huggingface import (
        HuggingFaceTokenizer,
    )
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from transformers import AutoTokenizer

    pipeline_opts = PdfPipelineOptions()
    pipeline_opts.do_ocr = False  # academic PDFs are seldom scanned
    pipeline_opts.do_table_structure = True
    pipeline_opts.table_structure_options.mode = TableFormerMode.ACCURATE
    pipeline_opts.table_structure_options.do_cell_matching = False

    device = _detect_device()
    log.info("Docling using device: %s", device)

    pipeline_opts.accelerator_options = AcceleratorOptions(
        device=AcceleratorDevice.CUDA if device == "cuda" else AcceleratorDevice.CPU,
        cuda_use_flash_attention2=False,
    )

    if device == "cuda":
        docling_settings.perf.page_batch_size = GPU_PAGE_BATCH
    else:
        docling_settings.perf.page_batch_size = 1

    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)
        }
    )
    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
        max_tokens=PARENT_MAX_TOKENS,
    )
    _chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    _splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_CHARS,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def file_paper_id(path: str) -> str:
    """SHA-256 content hash → 24-char hex. Stable, filename-independent."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:24]


def det_id(*parts: str) -> str:
    """Deterministic UUID5 from arbitrary string parts. Valid Qdrant point ID."""
    return str(uuid.uuid5(ID_NAMESPACE, ":".join(parts)))


def extract_doi_from_pdf(pdf_path: str) -> Optional[str]:
    """Extract DOI from the first two pages of a PDF via pymupdf.

    Docling does *not* extract DOIs — they live in free-text body copy,
    not in structured elements.  Academic PDFs almost always print the
    DOI on page 1 (often as ``http://dx.doi.org/10.xxxx/…``).
    """
    try:
        import pymupdf as _fitz

        doc = _fitz.open(pdf_path)
        for i in range(min(2, len(doc))):
            page_text = doc[i].get_text()
            m = DOI_PATTERN.search(page_text)
            if m:
                doc.close()
                return m.group(2).rstrip(".,;)")
        doc.close()
    except Exception:
        pass
    return None


def extract_doi_from_markdown(text: str) -> Optional[str]:
    """Fallback DOI from the first 3,000 characters of markdown."""
    m = DOI_PATTERN.search(text[:3000])
    if not m:
        return None
    return m.group(2).rstrip(".,;)")


def extract_title(doc, full_md: str, fallback: str) -> str:
    """Extract the paper title from the Docling document.

    Universal strategy — works for ANY paper without per-paper hardcoding:

    1.  ``doc.name`` — set by Docling when the PDF has an embedded title.
        Skipped when it looks like a filename (hyphens/dots, no spaces).
    2.  Walk ``doc.iterate_items()`` and collect all ``section_header``
        items that appear BEFORE any body text item.  The title is the
        LONGEST such heading — journal section labels ("Special Report",
        "Original Research", "Review") are always short, while paper
        titles are always long (describing the topic).
    3.  Markdown fallback: longest H1/H2 in first 80 lines.
    4.  Filename stem (last resort).
    """
    from docling_core.types.doc import DocItemLabel

    # --- 1. Embedded title from PDF metadata ---
    name = getattr(doc, "name", None)
    if name and " " in name and not re.search(r"[-_.]{2,}", name):
        return name

    # --- 2. Walk document structure ---
    # Collect all section_headers before body text, then pick the longest.
    seen_body_text = False
    candidates = []  # (length, text) for each section_header before body

    for item, level in doc.iterate_items():
        label = getattr(item, "label", None)
        text = getattr(item, "text", "").strip()

        # Track whether we've hit real body content
        if label in (
            DocItemLabel.TEXT,
            DocItemLabel.PARAGRAPH,
            DocItemLabel.FOOTNOTE,
            DocItemLabel.LIST_ITEM,
        ):
            seen_body_text = True

        # Only consider section headers
        if label != DocItemLabel.SECTION_HEADER:
            continue

        # Once we've seen body text, stop — title is always before body
        if seen_body_text:
            break

        # Skip very short headings (< 10 chars — journal labels, "Keywords:", etc.)
        if len(text) < 10:
            continue

        candidates.append((len(text), text))

    if candidates:
        # The title is the longest heading before body text
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    # --- 3. Markdown fallback ---
    best = ""
    for line in full_md.splitlines()[:80]:
        stripped = line.strip()
        if stripped.startswith("## "):
            heading = stripped[3:].strip()
        elif stripped.startswith("# "):
            heading = stripped[2:].strip()
        else:
            continue
        if len(heading) > len(best):
            best = heading

    return best if best else fallback


# --------------------------------------------------------------------------- #
# Stage A — runs inside a worker process (no embedding model, no Qdrant, no
# SQLite here). Returns plain picklable dicts back to the main process.
# --------------------------------------------------------------------------- #


def _normalized_find(haystack: str, needle: str, start: int = 0) -> int:
    """Find *needle* in *haystack* after collapsing runs of whitespace.

    The HybridChunker sometimes emits text with slightly different
    whitespace (extra newlines, collapsed spaces) compared to
    ``export_to_markdown()``.  A plain ``str.find()`` therefore misses
    many chunks.  This helper normalises both sides to single spaces
    before searching and returns the character offset in the *original*
    (un-normalised) haystack so that ``body_start`` / ``body_end`` stay
    useful for passage highlighting.
    """
    _ws = re.compile(r"\s+")
    norm_hay = _ws.sub(" ", haystack)
    norm_need = _ws.sub(" ", needle).strip()
    pos = norm_hay.find(norm_need, start)
    if pos == -1:
        return -1
    # Map back to original offset: walk the original string counting
    # characters that were consumed by the normalised version.
    orig_pos = 0
    norm_pos = 0
    while norm_pos < pos and orig_pos < len(haystack):
        if haystack[orig_pos].isspace():
            # Skip the entire whitespace run in both strings.
            while orig_pos < len(haystack) and haystack[orig_pos].isspace():
                orig_pos += 1
            norm_pos += 1  # normalised version has exactly one space
        else:
            orig_pos += 1
            norm_pos += 1
    return orig_pos


def _find_chunk_offset(haystack: str, chunk_text: str, cursor: int = 0) -> int:
    """Find the character offset of *chunk_text* in *haystack*.

    Universal approach — progressive anchor search:
    1.  Try the full chunk text (with whitespace normalization).
    2.  If that fails, try the first sentence (up to first '.').
    3.  If that fails, try the first N words (first 80 chars).
    4.  If that fails, try the first 50 chars.
    5.  If all fail, return -1.

    This handles cases where the HybridChunker emits text with different
    whitespace, line breaks, or minor formatting differences from
    ``export_to_markdown()``.
    """
    # Try full chunk text first
    pos = _normalized_find(haystack, chunk_text, cursor)
    if pos != -1:
        return pos

    # Try from the beginning (in case cursor skipped past it)
    pos = _normalized_find(haystack, chunk_text)
    if pos != -1:
        return pos

    # Progressive anchor: try shorter prefixes
    for anchor_len in (200, 100, 80, 50):
        if len(chunk_text) <= anchor_len:
            continue
        anchor = chunk_text[:anchor_len]
        pos = _normalized_find(haystack, anchor, cursor)
        if pos != -1:
            return pos
        pos = _normalized_find(haystack, anchor)
        if pos != -1:
            return pos

    # Last resort: first sentence (up to first period)
    period_idx = chunk_text.find(".")
    if period_idx > 20:
        anchor = chunk_text[:period_idx]
        pos = _normalized_find(haystack, anchor, cursor)
        if pos != -1:
            return pos
        pos = _normalized_find(haystack, anchor)
        if pos != -1:
            return pos

    return -1


def parse_and_chunk(path: str, paper_id: str) -> dict:
    result = _converter.convert(path)
    doc = result.document
    full_md = doc.export_to_markdown()

    title = extract_title(doc, full_md, Path(path).stem)
    doi = extract_doi_from_pdf(path) or extract_doi_from_markdown(full_md)

    parent_rows = []
    pending_points = []
    child_idx = 0
    cursor = 0

    for p_idx, chunk in enumerate(_chunker.chunk(dl_doc=doc)):
        headings = chunk.meta.headings or []
        section = headings[-1] if headings else ""

        # Skip bibliography / references section — citation-formatted text
        # is noise in a content-retrieval system.
        if any(REF_HEADING.match(h.strip()) for h in headings):
            continue

        page = None
        if chunk.meta.doc_items and chunk.meta.doc_items[0].prov:
            page = chunk.meta.doc_items[0].prov[0].page_no

        start = _find_chunk_offset(full_md, chunk.text, cursor)
        end = (start + len(chunk.text)) if start != -1 else -1
        if start != -1:
            cursor = end

        parent_id = det_id(paper_id, "parent", str(p_idx))
        contextualized = _chunker.contextualize(chunk=chunk)

        parent_rows.append(
            (parent_id, paper_id, contextualized, section, start, end, page)
        )

        for c_text in _splitter.split_text(contextualized):
            pending_points.append(
                {
                    "id": det_id(paper_id, "child", str(child_idx)),
                    "text": c_text,
                    "parent_id": parent_id,
                    "section": section,
                    "page": page,
                }
            )
            child_idx += 1

    return {
        "paper_id": paper_id,
        "path": path,
        "title": title,
        "doi": doi,
        "full_md": full_md,
        "parent_rows": parent_rows,
        "pending_points": pending_points,
    }


# --------------------------------------------------------------------------- #
# Stage B — runs in the main process only. Embedding model, Qdrant client,
# and SQLite connection never cross into the worker processes.
# --------------------------------------------------------------------------- #


def embed_and_store(parsed: dict, embeddings, qdrant, conn: sqlite3.Connection) -> int:
    from qdrant_client import models

    paper_id = parsed["paper_id"]
    path = parsed["path"]
    title = parsed["title"]
    doi = parsed["doi"]

    # Save parsed markdown — citation rendering and future passage highlighting
    # depend on this. Parsing never needs to repeat if chunking changes later.
    Path(MARKDOWN_DIR).mkdir(parents=True, exist_ok=True)
    with open(f"{MARKDOWN_DIR}/{paper_id}.md", "w", encoding="utf-8") as f:
        f.write(parsed["full_md"])

    # Warn on DOI collision (duplicate or preprint/published pair).
    # Don't auto-merge — let the operator decide which copy to keep.
    if doi:
        clash = conn.execute(
            "select paper_id, filename from papers where doi = ? and paper_id != ?",
            (doi, paper_id),
        ).fetchone()
        if clash:
            log.warning(
                "DOI collision: %s shares doi=%s with already-indexed %s (%s) "
                "— possible duplicate. Keep one and run --delete on the other.",
                Path(path).name,
                doi,
                clash[1],
                clash[0],
            )

    pts = parsed["pending_points"]
    if not pts:
        log.warning(
            "%s produced 0 child chunks — check parsing output.", Path(path).name
        )

    texts = [p["text"] for p in pts]
    dense_vectors = embeddings.embed_documents(texts) if texts else []

    qdrant_points = [
        models.PointStruct(
            id=p["id"],
            vector={
                DENSE_VEC: dv,
                SPARSE_VEC: models.Document(text=p["text"], model=SPARSE_MODEL),
            },
            payload={
                "page_content": p["text"],
                "metadata": {
                    "paper_id": paper_id,
                    "parent_id": p["parent_id"],
                    "source": Path(path).name,
                    "doc_title": title,
                    "doc_doi": doi,
                    "section_title": p["section"],
                    "page": p["page"],
                    "content_type": "text",
                },
            },
        )
        for p, dv in zip(pts, dense_vectors)
    ]

    if qdrant_points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=qdrant_points)

    conn.executemany(
        "insert or replace into parents "
        "(parent_id, paper_id, text, section_title, body_start, body_end, page) "
        "values (?,?,?,?,?,?,?)",
        parsed["parent_rows"],
    )
    conn.execute(
        "insert or replace into papers "
        "(paper_id, filename, title, doi, status, num_chunks, error, indexed_at) "
        "values (?,?,?,?,'indexed',?,NULL,?)",
        (paper_id, Path(path).name, title, doi, len(qdrant_points), time.time()),
    )
    conn.commit()
    return len(qdrant_points)


# --------------------------------------------------------------------------- #
# Qdrant collection management
# --------------------------------------------------------------------------- #


def ensure_collection(qdrant) -> None:
    from qdrant_client import models

    if qdrant.collection_exists(COLLECTION_NAME):
        return
    log.info("Creating collection %s …", COLLECTION_NAME)
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            DENSE_VEC: models.VectorParams(
                size=DENSE_DIM,
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            # IDF modifier is mandatory for Qdrant/bm25 — without it the sparse
            # side does not apply BM25 scoring at all.
            SPARSE_VEC: models.SparseVectorParams(modifier=models.Modifier.IDF),
        },
        # Disable HNSW during bulk load; re-enable in finalize_collection().
        # Building the graph incrementally on every batch wastes hours of work.
        hnsw_config=models.HnswConfigDiff(m=0),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0),
    )
    qdrant.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="paper_id",
        field_schema="keyword",
    )
    log.info("Collection created.")


def finalize_collection(qdrant) -> None:
    from qdrant_client import models

    log.info("Re-enabling HNSW index build …")
    qdrant.update_collection(
        collection_name=COLLECTION_NAME,
        hnsw_config=models.HnswConfigDiff(m=16),
        optimizers_config=models.OptimizersConfigDiff(indexing_threshold=20000),
    )


# --------------------------------------------------------------------------- #
# SQLite state store
# --------------------------------------------------------------------------- #


def init_db() -> sqlite3.Connection:
    Path(STATE_DB).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(STATE_DB)
    conn.execute("pragma journal_mode=WAL")
    conn.execute("""
        create table if not exists papers (
            paper_id   text primary key,
            filename   text not null,
            title      text,
            doi        text,
            status     text not null default 'pending',
            num_chunks integer,
            error      text,
            indexed_at real
        )
    """)
    conn.execute("""
        create table if not exists parents (
            parent_id     text primary key,
            paper_id      text not null,
            text          text not null,
            section_title text,
            body_start    integer,
            body_end      integer,
            page          integer
        )
    """)
    conn.execute("create index if not exists idx_parents_paper on parents(paper_id)")
    conn.execute("""
        create table if not exists config (
            key   text primary key,
            value text not null
        )
    """)
    conn.commit()
    return conn


# --------------------------------------------------------------------------- #
# Delete a paper — removes it from everywhere
# --------------------------------------------------------------------------- #


def delete_paper(paper_id: str, qdrant, conn: sqlite3.Connection) -> None:
    """Erase a paper's entire existence:
    1. All Qdrant chunk vectors (single filtered delete via payload index)
    2. All parent rows in SQLite
    3. Catalog row in SQLite
    4. Saved markdown file
    """
    from qdrant_client import models

    log.info("Deleting paper_id=%s from Qdrant …", paper_id)
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="paper_id",
                        match=models.MatchValue(value=paper_id),
                    ),
                ]
            )
        ),
    )
    conn.execute("delete from parents where paper_id = ?", (paper_id,))
    conn.execute("delete from papers  where paper_id = ?", (paper_id,))
    conn.commit()
    md = Path(MARKDOWN_DIR) / f"{paper_id}.md"
    md.unlink(missing_ok=True)
    log.info("Deleted paper_id=%s — gone from Qdrant, SQLite, and markdown.", paper_id)


# --------------------------------------------------------------------------- #
# CLI commands
# --------------------------------------------------------------------------- #


def cmd_status(conn: sqlite3.Connection) -> None:
    rows = conn.execute(
        "select status, count(*) from papers group by status"
    ).fetchall()
    total = conn.execute("select count(*) from papers").fetchone()[0]
    print(f"\nKnowledge base catalog — {STATE_DB}")
    print(f"  Total papers: {total}")
    for status, n in sorted(rows):
        print(f"  {status:12s}: {n}")
    print()


def cmd_delete(qdrant, conn: sqlite3.Connection) -> None:
    query = input("Search (filename or title keyword): ").strip()
    if not query:
        print("Cancelled.")
        return

    rows = conn.execute(
        "select paper_id, filename, title, num_chunks from papers "
        "where filename like ? or title like ? "
        "order by filename",
        (f"%{query}%", f"%{query}%"),
    ).fetchall()

    if not rows:
        print(f"No papers found matching '{query}'.")
        return

    print()
    for i, (pid, fname, title, n) in enumerate(rows):
        print(f"  [{i}] {fname}")
        print(f"       title:  {title or '(unknown)'}")
        print(f"       chunks: {n or 0}  |  paper_id: {pid}")
    print()

    if len(rows) == 1:
        idx = 0
    else:
        try:
            idx = int(input(f"Enter number to delete [0-{len(rows) - 1}]: "))
        except ValueError:
            print("Cancelled.")
            return

    if not (0 <= idx < len(rows)):
        print("Invalid selection.")
        return

    pid, fname, title, _ = rows[idx]
    confirm = input(
        f"\nDelete '{fname}' ({title or 'no title'})? This cannot be undone. [y/N]: "
    )
    if confirm.strip().lower() != "y":
        print("Cancelled.")
        return

    delete_paper(pid, qdrant, conn)
    print(f"\n'{fname}' has been completely removed from the knowledge base.")


def cmd_ingest(qdrant, conn: sqlite3.Connection) -> None:
    # Import here so the worker processes (spawned, not forked) don't pull
    # in the embedding model at import time.
    from backend.services.rag_engine import PhytoQueryEmbeddings

    embeddings = PhytoQueryEmbeddings(primary_model=EMBEDDING_MODEL_NAME)

    ensure_collection(qdrant)

    all_paths = sorted(
        glob.glob(os.path.join(RAW_DIR, "**/*.pdf"), recursive=True)
        + glob.glob(os.path.join(RAW_DIR, "*.pdf"))
    )
    all_paths = sorted(set(all_paths))
    log.info("Found %d PDFs in %s", len(all_paths), RAW_DIR)

    # filename → paper_id already on record; detects revised PDFs.
    existing = dict(conn.execute("select filename, paper_id from papers").fetchall())

    pending = []
    for path in all_paths:
        pid = file_paper_id(path)
        fname = Path(path).name

        # Detect revised PDFs (same filename, different content).
        old_pid = existing.get(fname)
        if old_pid and old_pid != pid:
            log.info("%s: content changed — removing stale entry %s", fname, old_pid)
            delete_paper(old_pid, qdrant, conn)

        row = conn.execute(
            "select status from papers where paper_id = ?", (pid,)
        ).fetchone()
        if row and row[0] == "indexed":
            continue

        pending.append((path, pid))

    log.info(
        "%d to process  |  %d already indexed  |  %d parse workers",
        len(pending),
        len(all_paths) - len(pending),
        PARSE_WORKERS,
    )

    if not pending:
        log.info("Nothing to do.")
        finalize_collection(qdrant)  # re-enable HNSW even if nothing was added
        return

    ok = failed = 0
    # "spawn" avoids inheriting the CUDA context from the parent process.
    # "fork" + CUDA = known source of silent hangs on Linux.
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=PARSE_WORKERS,
        initializer=_init_worker,
        mp_context=ctx,
    ) as pool:
        futures = {
            pool.submit(parse_and_chunk, path, pid): (path, pid)
            for path, pid in pending
        }
        for future in as_completed(futures):
            path, pid = futures[future]
            t0 = time.perf_counter()  # covers wait-for-parse + embed + store
            try:
                parsed = future.result()
                n_chunks = embed_and_store(parsed, embeddings, qdrant, conn)
                ok += 1
                log.info(
                    "OK   %-50s  %3d chunks  %.1fs",
                    Path(path).name,
                    n_chunks,
                    time.perf_counter() - t0,
                )
            except Exception as exc:
                failed += 1
                conn.execute(
                    "insert or replace into papers "
                    "(paper_id, filename, status, error, indexed_at) "
                    "values (?,?,'error',?,?)",
                    (pid, Path(path).name, str(exc), time.time()),
                )
                conn.commit()
                log.error("FAIL %-50s  %s", Path(path).name, exc)
                traceback.print_exc()

    finalize_collection(qdrant)

    conn.execute(
        "insert or replace into config (key, value) values (?, ?)",
        ("embedding_model", EMBEDDING_MODEL_NAME),
    )
    conn.execute(
        "insert or replace into config (key, value) values (?, ?)",
        ("embedding_dim", str(DENSE_DIM)),
    )
    conn.commit()

    log.info("Done — %d indexed, %d failed. Re-run to retry failures.", ok, failed)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    from qdrant_client import QdrantClient

    parser = argparse.ArgumentParser(description="Knowledge base ingestion pipeline")
    parser.add_argument(
        "--status", action="store_true", help="Print catalog summary and exit"
    )
    parser.add_argument(
        "--delete", action="store_true", help="Interactive paper deletion"
    )
    args = parser.parse_args()

    qdrant = QdrantClient(url=QDRANT_URL)
    conn = init_db()

    if args.status:
        cmd_status(conn)
        sys.exit(0)

    if args.delete:
        cmd_delete(qdrant, conn)
        sys.exit(0)

    cmd_ingest(qdrant, conn)


if __name__ == "__main__":
    main()
