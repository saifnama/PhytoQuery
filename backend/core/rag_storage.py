import os
from pathlib import Path


BASE_DATA_DIR = Path(os.getcwd()) / "data"
RAG_UPLOADS_DIR = BASE_DATA_DIR / "uploads"
RAG_JOBS_DIR = BASE_DATA_DIR / "rag_jobs"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_user_upload_dir(user_id: str) -> Path:
    return ensure_dir(RAG_UPLOADS_DIR / user_id)


def get_user_upload_file_path(user_id: str, filename: str) -> Path:
    return get_user_upload_dir(user_id) / filename


def get_user_markdown_file_path(user_id: str, filename: str) -> Path:
    """Return the path where the extracted markdown view of an
    uploaded file lives. Convention: ``<filename>.md`` next to the
    original PDF in the per-user upload directory. Used by citation
    rendering — clicking a [N] superscript opens the paper's
    extracted markdown with the cited chunk highlighted."""
    return get_user_upload_dir(user_id) / f"{filename}.md"


def get_job_store_dir() -> Path:
    return ensure_dir(RAG_JOBS_DIR)


def delete_user_uploads(user_id: str) -> None:
    upload_dir = RAG_UPLOADS_DIR / user_id
    if not upload_dir.exists():
        return
    for child in upload_dir.iterdir():
        if child.is_file():
            child.unlink()
    try:
        upload_dir.rmdir()
    except OSError:
        pass


def delete_user_upload_file(user_id: str, filename: str) -> None:
    file_path = get_user_upload_file_path(user_id, filename)
    if file_path.exists():
        file_path.unlink()


def extract_paper_markdown(pdf_path) -> str:
    """Fast PDF → markdown using plain PyMuPDF (no OCR, no layout model).

    Per page: ``page.get_text("text")`` plus vector tables
    (``page.find_tables()``) appended as GFM pipe tables. Pages are
    joined with ``<!-- Page N -->`` markers that the citation pipeline
    relies on for page attribution and highlight offsets.

    Single shared helper for both ingest (services/rag_engine) and the
    legacy preview regen (api/rag) — parent-chunk offsets are byte
    offsets into exactly this text, so the two paths must never drift.
    """
    import pymupdf

    text_parts = []
    with pymupdf.open(pdf_path) as doc:
        for idx, page in enumerate(doc):
            parts = [(page.get_text("text") or "").strip()]
            try:
                for table in page.find_tables():
                    rows = table.extract() or []
                    cells = [c for r in rows if r for c in r]
                    if not any((c or "").strip() for c in cells):
                        continue  # border-only artifact, no content
                    md = (table.to_markdown() or "").strip()
                    if md:
                        parts.append(md)
            except Exception:
                pass  # tables are best-effort; page text is captured above
            text = "\n\n".join(p for p in parts if p).strip()
            if text:
                text_parts.append(f"<!-- Page {idx + 1} -->\n\n{text}")
    return "\n\n".join(text_parts)
