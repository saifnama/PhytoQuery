"""Upload plumbing — canonical home (Phase 2 merge).

Consolidates the former rag_storage.py (paths + markdown extractor),
upload_jobs.py (JSON job store) and user_locks.py (per-user mutex).

Fixes vs originals:
- per-user lock has an acquire timeout (60 s) — a stuck upload can no
  longer wedge the user forever; contention surfaces as HTTP 503.
- idle locks are evicted (1 h) — the lock dict no longer leaks.
- stored filenames are sanitized in one place (`safe_filename`).
"""
import asyncio
import json
import os
import re
import tempfile
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from backend.src.common.paths import data_dir


BASE_DATA_DIR = data_dir()
RAG_UPLOADS_DIR = BASE_DATA_DIR / "uploads"
RAG_JOBS_DIR = BASE_DATA_DIR / "rag_jobs"

_LOCK_ACQUIRE_TIMEOUT = 60.0
_LOCK_IDLE_EVICT_SECONDS = 3600.0


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_filename(filename: str) -> str:
    """Strip directories and unsafe chars. Never trust UploadFile.filename."""
    base = os.path.basename(filename or "")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    return safe or "upload.pdf"


def get_user_upload_dir(user_id: str) -> Path:
    return ensure_dir(RAG_UPLOADS_DIR / user_id)


def get_user_upload_file_path(user_id: str, filename: str) -> Path:
    return get_user_upload_dir(user_id) / safe_filename(filename)


def get_user_markdown_file_path(user_id: str, filename: str) -> Path:
    """Extracted-markdown sidecar: ``<filename>.md`` next to the PDF.

    Used by citation rendering — clicking a [N] superscript opens the
    paper's extracted markdown with the cited chunk highlighted."""
    return get_user_upload_dir(user_id) / f"{safe_filename(filename)}.md"


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


class UploadJobStore:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else get_job_store_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        return self.base_dir / f"{job_id}.json"

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        fd, temp_path = tempfile.mkstemp(dir=str(self.base_dir), prefix=path.stem, suffix=".tmp")
        temp_file = Path(temp_path)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
            temp_file.replace(path)
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._write_json(self._job_path(payload["job_id"]), payload)
        return payload

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        path = self._job_path(job_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def update(self, job_id: str, fields: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        existing = self.get(job_id)
        if not existing:
            return None
        existing.update(fields)
        self._write_json(self._job_path(job_id), existing)
        return existing

    def list_for_user(self, user_id: str) -> List[Dict[str, Any]]:
        jobs = []
        for path in self.base_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("user_id") == user_id:
                jobs.append(payload)
        return sorted(jobs, key=lambda item: item.get("created_at", ""))

    def delete(self, job_id: str) -> None:
        path = self._job_path(job_id)
        if path.exists():
            path.unlink()

    def delete_user_jobs(self, user_id: str) -> None:
        for job in self.list_for_user(user_id):
            self.delete(job["job_id"])

    def prune(self, max_age_seconds: float = 604800) -> int:
        """Remove job records older than max_age_seconds (default 7 days)."""
        cutoff = datetime.now(timezone.utc).timestamp() - max_age_seconds
        deleted = 0
        for path in self.base_dir.glob("*.json"):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                created_at_str = payload.get("created_at")
                if not created_at_str:
                    continue
                created_ts = datetime.fromisoformat(created_at_str).timestamp()
                if created_ts < cutoff:
                    path.unlink()
                    deleted += 1
            except (json.JSONDecodeError, ValueError, OSError):
                continue
        return deleted


class UserLockManager:
    """One asyncio mutex per user. Different users proceed in parallel."""

    def __init__(
        self,
        acquire_timeout: float = _LOCK_ACQUIRE_TIMEOUT,
        idle_evict_seconds: float = _LOCK_IDLE_EVICT_SECONDS,
    ):
        self._locks: dict[str, tuple[asyncio.Lock, float]] = {}
        self._locks_guard = asyncio.Lock()
        self._acquire_timeout = acquire_timeout
        self._idle_evict_seconds = idle_evict_seconds

    async def _get_lock(self, user_id: str) -> asyncio.Lock:
        async with self._locks_guard:
            now = time.monotonic()
            for uid, (lock, last_used) in list(self._locks.items()):
                if uid != user_id and not lock.locked() and now - last_used > self._idle_evict_seconds:
                    del self._locks[uid]
            entry = self._locks.get(user_id)
            if entry is None:
                entry = (asyncio.Lock(), now)
                self._locks[user_id] = entry
            return entry[0]

    def _touch(self, user_id: str) -> None:
        entry = self._locks.get(user_id)
        if entry is not None:
            self._locks[user_id] = (entry[0], time.monotonic())

    @asynccontextmanager
    async def lock(self, user_id: str):
        lock = await self._get_lock(user_id)
        try:
            await asyncio.wait_for(lock.acquire(), timeout=self._acquire_timeout)
        except TimeoutError:
            raise HTTPException(
                status_code=503,
                detail="Another operation for this session is still running. Retry shortly.",
            )
        try:
            yield
        finally:
            lock.release()
            self._touch(user_id)


user_lock_manager = UserLockManager()
