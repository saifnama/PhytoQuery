import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.core.rag_storage import get_job_store_dir


class UploadJobStore:
    def __init__(self, base_dir: Optional[Path] = None):
        self.base_dir = Path(base_dir) if base_dir else get_job_store_dir()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _ensure_base_dir(self) -> None:
        # Re-create the base directory on every operation. ``__init__``'s
        # mkdir runs once at module load, but the directory can vanish
        # later (manual cleanup, fresh checkout, blown-away ``data/``
        # dir). mkdir on an existing directory is a no-op so this is
        # essentially free and makes every method below resilient.
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _job_path(self, job_id: str) -> Path:
        self._ensure_base_dir()
        return self.base_dir / f"{job_id}.json"

    def _write_json(self, path: Path, payload: Dict[str, Any]) -> None:
        self._ensure_base_dir()
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
        self._ensure_base_dir()
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
        """Remove job records older than max_age_seconds (default 7 days).

        Returns the number of deleted records.
        """
        self._ensure_base_dir()
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
