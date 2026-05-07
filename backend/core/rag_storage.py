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
