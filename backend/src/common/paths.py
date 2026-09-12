"""Filesystem roots. Single place — never os.getcwd() (breaks under systemd/Docker).

Layout anchor: this file is backend/src/common/paths.py, so repo root is 4 levels up.

  tmp/              per-user temp: uploads, rag jobs, markdown sidecars,
                    ner uploads, API caches, session secret, embedded Qdrant
  knowledge_base/   ingest pipeline runtime: papers PDFs, parsed markdown,
                    kb.sqlite state (see scripts/ingest_kb.py)
Both are git-ignored runtime — code that needs them comes here.
"""
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()


def repo_root() -> Path:
    return _THIS_FILE.parent.parent.parent.parent


def tmp_dir() -> Path:
    return repo_root() / "tmp"


def kb_dir() -> Path:
    return repo_root() / "knowledge_base"


def frontend_dist() -> Path:
    return repo_root() / "frontend" / "dist"


def safe_join(base: Path, *parts: str) -> Path | None:
    """Join untrusted path parts onto base. Returns None on traversal outside base."""
    candidate = (base.joinpath(*parts)).resolve()
    try:
        candidate.relative_to(base.resolve())
    except ValueError:
        return None
    return candidate
