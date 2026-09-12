# AGENTS.md — backend

- Run: `python run_api.py` from repo root (imports `backend.app:app`).
- Tests: `pytest backend/tests/ -q`. Full gate before ML pin bumps: tests + one PDF upload + query.
- Config: `backend/config.py` (moving to `src/settings.py` in Phase 1). Precedence: OS env > `.env.<profile>` > `.env` > defaults.
- Conventions: routers thin (no SQL), logic in `services/` (moving to domain dirs). Pydantic in `schemas/` (moving to `domain/`).
- Gotchas: Python 3.14 `resource_tracker` patch in `app.py` must stay first; embedded Qdrant = `--workers 1` only.
