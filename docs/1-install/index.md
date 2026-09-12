# 1 · Install

Prereqs: Python 3.12, Node 18+, Docker or Podman (for Qdrant Server).

```bash
# Qdrant (Linux/macOS; Windows: .\scripts\qdrant.ps1 start)
./scripts/qdrant.sh start

# Backend
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env          # set LLM_API_BASE_URL / KEY / MODEL
python run_api.py             # → http://localhost:8000

# Frontend
cd frontend && npm install && npm run dev   # → http://localhost:5173
```

Workers: Qdrant **Server** allows any `uvicorn --workers N`. Embedded mode
(no `QDRANT_URL`) takes an exclusive lock — **exactly 1 worker**.

Profiles: `BLOOMINDEX_PROFILE=macbook|server|demo` selects `.env.<profile>`
over base `.env`. Real OS env vars always win. See `CONFIGURATION.md`.
