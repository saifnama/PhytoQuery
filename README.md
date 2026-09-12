# BloomIndex

Literature RAG + NER + paper-reader workbench for phytochemistry,
ethnobotany and natural-product chemistry.

- **Explore** — Europe PMC / OpenAlex search + SQLite knowledge-base dashboard
- **Analyse** — PDF upload, dictionary+LLM entity extraction, comparison
- **Chat** — hybrid dense+BM25 RAG with `[cN]` citations
- **Paper** — full-text reader, entity highlighting, knowledge graph

## Quick start

```bash
./scripts/qdrant.sh start        # or .\scripts\qdrant.ps1 start (Windows)
python -m venv .venv && source .venv/bin/activate
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env             # set LLM_API_BASE_URL / KEY / MODEL
python run_api.py                # API → http://localhost:8000
cd frontend && npm install && npm run dev   # UI → http://localhost:5173
```

Details: [install](docs/1-install/index.md) · [concepts](docs/2-concepts/index.md) ·
[guides](docs/3-guides/index.md) · [config](CONFIGURATION.md) ·
[architecture](docs/architecture.md) · [API snapshot](packages/contracts/openapi.json)

## Layout

```text
backend/src/{routers,domain,common,db,papers,search,ner,chat}/  # FastAPI
frontend/src/{pages,features,components,lib,stores}/            # React 19 + Vite
scripts/   # ingest_kb, import_sqlite, qdrant helpers
docs/      # 0-start, 1-install, 2-concepts, 3-guides
tests/e2e/ # live-server smoke (smoke_api.py) + browser spec
```

## Verify

```bash
pytest backend/tests/ -q              # backend (mock-transport, ~2 min cold)
cd frontend && npm run build          # tsr + tsc + vite
python tests/e2e/smoke_api.py         # needs `python run_api.py` running
```

Workers: Qdrant Server → any `--workers N`; embedded mode → exactly 1.
