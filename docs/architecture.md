# Architecture (condensed — RESTRUCTURE_PLAN.md §1 is authoritative)

```text
frontend/src/pages/{explore,analyse,chat,paper}/   # route components
frontend/src/features/{search,reader,graph,chat}/  # reusable domain chunks
frontend/src/{components/ui,layout,lib/api/*,stores}/
backend/src/
  main.py settings.py dependencies.py* exceptions.py middleware.py*
  routers/{papers,search,ner,chat,dashboard,health}.py   # HTTP only, no SQL
  domain/            # Pydantic entities (no SQL/HTTP)
  common/{paths,cache,http,sessions,sanitize,highlight,uploads}.py
  db/{session,models,repository,importer}.py  # SQL only in repository.py
  papers/{service,europepmc,openalex,resolver,jats}.py
  search/service.py  ner/{service,dictionary,llm,dictionaries/}
  chat/{service,ingest,retrieval,citations,embeddings,llm,config,ai/}
  dashboard/service.py*
```
`*` = planned, not yet extracted. Shims at every pre-restructure import path
(`backend/api/*`, `backend/core/*`, `backend/services/*`, …) re-export the
canonical modules; delete in Phase 3.

Data: Qdrant (per-user collections + `kb_papers`) · SQLite `backend/src/db/`
(`papers`, `paper_entities`) · `data/{cache,uploads,qdrant}/` runtime.
Contracts: `packages/contracts/openapi.json` + generated TS (see VISION.md).
