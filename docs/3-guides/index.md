# 3 · Guides

**Import data.** `python scripts/import_sqlite.py -i data.csv -d backend/src/db/bloomindex.sqlite`
(normalized NER CSV → papers + paper_entities; idempotent).

**Migrations.** Fresh DB: `alembic -c backend/alembic.ini upgrade head`.
Existing prod DB (already has the shape): `alembic stamp head` — never upgrade.

**Evals.** `backend/evals/ner/` (dictionary vs LLM vs hybrid on gold labels),
`backend/evals/rag/` (MCQ accuracy + RAGAS open-ended). Unpinned `ragas`/
`nervaluate` — pin before publishing numbers.

**Tests.** `pytest backend/tests/ -q` (mock-transport, ~2 min cold) ·
`npx tsc -b --noEmit` + `npm run build` in `frontend/`.

**Troubleshooting.** Port clash → `scripts/qdrant.sh status`. Empty dashboard →
`backend/src/db/bloomindex.sqlite` missing (see import). Chat 503 → another op holds
the user lock (60 s timeout) or Qdrant down. Slow first NER → gazetteer
pickle build (~1 min, once).
