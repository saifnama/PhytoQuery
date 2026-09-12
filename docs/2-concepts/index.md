# 2 · Concepts

**Chat/RAG.** PDFs → PyMuPDF/Docling extract → parent/child chunks → Qwen3/bge-m3
dense + BM25 sparse vectors in per-user Qdrant collections → server-side RRF
fusion → cross-encoder rerank (`RAG_RERANKER_MODEL`) → answer with `[cN]`
citations resolved against whole chunks. Code: `backend/src/chat/`
(`ingest/retrieval/citations/`).

**NER (hybrid).** 8 dictionary matchers (`ner/dictionaries/`, 300k+ terms) run
first; an LLM pass adds context types (LOCATION, DISEASE); validation-retry +
whole-word recounts drop hallucinations. `NER_HYBRID=false` = dictionary-only.

**KB vs uploads.** `scripts/ingest_kb.py` builds the permanent `kb_papers`
collection + SQLite KB (`backend/src/db/bloomindex.sqlite`: `papers` + `paper_entities`,
queried only via `db/repository.py`). Chat uploads are per-user, ephemeral.

**Health.** `/health/ready` dials the LLM `/models` endpoint (cached 60 s,
non-billable) and Qdrant when booted: `ready | degraded | down` (503).
