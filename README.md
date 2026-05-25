# PhytoQuery

A research-paper reader and RAG workbench for **phytochemistry, ethnobotany, and natural-product chemistry**. Fetches papers from Europe PMC / OpenAlex / PubMed, parses JATS XML, renders full-text inline with dictionary-backed entity highlighting, lets you upload PDFs for hybrid retrieval-augmented chat, and indexes extracted entities into a SQLite knowledge base that powers dashboards and graph views.

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │                         React 19 + Vite (TS)                        │
 │  Paper Reader | Search | Chat (RAG) | NER | Dashboard | Graph 3D    │
 └─────────────────────────────────────────────────────────────────────┘
                                  │
                          REST / NDJSON streams
                                  │
 ┌─────────────────────────────────────────────────────────────────────┐
 │                  FastAPI backend (async, uvicorn)                   │
 │                                                                     │
 │   ┌────────────┐  ┌────────────┐  ┌─────────────┐  ┌──────────────┐ │
 │   │ Paper svc  │  │ NER engine │  │ RAG engine  │  │ Dashboard svc│ │
 │   │ EuropePMC  │  │ spaCy +    │  │ Qwen3-Emb / │  │ SQLA per-    │ │
 │   │ OpenAlex   │  │ LLM        │  │ bge-m3 +    │  │ type schema  │ │
 │   │ PubMed     │  │ assist     │  │ zerank-2    │  │              │ │
 │   └────────────┘  └────────────┘  └─────────────┘  └──────────────┘ │
 │         │                │               │                  │      │
 └─────────┼────────────────┼───────────────┼──────────────────┼──────┘
           │                │               │                  │
           ▼                ▼               ▼                  ▼
   ┌──────────────┐ ┌────────────┐ ┌──────────────┐  ┌─────────────────┐
   │ Europe PMC / │ │ Gazetteers │ │ Qdrant       │  │ SQLite (WAL)    │
   │ OpenAlex /   │ │ (CSV →     │ │ Server       │  │ db_data/        │
   │ PubMed APIs  │ │  pickled   │ │ (Docker, or  │  │ phytoquery.     │
   │              │ │  Phrase-   │ │ embedded     │  │ sqlite          │
   │              │ │  Matcher)  │ │ fallback)    │  │                 │
   └──────────────┘ └────────────┘ └──────────────┘  └─────────────────┘
```

---

## Features

### Paper Reader
- Auto-generated 2-level TOC (H2 → H3) with scroll-spy highlighting
- Continuous smooth scroll — every section rendered inline
- Chemical-entity popups with client-side molecule rendering from SMILES (smiles-drawer)
- Inline PDF download
- Source fallback chain — DOI → Europe PMC (full text) → OpenAlex (abstract + direct PDF) → PubMed

### Search
- Search **Europe PMC** or **OpenAlex** (source selectable in the UI)
- Filter by Open Access, Full Text, Article Type
- Sort by Relevance, Citations, or Date
- Page size fixed at 25

### Chat / RAG
- Upload PDFs, ask questions, get cited answers
- **Hybrid retrieval** — Qdrant runs the dense + sparse fusion natively. Dense vectors (Qwen3-Embedding-4B at 1024 dims via MRL truncation, or bge-m3 fallback at 1024 native) and sparse BM25 vectors (FastEmbed's `Qdrant/bm25`, IDF-weighted with stemming + stop-word removal) live on the same point under named-vector slots. A single `query_points(prefetch=[dense, sparse], FusionQuery(RRF))` call runs the Reciprocal Rank Fusion server-side — no Python-side merge.
- **Cross-encoder rerank** — `zeroentropy/zerank-2` reranks the fused top-N
- **Structured-output citations** — schema-enforced JSON-mode follow-up selects which retrieved chunks were used; the reranker attaches `[cN]` markers to the sentence each chunk best supports. The schema rejects empty citation lists, so answers cannot be uncited.
- **Click-to-highlight** — clicking a `[cN]` chip jumps to the source paper and flash-highlights the exact sentence via byte-precise `body_start`/`body_end` offsets recorded at index time
- **Dual parsers** — PyMuPDF (fast) or Docling (detailed, table/structure-preserving)
- **Parallel ingestion** — per-user upload jobs run PDFs concurrently in a `ThreadPoolExecutor` (PyMuPDF and Docling both release the GIL); per-file `try/except` so one bad PDF can't kill a 1000-PDF batch; flush to Qdrant every 50 files so peak RAM stays bounded and partial progress survives a crash
- **Chunked uploads** — the frontend slices large multi-file uploads into 20-PDF batches so reverse-proxy body limits (nginx, Cloudflare 100 MB) don't trip on multi-GB selections
- **Embedding-dim safety** — `_ensure_qdrant_collection` clamps `RAG_EMBEDDING_DIM` to the loaded model's native dim. MRL truncation can shrink but never expand; configuring 1024 with a 384-dim fallback model now warns and creates the collection at 384, instead of failing every upload after the fact.
- **Permissive JSON parsing** — `json-repair` recovers the structured citation object from common LLM JSON-mode malformations (trailing commas, code fences, single quotes, unclosed braces) with no extra LLM round-trip
- **Lazy model loading** — RAG models load on first use, ~200 MB startup footprint

### Named Entity Recognition (NER)
- **Dictionary-backed** matchers (spaCy PhraseMatcher with stemmed surface forms): PLANT PART, ANALYTICAL TECHNIQUE, EXTRACTION METHOD, DEVELOPMENT STAGE, SEASON, SPECIES, CHEMICAL, BIOACTIVITY
- **LLM-assisted** path (requires Ollama / Groq / OpenRouter / llama.cpp config)
- Entities highlighted inline and grouped in a sidebar; CSV export
- **Graph View** — physics-based knowledge graph linking the paper's DOI to its extracted entities (vis-network)

### Dashboards (over the SQLite knowledge base)
- Per-entity-type counts and timelines
- Journal-distribution widget, entity doughnut, publication timeline
- Plant-origin geo heatmap (ECharts `effectScatter`)
- 3D entity-co-occurrence graph

---

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI 0.136, uvicorn (uvloop on Linux/macOS), Python 3.10 → 3.14 |
| Frontend | React 19.2, Vite 8, TypeScript 5.9, Tailwind CSS v4 |
| Routing | TanStack Router (file-based, Zod-typed search params, scroll restoration, route-level error/404 fallbacks) |
| UI | shadcn/ui (Radix primitives + CVA), Phosphor Icons, `tw-animate-css`, Sonner toasts |
| Client state | Zustand (UI), TanStack Query (server data + polling) |
| Dictionary NER | spaCy 3.8 PhraseMatcher |
| Embeddings | Qwen/Qwen3-Embedding-4B (primary, 2560 native → 1024 via MRL), BAAI/bge-m3 (fallback, 1024 native) |
| Reranker | zeroentropy/zerank-2 (CrossEncoder, ~1 GB) |
| Vector DB | Qdrant 1.18 — Server (Docker) or embedded local, hybrid dense + sparse named vectors |
| Sparse encoder | FastEmbed `Qdrant/bm25` (IDF-weighted, stemming, stop-words) |
| RAG glue | LangChain 1.2, langchain-qdrant 1.1, json-repair |
| PDF parsing | Docling 2.92 (detailed, structure-preserving), PyMuPDF / fitz 1.27 (fast) + pymupdf4llm |
| Charts | ECharts |
| Graph | vis-network (per-paper KG, lazy-loaded) |
| Molecules | smiles-drawer (SMILES → 2D structure) |
| Sanitization | nh3 (server, Rust-backed), DOMPurify (client) |
| Paper sources | Europe PMC API, OpenAlex API, PubMed eutils |
| RAG LLM | llama.cpp / vLLM / LM Studio (OpenAI-compatible) > Groq > OpenRouter > Ollama |
| NER LLM | llama.cpp > Ollama > Groq > OpenRouter |
| Knowledge base | SQLite (WAL, FK on, busy-timeout) + async SQLAlchemy (`sqlite+aiosqlite`) |
| Config | python-dotenv + per-environment `.env.<profile>` switching |

---

## Quick start

### Prerequisites
- Python 3.10+ (3.12 recommended; 3.14 works with the threading guards in `backend/app.py`)
- Node.js 18+ or [Bun](https://bun.sh/) for the frontend
- Docker (for Qdrant Server — the recommended setup). Rootless Docker or membership in the `docker` group is enough; no `sudo` required.

### 1. Start Qdrant

The bundled helper handles container lifecycle, storage paths, and a health probe. Same subcommands on every OS:

```bash
# Linux / macOS
./scripts/qdrant.sh start          # creates on first run; idempotent
./scripts/qdrant.sh status         # state + health + collection count
./scripts/qdrant.sh stop           # stops, keeps storage on disk
./scripts/qdrant.sh restart
./scripts/qdrant.sh logs           # tail -f
./scripts/qdrant.sh remove         # delete container; storage preserved
```

```powershell
# Windows
.\scripts\qdrant.ps1 start
.\scripts\qdrant.ps1 status
.\scripts\qdrant.ps1 stop
.\scripts\qdrant.ps1 restart
.\scripts\qdrant.ps1 logs
.\scripts\qdrant.ps1 remove
```

Defaults:

| Setting | Linux/macOS | Windows |
|---------|-------------|---------|
| Container name | `phytoquery-qdrant` | `pq_qdrant` |
| Storage | `~/.local/share/phytoquery/qdrant_storage` | `%LOCALAPPDATA%\phytoquery\qdrant_storage` |
| Image | `qdrant/qdrant:v1.18.0` | same |
| REST port | 6333 | same |
| gRPC port | 6334 | same |

Override any of them with env vars: `QDRANT_CONTAINER`, `QDRANT_STORAGE_DIR`, `QDRANT_VERSION`, `QDRANT_PORT_REST`, `QDRANT_PORT_GRPC`. Once it's up, the helper prints the REST URL (`http://localhost:6333`) and the Web UI URL (`http://localhost:6333/dashboard`).

> **Don't have Docker?** You can fall back to embedded mode by leaving `RAG_QDRANT_URL` unset — the backend will use an in-process Qdrant client backed by `data/qdrant/`. See [Embedded mode caveats](#embedded-mode-caveats) below.

### 2. Backend

```bash
# create a venv
python -m venv phytovenv

# activate
phytovenv\Scripts\Activate.ps1       # Windows PowerShell
phytovenv\Scripts\activate           # Windows CMD
source phytovenv/bin/activate        # Linux / macOS

# deps
pip install -r backend/requirements.txt
python -m spacy download en_core_web_sm

# point at Qdrant + configure providers (see "Configuration" below)
cp .env.example .env
# edit .env — at minimum set RAG_QDRANT_URL and one LLM provider

# run
uvicorn backend.app:app --host 0.0.0.0 --port 8000
```

`backend/app.py` sets a few env-var floors **before any ML library imports**, to keep fastembed / transformers / torch from spawning over-eager worker pools and segfaulting on Python 3.14:

```
LOKY_MAX_CPU_COUNT=1
TOKENIZERS_PARALLELISM=false
OMP_NUM_THREADS=ceil(cpu_count / 2)
MKL_NUM_THREADS=ceil(cpu_count / 2)
```

All four use `setdefault`, so anything you `export` (or set in `.env` / Slurm batch) wins.

### 3. Frontend

```bash
cd frontend
bun install        # or: npm install / pnpm install
bun run dev        # http://localhost:5173 (dev server, HMR)

# OR — production build, served by the FastAPI process itself
bun run build      # writes frontend/dist/
# then visit http://localhost:8000 — FastAPI serves the SPA + API on one port
```

### 4. Workers

| Qdrant mode | `uvicorn --workers` | Notes |
|-------------|---------------------|-------|
| **Server (Docker)** | any N | All workers share one Qdrant Server over HTTP/gRPC. Recommended for production. |
| **Embedded (local)** | **1 only** | The embedded client takes an exclusive `flock()` on its storage directory; multi-worker deadlocks immediately. |

---

## Configuration

### Profiles — one-switch environment selection

Set **one** env var and the matching `.env.<profile>` file loads automatically:

```bash
export PHYTOQUERY_PROFILE=macbook    # loads .env.macbook
export PHYTOQUERY_PROFILE=server     # loads .env.server  (A100 / Slurm)
export PHYTOQUERY_PROFILE=demo       # loads .env.demo  (if you create it)
```

Precedence (highest → lowest):

1. **Real OS env vars** — Slurm's `CUDA_VISIBLE_DEVICES`, `docker run -e`, systemd `Environment=` — always win
2. **`.env.<PHYTOQUERY_PROFILE>`** — profile-specific overrides
3. **`.env`** — base / shared defaults
4. **Defaults in `backend/config.py`**

Set the profile **once per environment** (shell rc, systemd unit, Slurm batch script, `docker run -e`) and the right values load automatically. Replaces the old `cp .env.macbook .env` shuffle.

The legacy workflow still works if you prefer it — just copy the right preset to `.env`:

```bash
cp .env.server .env       # or cp .env.macbook .env
```

### LLM providers

PhytoQuery dispatches across four providers per pipeline. The **first one with credentials wins**; the rest stand by as fallbacks.

**RAG priority:** llama.cpp (or any OpenAI-compatible self-host) → Groq → OpenRouter → Ollama

```bash
# Self-hosted OpenAI-compatible (highest priority when URL is set —
# explicit opt-in always wins). Works with llama.cpp `server`, vLLM,
# LM Studio, LocalAI, Text Generation WebUI. URL accepts any of:
#   https://name.trycloudflare.com
#   https://name.trycloudflare.com/v1
#   https://name.trycloudflare.com/v1/chat/completions
# RAG_LLAMACPP_URL=https://your-name.trycloudflare.com
# RAG_LLAMACPP_MODEL=qwen2.5-7b-instruct
# RAG_LLAMACPP_API_KEY=               # optional; only if server uses --api-key

# Groq (fastest cloud, generous free tier — https://console.groq.com/keys)
RAG_GROQ_API_KEY=gsk_...
RAG_GROQ_MODEL=llama-3.3-70b-versatile

# OpenRouter (diverse model catalog)
RAG_OPENROUTER_API_KEY=sk-or-v1-...
RAG_OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free

# Ollama (local fallback — uses /api/chat, NOT OpenAI-compatible)
RAG_OLLAMA_URL=http://localhost:11434
RAG_OLLAMA_MODEL=llama3.1:8b
```

**NER priority:** llama.cpp → Ollama → Groq → OpenRouter (local-first; bulk per-paper extraction is cheaper local)

```bash
# NER_LLAMACPP_URL=https://your-name.trycloudflare.com
NER_OLLAMA_URL=http://localhost:11434
NER_OLLAMA_MODEL=llama3.1:8b
NER_GROQ_API_KEY=gsk_...
NER_OPENROUTER_API_KEY=sk-or-v1-...
```

### Qdrant — server vs embedded

```bash
# Server mode (recommended). Leave unset for embedded.
RAG_QDRANT_URL=http://localhost:6333
# Optional bearer (Qdrant Cloud, or any server started with --service.api_key=...)
# RAG_QDRANT_API_KEY=

# Embedded mode storage path. Only consulted when RAG_QDRANT_URL is empty.
# Leave empty for default `<repo>/data/qdrant/`. ~ is expanded; relative
# paths resolve to absolute at startup.
# RAG_QDRANT_DIR=
```

### Embedding + reranker

```bash
RAG_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
RAG_FALLBACK_EMBEDDING_MODEL=BAAI/bge-m3

# MRL (Matryoshka) truncation. The new safety clamp means:
#   * If RAG_EMBEDDING_DIM <= model_dim: used verbatim (truncates down)
#   * If RAG_EMBEDDING_DIM >  model_dim: clamped to model_dim + warning
# So setting 1024 here is safe regardless of which model actually loaded.
RAG_EMBEDDING_DIM=1024

RAG_EMBEDDING_INSTRUCTION=Instruct: Given a scientific query about phytochemistry, biology, or natural products, retrieve relevant research passages.

RAG_RERANKER_MODEL=zeroentropy/zerank-2     # commercial: zerank-1-small (Apache-2.0)

# GPU settings (CUDA only)
RAG_USE_FLASH_ATTENTION=true
RAG_MULTI_GPU=true

# Retrieval tuning
RAG_TEMPERATURE=0.1
RAG_CONTEXT_WINDOW=8192
RAG_TOP_K=10
RAG_SIMILARITY_THRESHOLD=0.85
```

### GPU auto-detection

| Environment | Behaviour |
|-------------|-----------|
| Slurm A100 / 4090 | Slurm sets `CUDA_VISIBLE_DEVICES` → PyTorch picks CUDA |
| MacBook (M-series) | `torch.backends.mps` available → Apple Metal |
| Windows / CPU box | No GPU detected → CPU |

Force a specific device with `RAG_DEVICE=cuda|mps|cpu` if auto-detection picks wrong.

---

## API endpoints

### Paper
| Endpoint | Method | What |
|----------|--------|------|
| `/paper/json` | POST | Fetch a paper by DOI/PMCID/PMID (`source: europepmc \| openalex`) |
| `/paper/section/json` | POST | Switch the rendered section |
| `/paper/pdf?identifier=…` | GET | Download the paper PDF |
| `/paper/pdf-proxy?url=…` | GET | CORS-friendly PDF proxy |
| `/paper/db/list` | GET | Paginated list of indexed papers (knowledge base) |
| `/paper/db/{doi}/entities` | GET | All entities for a paper, grouped by type |

### Search
| Endpoint | Method | What |
|----------|--------|------|
| `/search/json` | POST | Search Europe PMC or OpenAlex |
| `/search/types?source=openalex` | GET | Supported article-type filters per source |

### NER
| Endpoint | Method | What |
|----------|--------|------|
| `/ner/process` | POST | Run NER on raw text |
| `/ner/doi/json` | POST | Run NER on a paper by DOI |
| `/ner/cache/{doi}` | DELETE | Drop the cached NER result |
| `/ner/upload/json` | POST | Upload a PDF for one-shot NER |
| `/ner/uploaded/{stored_filename}` | GET / DELETE | View / delete an uploaded PDF |

### RAG / Chat
| Endpoint | Method | What |
|----------|--------|------|
| `/api/chat/upload/json` | POST | Upload PDFs; returns `job_id` (processing happens in the background) |
| `/api/chat/upload/status/{job_id}` | GET | Poll a job — `processing` / `completed` / `failed` |
| `/api/chat/upload/jobs` | GET | Active jobs for the current session |
| `/api/chat/files/json` | GET | Indexed files in the user's corpus |
| `/api/chat/files/{name}/markdown` | GET | Extracted markdown (for the citation preview panel) |
| `/api/chat/files/{name}/content` | GET | Inline PDF stream for the viewer |
| `/api/chat/files/{name}` | DELETE | Drop one source + its chunks |
| `/api/chat/reset` | POST | Delete **all** chunks for this user |
| `/api/chat/cleanup` | POST | Drop all user data — chunks, uploads, markdown, jobs |
| `/api/chat/query/json` | POST | Non-streaming RAG query |
| `/api/chat/query/stream` | POST | NDJSON streaming. Frame types: `text_delta` (token chunk), `sources` (retrieved chunk list), `answer_corrected` (final answer after the citation-attach pass), `citations` (the `[cN]` → chunk mapping), `done` (clean end), `error` (mid-stream fatal). |
| `/api/chat/suggest` | POST | Follow-up question suggestions |

### Dashboard / DOI / Health
| Endpoint | Method | What |
|----------|--------|------|
| `/api/dashboard/metrics` | GET | Counts, timelines, distributions |
| `/api/dashboard/sunburst` | GET | Hierarchical entity-type rollup |
| `/api/dashboard/graph3d` | GET | 3D co-occurrence graph data |
| `/doi/abstract` | GET | DOI abstract via Crossref / OpenAlex fallback |
| `/health/ready` | GET | Readiness probe |

---

## Project structure

```
PhytoQuery/
├── backend/
│   ├── app.py                  # FastAPI app; sets ML-safe env defaults BEFORE imports
│   ├── config.py               # env loader + provider selection
│   ├── api/                    # FastAPI routers
│   │   ├── paper.py            # Paper fetching + DB-backed paper endpoints
│   │   ├── search.py           # Europe PMC / OpenAlex search
│   │   ├── rag.py              # /api/chat/* — upload, query, stream, files, reset
│   │   ├── ner.py              # Text/DOI NER
│   │   ├── ner_pdf.py          # PDF-upload NER
│   │   ├── dashboard.py        # Aggregates over the SQLite knowledge base
│   │   ├── doi.py              # DOI abstract fallback
│   │   └── health.py           # Readiness probe
│   ├── core/                   # Cross-cutting utilities
│   │   ├── caching.py          # File-based paper / NER cache
│   │   ├── sanitizer.py        # nh3 HTML allowlist
│   │   ├── highlighter.py      # Entity-aware HTML highlighting
│   │   ├── http_client.py      # Shared httpx.AsyncClient lifecycle
│   │   ├── session.py          # Signed session cookie
│   │   ├── rag_storage.py      # Per-user upload / markdown paths
│   │   ├── upload_jobs.py      # Background-job store (in-memory + JSON spill)
│   │   └── user_locks.py       # Per-user asyncio.Lock for upload + reset races
│   ├── services/
│   │   ├── rag_engine.py       # The big one — embedder, reranker, Qdrant client,
│   │   │                       #   collection lifecycle, dim clamp, hybrid query,
│   │   │                       #   structured citation pass, NDJSON stream
│   │   ├── ner_engine.py       # spaCy matchers + LLM-assist with validation/retry
│   │   ├── search_service.py   # Europe PMC / OpenAlex search orchestration
│   │   ├── doi_resolver.py     # DOI → full-text/abstract chain
│   │   ├── europe_pmc/         # Europe PMC client + JATS parser
│   │   └── openalex/           # OpenAlex client
│   ├── gazetteer/              # spaCy PhraseMatcher dictionary backends
│   │   ├── data/               # CSV dictionaries (see below)
│   │   ├── build_matcher.py
│   │   └── <type>_matcher.py
│   ├── db/                     # SQLAlchemy models + migrations
│   │   ├── database.py         # async engine, WAL pragmas
│   │   ├── models.py           # Per-type schema (Paper + 7 entity tables + 7 junctions)
│   │   ├── migrate_schema.py   # v1 → v2 migration (legacy)
│   │   ├── migrate_schema_v2.py# One-shot per-type migration (Concrete Table Inheritance)
│   │   └── import_entities.py  # Bulk import from Excel/CSV
│   ├── schemas/schemas.py      # Pydantic request/response models
│   └── tests/                  # pytest
├── frontend/                   # React 19 + Vite 8 + TS 5.9
│   └── src/
│       ├── features/           # Page-level components (PaperPage, ChatPage, ...)
│       ├── layout/             # Header, Sidebar
│       ├── components/         # Cross-feature (UploadStatusListener, ...)
│       ├── stores/             # Zustand (uploadStore, ...)
│       ├── hooks/              # TanStack Query hooks
│       ├── ui/                 # Reusable primitives (ErrorBoundary, ...)
│       └── lib/                # API client (axios + ragApi / paperApi)
├── scripts/
│   ├── qdrant.sh               # Qdrant Server lifecycle helper (Linux / macOS)
│   └── qdrant.ps1              # Same for Windows PowerShell
├── db_data/                    # SQLite knowledge base (WAL + SHM files)
│   └── phytoquery.sqlite
├── data/                       # Runtime data (created lazily on first use)
│   ├── cache/                  # Paper + NER file cache
│   ├── qdrant/                 # Only used in embedded Qdrant mode (override with RAG_QDRANT_DIR)
│   └── uploads/                # Per-user uploaded PDFs + extracted markdown
├── .env.example                # Template — copy to .env or set PHYTOQUERY_PROFILE
├── .env.macbook                # MacBook (M-series, MPS) preset
├── .env.server                 # Institute server (A100 / Slurm) preset
└── README.md
```

### Gazetteers (`backend/gazetteer/data/`)

| File | Entries |
|------|---------|
| `chemical.csv` | 107K+ compounds |
| `species.csv` | 235K+ species |
| `plant_part.csv` | ~345 terms |
| `analytical_technique.csv` | ~184 techniques |
| `extraction_method.csv` | ~77 methods |
| `development_stage.csv` | ~45 stages |
| `season.csv` | ~55 terms |
| `bioactivity.csv` | ~124 activities |

---

## Testing

```bash
pytest backend/tests/ -v
```

The regression suite covers the RAG engine's collection lifecycle, the embedding-dim clamp, the structured citation schema, the per-user upload-job store, and the user-lock manager.

---
