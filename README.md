# PhytoQuery

A research paper reader with Named Entity Recognition (NER) for phytochemical and ethnobotanical research. Retrieves papers from Europe PMC, parses JATS XML, and renders content with dictionary-backed entity highlighting.

## Features

### Paper Reader
- Auto-generated Table of Contents with 2-level hierarchy (H2 → H3)
- Continuous smooth scrolling — all sections rendered inline
- Scroll-spy highlighting — active section reflected in the sidebar
- Chemical entity popups with client-side molecule rendering from SMILES strings
- PDF download directly from paper page

### Search
- Search **Europe PMC** or **OpenAlex** (select source in UI)
- Filter by Open Access, Full Text, Article Type
- Sort by Relevance, Citations, or Date
- Page size: fixed at 25 results

### Chat / RAG
- Upload PDFs and query with AI
- Inline PDF viewer with side-by-side citation preview panel
- Upload to RAG directly from paper (stays on page, no navigation)
- **Structured-output citations** — schema-enforced JSON-mode follow-up call selects which retrieved chunks were used; the cross-encoder reranker then attaches `[cN]` markers to the sentence each chunk best supports. Replaces inline marker prompting (which silently dropped on list-style answers) with a guarantee: the schema rejects empty citation lists, so an answer can never be uncited.
- **Click-to-highlight**: clicking a citation chip opens the source paper at the cited passage, with the exact sentence flash-highlighted via byte-precise `body_start`/`body_end` offsets recorded at index time
- **Dual parser**: PyMuPDF (fast) or Docling (detailed, structure-preserving)
- **Hybrid retrieval**: dense vectors (Qdrant HNSW) + BM25 keyword matching with server-side Reciprocal Rank Fusion. Sparse vectors stored alongside dense vectors per point via Qdrant's native named-vectors schema (`Modifier.IDF` for BM25 scoring); a single `query_points(prefetch=[dense, sparse], FusionQuery(RRF))` call runs the hybrid merge in Qdrant rather than Python. Tokenization uses FastEmbed's `Qdrant/bm25` (stemming + stop-word removal + IDF weighting) — measurable retrieval quality improvement on scientific text vs naive whitespace split.
- **Parallel ingestion**: per-user upload jobs parse PDFs concurrently via `ThreadPoolExecutor` (PyMuPDF and Docling both release the GIL); per-file `try/except` so one bad PDF can't kill a 1000-PDF batch; flushes to Qdrant every 50 files so peak RAM is bounded and partial progress survives a crash
- **Chunked upload**: the frontend slices large multi-file uploads into 20-PDF batches so reverse-proxy body limits (nginx, Cloudflare 100 MB) don't trip on multi-GB selections
- **Instruction-Aware Architecture**: Custom domain prompts for both embedding (Qwen3) and reranking (zerank-2)
- **MRL Truncation**: Storage-efficient vectors (1024 dims) using Matryoshka Representation Learning
- **Lazy model loading**: RAG models only load on first use (~200MB startup)
- **Permissive JSON parsing**: `json-repair` recovers from trailing commas, code fences, single quotes, and unclosed braces in LLM JSON-mode output — no extra LLM call needed

### Named Entity Recognition (NER)
- **Dictionary-backed**: PLANT PART, ANALYTICAL TECHNIQUE, EXTRACTION METHOD, DEVELOPMENT STAGE, SEASON, SPECIES, CHEMICAL, BIOACTIVITY
- **LLM-assisted** (requires OpenRouter/Ollama config)
- Click "Find Key Terms" to run extraction
- Entities highlighted inline and grouped in sidebar
- **Graph View**: Interactive, physics-based knowledge graph linking DOI to extracted entities
- Export to CSV

### Source Fallbacks
- DOI → Europe PMC (full text) → OpenAlex (abstract + direct PDF)
- PMID → Europe PMC → PubMed
- PMCID → Europe PMC → NCBI PMC

## Tech Stack

| Layer | Technology |
|-------|------------|
| Backend | FastAPI (Python), uvicorn (dev server) |
| Frontend | React 19, TypeScript, Vite, Tailwind CSS v4 |
| Frontend routing | TanStack Router (file-based routes, typed search params via Zod, scroll restoration, route-level error/404 fallbacks) |
| Frontend UI | shadcn/ui (Radix primitives + class-variance-authority), Phosphor Icons, `tw-animate-css`, Sonner toasts |
| Frontend state | Zustand (client UI), TanStack Query (server data + polling cache) |
| NLP | spaCy PhraseMatcher (dictionary-backed) |
| Embeddings | Qwen3-Embedding-4B (primary), BAAI/bge-m3 (fallback) |
| Reranker | zeroentropy/zerank-2 (CrossEncoder) |
| RAG | LangChain, Qdrant (embedded, native hybrid), sentence-transformers, FastEmbed (Qdrant/bm25 sparse), json-repair |
| PDF Parsing | Docling (detailed), PyMuPDF/fitz (fast) |
| Charts | ECharts (dashboard: journal distribution widget, entity doughnut, publication timeline, plant-origin geo heatmap with `effectScatter`) |
| Graph | vis-network (per-paper knowledge graph, lazy-loaded) |
| Molecules | smiles-drawer (chemical structure rendering from SMILES strings) |
| Sanitization | nh3 (server), DOMPurify (client) |
| Paper Sources | Europe PMC API, OpenAlex API |
| LLM | Groq / OpenRouter / Ollama (provider-agnostic dispatch) |
| Config | python-dotenv (.env files per environment) |

## Project Structure

```
PhytoQuery/
├── backend/
│   ├── api/           # FastAPI endpoints
│   │   ├── paper.py    # Paper fetching with fallback
│   │   ├── search.py   # Europe PMC search
│   │   ├── rag.py       # RAG chat endpoints
│   │   ├── ner.py       # Standalone NER
│   │   └── doi.py      # DOI abstract fallback
│   ├── core/           # Utilities
│   │   ├── caching.py   # Simple file-based cache
│   │   ├── sanitizer.py # HTML sanitization
│   │   ├── highlighter.py # Entity highlighting
│   │   └── http_client.py # HTTP abstraction
│   ├── services/       # Business logic
│   │   ├── europe_pmc/ # Paper fetching/parsing
│   │   ├── openalex/  # OpenAlex API integration
│   │   ├── ner_engine.py # NER pipeline
│   │   ├── rag_engine.py # RAG + chat
│   │   └── doi_resolver.py # DOI fallback sources
│   ├── gazetteer/      # Dictionary matchers
│   │   ├── data/       # CSV dictionaries
│   │   └── *_matcher.py # spaCy PhraseMatcher
│   └── tests/         # pytest tests
├── frontend/           # React app
│   ├── src/
│   │   ├── features/    # Page components
│   │   ├── layout/      # Header, Sidebar
│   │   ├── components/  # App-level shared components (UploadStatusListener, …)
│   │   ├── stores/      # Zustand stores (uploadStore, …)
│   │   ├── hooks/       # TanStack Query hooks (useIndexedFiles, useUploadJobStatus, …)
│   │   ├── ui/          # Reusable UI primitives (ErrorBoundary, …)
│   │   └── lib/         # API client (axios + ragApi/paperApi)
│   └── dist/        # Built output
├── data/             # Runtime data
│   ├── cache/       # Paper & NER cache
│   ├── qdrant/      # Embedded Qdrant storage: per-user collections with named dense + sparse vectors, plus per-user parent JSON
│   └── uploads/     # Uploaded PDFs (per-user folders)
└── README.md
```

## Quick Start

### 1. Backend Setup

```bash
# Create and activate virtual environment
python -m venv phytovenv

# Windows (CMD)
phytovenv\Scripts\activate

# Windows (PowerShell)
phytovenv\Scripts\Activate.ps1

# Linux/Mac
source phytovenv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt

# Download spaCy model (for dictionary matchers)
python -m spacy download en_core_web_sm

# Run backend

# Option 1: Via uvicorn (recommended for dev)
# From PhytoQuery directory:
cd C:\Users\saif\saifnama_lab\PhytoQuery
uvicorn backend.app:app --host 0.0.0.0 --port 8000

# Option 2: Direct Python
cd C:\Users\saif\saifnama_lab\PhytoQuery
python -m backend.app --host 0.0.0.0 --port 8000
```

#### Production deployment (single-worker only)

The embedded Qdrant client (`QdrantClient(path="data/qdrant/")`) holds
an exclusive file lock on its storage directory, so the backend MUST
run with `--workers 1`:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --workers 1
```

Multi-worker mode (`--workers N > 1`) is not supported with the current
setup — every worker would fight for the same Qdrant lock and all but
one would fail with "database is locked". If you need horizontal
scaling, either run multiple FastAPI instances behind a reverse proxy
(each pointing at its own `data/qdrant/` directory), or re-introduce a
server-mode branch in `backend/services/rag_engine.py` that connects
multiple workers to one remote Qdrant via HTTP.

### 2. Frontend Setup

```bash
cd frontend
bun install
bun run dev
```

### 3. Access

Open http://localhost:8000

## Configuration

### Environment Variables

PhytoQuery uses `.env` files for per-environment configuration. Copy a preset:

```bash
# Institute server (A100 GPU):
cp .env.server .env

# MacBook M4 Pro (24GB):
cp .env.macbook .env
```

> **Priority**: Real env vars (e.g., Slurm's `CUDA_VISIBLE_DEVICES`) always override `.env` values.

#### GPU Auto-Detection

| Environment | How It Works |
|---|---|
| Slurm A100 | Slurm sets `CUDA_VISIBLE_DEVICES` → PyTorch detects CUDA → uses GPU |
| MacBook M4 | PyTorch detects `torch.backends.mps` → uses Apple Metal GPU |
| Windows/CPU | No GPU detected → uses CPU |

#### RAG Settings
```bash
# Embedding model
# Primary: BAAI/bge-m3 (great quality, ~2GB RAM, recommended for testing)
# Fallback: BAAI/bge-small-en-v1.5 (~130MB, blazing fast)
# Production: Qwen/Qwen3-Embedding-4B (~8GB VRAM, best quality)
RAG_EMBEDDING_MODEL=BAAI/bge-m3
RAG_FALLBACK_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# MRL dimension truncation (None=full 2560, 1024=balanced, 512=fast)
RAG_EMBEDDING_DIM=1024

# Domain-specific instruction for embedding (Qwen3 only)
RAG_EMBEDDING_INSTRUCTION="Instruct: Given a scientific query about phytochemistry, biology, or natural products, retrieve relevant research passages."

# Reranker
# Testing: cross-encoder/ms-marco-MiniLM-L-6-v2 (~90MB)
# Production: zeroentropy/zerank-2 (~1GB)
RAG_RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# GPU acceleration (CUDA only)
RAG_USE_FLASH_ATTENTION=true
RAG_MULTI_GPU=false

# RAG Tuning
RAG_TEMPERATURE=0.1
RAG_CONTEXT_WINDOW=8192
RAG_TOP_K=10
RAG_SIMILARITY_THRESHOLD=0.85
```

#### LLM Providers

PhytoQuery supports three LLM backends. Configure as many as you want — the
first configured one in the priority order is used, the rest are fallbacks
if the primary errors out.

**RAG priority: Groq → OpenRouter → Ollama** (cloud-fast → cloud-diverse → local)

```bash
# Groq (recommended — fastest cloud, generous free tier)
# Get a key at https://console.groq.com/keys
RAG_GROQ_API_KEY=gsk_...
RAG_GROQ_MODEL=llama-3.3-70b-versatile

# OpenRouter (diverse model selection)
RAG_OPENROUTER_API_KEY=sk-or-v1-...
RAG_OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free

# Ollama (local fallback)
RAG_OLLAMA_URL=http://localhost:11434
RAG_OLLAMA_MODEL=llama3.1:8b
```

**NER priority: Ollama → Groq → OpenRouter** (local-first for bulk extraction)

```bash
# Ollama (primary — local, fast for bulk per-paper extraction)
NER_OLLAMA_URL=http://localhost:11434
NER_OLLAMA_MODEL=llama3.1:8b

# Groq (cloud-fast fallback)
NER_GROQ_API_KEY=gsk_...
NER_GROQ_MODEL=llama-3.3-70b-versatile

# OpenRouter (cloud-diverse fallback)
NER_OPENROUTER_API_KEY=sk-or-v1-...
NER_OPENROUTER_MODEL=qwen/qwen3.6-plus:free
```

> Groq and OpenRouter both speak OpenAI-compatible chat completions, so
> swapping between them is a one-line config change. Ollama uses its own
> `/api/chat` schema, so the URL must include the host:port (no trailing
> path).

### Dictionary Matchers

Gazetteer CSV files in `backend/gazetteer/data/`:
- `chemical.csv` — 107K+ compounds
- `species.csv` — 235K+ species
- `plant_part.csv` — ~345 terms
- `analytical_technique.csv` — ~184 techniques
- `extraction_method.csv` — ~77 methods
- `development_stage.csv` — ~45 stages
- `season.csv` — ~55 terms
- `bioactivity.csv` — ~124 activities

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/paper/json` | POST | Fetch paper by DOI/PMCID/PMID (source: europepmc or openalex) |
| `/search/json` | POST | Search Europe PMC or OpenAlex |
| `/api/chat/query/stream` | POST | RAG chat — NDJSON streaming (`text_delta`, `sources`, `answer_corrected`, `citations`, `done`) |
| `/api/chat/query/json` | POST | RAG chat — non-streaming fallback |
| `/api/chat/upload/json` | POST | Upload PDFs to RAG (returns `job_id`; processing runs in the background) |
| `/api/chat/upload/status/{job_id}` | GET | Poll an upload job — `processing` / `completed` / `failed` |
| `/api/chat/upload/jobs` | GET | List active upload jobs for the current session |
| `/api/chat/files/json` | GET | List indexed files in the user's RAG corpus |
| `/api/chat/files/{name}/markdown` | GET | Extracted markdown for the citation preview panel |
| `/api/chat/files/{name}/content` | GET | Inline PDF viewer source |
| `/ner/doi/json` | POST | Standalone NER |
| `/paper/pdf` | GET | Download paper PDF |

## Testing

```bash
pytest backend/tests/ -v
```

## Troubleshooting

### Qdrant indexing errors on PDF upload

The backend recognises and auto-recovers from two Qdrant failure modes.
If you see one of these in the logs:

| Symptom in logs | What it means |
|---|---|
| `Wrong vector size` / `Wrong vector dimension` | The embedding-model dim changed since the collection was first created (e.g., switching from `bge-m3` 1024-dim to `Qwen3-Embedding-4B` 2560-dim). Each `RagService` derives a `model_suffix` hash and bakes it into the collection name (`user_{id}_{suffix}`) so a real model swap creates a fresh collection — but if you set `RAG_EMBEDDING_DIM` to a smaller MRL truncation after indexing at full dim, the points in the old collection no longer match. |
| `Collection not found` / `404` from Qdrant | The cached `QdrantVectorStore` wrapper still references a collection name that has since been dropped (e.g., manual `rm -rf data/qdrant`). The wrapper cache lives in-process so it survives the deletion. |

**Auto-recovery (already in place).** `services/rag_engine.py` does
three things to recover without manual intervention:

- The indexing retry path in `process_and_index_pdfs_with_texts` detects
  either signature (substring match on the exception message) and calls
  `_invalidate_user_collection(user_id)` to drop the stale wrapper from
  the in-process cache. If the failure looks like dimension drift, it
  also calls `_reset_user_chroma_in_place(user_id)` which delegates to
  Qdrant's `client.delete_collection(...)` — clean, atomic, no
  filesystem operations.
- The retry then re-deletes the affected sources, re-creates the
  collection on first access (`_ensure_qdrant_collection` is idempotent),
  and re-runs `add_documents`. The user sees the upload finish
  successfully on a transparent retry.
- Sparse vectors live alongside dense vectors on each point, so any
  collection reset/delete removes both atomically — no separate cache
  to invalidate and no possibility of stale BM25 serving results from
  a dropped corpus.

**Why this is simpler than the old Chroma path.** Earlier versions used
embedded ChromaDB (SQLite under the hood), which had a class of
"moved-inode readonly" failures on macOS/Linux when a directory was
deleted underneath an open connection. Qdrant's embedded client doesn't
have that pathology — it operates on its own page-locked storage and
recovers cleanly from `delete_collection` followed by re-creation.

**Manual recovery** (only if auto-recovery somehow loops):

```bash
# Stop the backend process first.
# macOS / Linux:
rm -rf data/qdrant
# Windows (PowerShell):
Remove-Item -Recurse -Force data\qdrant
# Restart the backend; the directory is recreated cleanly on next upload.
```

Only Qdrant collection storage (dense + sparse vectors) and parent
stores are dropped — uploaded PDFs in `data/uploads/` and the paper
cache in `data/cache/` are untouched.

### `Reranking failed: cannot reshape tensor of 0 elements into shape [...]`

**Cause.** The cross-encoder reranker's tokenizer produced a zero-length
sequence for at least one (query, passage) pair. PyTorch then can't infer
the `-1` dimension when reshaping a tensor of 0 elements (`8 × 0 × ? × 128`
is 0 for any value of `?`).

This happens when the query is empty / whitespace-only, or a passage
slipped through with no extractable text after tokenization.

**The code already handles this.** `rag_engine.py` does three layers of
defence:

1. **Filter out blank passages** before building rerank pairs.
2. **Skip rerank entirely** if the query string is empty.
3. **Drop pairs where either side is whitespace-only** before calling
   `reranker.predict`.

In all three cases the pipeline falls back to retrieval-order results, so
queries still return — only the rerank step is skipped. If you see this
log line, the system has already recovered; no manual action needed.

### `Error calling ollama LLM for RAG: [SSL: WRONG_VERSION_NUMBER]`

**Cause.** Almost always a URL-scheme mismatch:

- `RAG_OLLAMA_URL` is set to `https://...` but the server on the other end
  is serving plain HTTP, **or**
- `RAG_OLLAMA_URL` is set to `http://...` but the server requires HTTPS
  (e.g. behind a Cloudflare tunnel).

The TLS handshake fails because the bytes the client receives aren't a
valid TLS record.

**Fix.**

1. Check what `RAG_OLLAMA_URL` resolves to at runtime (real env vars
   override `.env` values):
   ```powershell
   $env:RAG_OLLAMA_URL
   ```
2. Match the scheme to your server:
   - **Local Ollama** (`ollama serve` on the same machine) → `http://localhost:11434`
   - **Cloudflare tunnel / hosted endpoint** → `https://<your-tunnel>.trycloudflare.com`
3. Restart the backend after changing.

The backend now logs an actionable hint when this happens — look for the
line "SSL handshake failed … Check that the URL scheme (http vs https)
matches what the server is actually serving."

### `InvalidArgument: Unexpected input data type. Actual: tensor(int32), expected: tensor(int64)` from pymupdf

**Cause.** `pymupdf 1.27.2.3`'s bundled layout submodule (`pymupdf.layout.onnx.BoxRFDGNN`) constructs `edge_index` as `int32` but its ONNX model declares the input as `tensor(int64)`. ONNX Runtime is strict about input types, so any non-trivial multi-page PDF crashes during layout analysis.

**Auto-recovery (already in place).** `rag_engine.py` has a one-time runtime patch (`_ensure_pymupdf_layout_int64_patch`) that wraps `BoxRFDGNN.predict`'s ONNX `session.run` call to coerce any int input narrower than `int64` up to `int64` before the model sees it. The patch:

- Runs once per process, idempotent via a class flag.
- Logged at INFO level on first install ("Installed pymupdf_layout int32→int64 coercion patch").
- Logs a clear warning if the upstream module path moves so a future pymupdf upgrade doesn't silently break extraction.
- Only touches the `BoxRFDGNN.predict` method — no monkey-patching of pip-installed source files (which `pip install -r requirements.txt` would wipe).

If you see this error in logs, the patch failed to apply — check the preceding log lines for the "skipping int64 patch" reason.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` | Next section |
| `←` | Previous section |
| `↑` | Scroll up 100px |
| `↓` | Scroll down 100px |
| `e` | Extract entities (on paper page) |
