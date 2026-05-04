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
- Inline PDF viewer
- Upload to RAG directly from paper (stays on page, no navigation)
- Citations with source references
- **Dual parser**: PyMuPDF (fast) or Docling (detailed, structure-preserving)
- **Hybrid retrieval**: Vector search + BM25 keyword matching with Reciprocal Rank Fusion
- **Instruction-Aware Architecture**: Custom domain prompts for both embedding (Qwen3) and reranking (zerank-2)
- **MRL Truncation**: Storage-efficient vectors (1024 dims) using Matryoshka Representation Learning
- **Lazy model loading**: RAG models only load on first use (~200MB startup)

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
| Frontend | React 19, TypeScript, Vite, Tailwind CSS |
| NLP | spaCy PhraseMatcher (dictionary-backed) |
| Embeddings | Qwen3-Embedding-4B (primary), BAAI/bge-m3 (fallback) |
| Reranker | zeroentropy/zerank-2 (CrossEncoder) |
| RAG | LangChain, ChromaDB, sentence-transformers, rank-bm25 |
| PDF Parsing | Docling (detailed), PyMuPDF/fitz (fast) |
| Graph | vis-network |
| Sanitization | nh3 (server), DOMPurify (client) |
| Paper Sources | Europe PMC API, OpenAlex API |
| LLM | OpenRouter / Ollama |
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
│   │   ├── features/  # Page components
│   │   ├── layout/   # Header, Sidebar
│   │   └── lib/     # API client
│   └── dist/        # Built output
├── data/             # Runtime data
│   ├── cache/       # Paper & NER cache
│   ├── chroma_db/  # Vector store
│   └── uploads/     # Uploaded PDFs
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

# Option 1: Via uvicorn (recommended)
# From PhytoQuery directory:
cd C:\Users\saif\saifnama_lab\PhytoQuery
python -m uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

# Option 2: Direct Python
cd C:\Users\saif\saifnama_lab\PhytoQuery
python -m backend.app --host 0.0.0.0 --port 8000
```

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
```bash
# OpenRouter (primary)
RAG_OPENROUTER_API_KEY=sk-or-v1-...
RAG_OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free

# Ollama (fallback — runs locally)
RAG_OLLAMA_URL=http://localhost:11434
RAG_OLLAMA_MODEL=llama3.1:8b
```

#### NER Providers
```bash
# OpenRouter
NER_OPENROUTER_API_KEY=sk-or-v1-...

# Ollama
NER_OLLAMA_URL=http://localhost:11434
NER_OLLAMA_MODEL=llama3.1:8b
```

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
| `/api/chat/query/json` | POST | RAG chat |
| `/api/chat/upload/json` | POST | Upload PDF to RAG |
| `/ner/doi/json` | POST | Standalone NER |
| `/paper/pdf` | GET | Download paper PDF |

## Testing

```bash
pytest backend/tests/ -v
```

## Troubleshooting

### `Could not connect to tenant default_tenant` on PDF upload

**Symptom** in backend logs:
```
Indexing failed for user sess_<id>; ... retrying once.
Upload job <uuid> failed: Could not connect to tenant default_tenant. Are you sure it exists
```

**Cause.** This is a chromadb-1.x error. The per-user persist directory at
`data/chroma_db/<sanitized_user_id>/` contains a SQLite file whose `tenants`
table is missing or empty. It happens in two situations:

1. **Interrupted init.** chromadb 1.x's `PersistentClient` writes the schema
   in stages — if the process is killed mid-init (or SQLite file locks
   contend on Windows), the directory is left with `chroma.sqlite3` present
   but no `default_tenant` row.
2. **Schema downgrade.** The directory was created by an older chromadb
   (pre-1.0) that didn't have the tenant model, and chromadb was then
   upgraded. The migration doesn't always backfill `default_tenant`.

**The code already handles this.** `services/rag_engine.py` does two things
to recover automatically:

- `_get_user_collection` constructs an explicit `chromadb.PersistentClient`
  rather than using `Chroma(persist_directory=…)` implicitly. The explicit
  client deterministically creates `default_tenant` and `default_database`
  on first init.
- `process_and_index_pdfs_with_texts`'s retry path detects the tenant
  error string (`"tenant"`/`"database"` in the exception message) and
  **wipes the user's persist directory** before retrying. Without the wipe
  the retry would re-open the same broken SQLite.

**Manual recovery** (if logs show the error persisting after a retry):

```powershell
# Stop the backend process first.
Remove-Item -Recurse -Force C:\Users\saif\saifnama_lab\PhytoQuery\data\chroma_db
# Restart the backend; the directory is recreated cleanly on next upload.
```

Only the user's vector indexes are dropped — uploaded PDFs in
`data/uploads/` and the paper cache in `data/cache/` are untouched.

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

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` | Next section |
| `←` | Previous section |
| `↑` | Scroll up 100px |
| `↓` | Scroll down 100px |
| `e` | Extract entities (on paper page) |
