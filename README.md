# PhytoQuery

## 1. Project Overview

PhytoQuery is a research paper reader that retrieves papers from Europe PMC, parses JATS XML, and renders the content with Named Entity Recognition (NER) highlights. The project uses a FastAPI backend, React frontend with Tailwind CSS, and DOMPurify for HTML sanitization.

## 2. Features

### Paper Reader
- Auto-generated Table of Contents with 2-level hierarchy (H2 → H3)
- Continuous smooth scrolling — all sections rendered inline for seamless reading
- Scroll-spy highlighting — active section is reflected in the TOC as you scroll
- Wide tables displayed with horizontal scrolling

### Search
- Search Europe PMC by keywords, DOI, PMCID, or PMID
- Filter by Open Access, Full Text, Article Type
- Sort by Citations or Date

### Chat / RAG
- Upload PDFs and query them with AI
- Inline PDF viewer — click any uploaded PDF to view it in the side panel
- Citations with source references

### Named Entity Recognition (NER)
- Extract entities from papers with dictionary-backed and LLM-assisted NER.
- Dictionary-backed entity types currently include `PLANT PART`, `ANALYTICAL TECHNIQUE`, `EXTRACTION METHOD`, `DEVELOPMENT STAGE`, and `SEASON`.
- Gazetteer entities are loaded from CSV dictionaries, matched with aliases, and normalized to canonical terms before sidebar grouping.
- Click "Find Key Terms" to run NER extraction.
- Entities are grouped by type in the sidebar and highlighted inline in the paper view.

### Security
- HTML sanitization via nh3 (server-side) and DOMPurify (client-side)
- XSS protection — script tags, inline handlers, and javascript: URLs are blocked

## 3. Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `→` | Smooth scroll to next section |
| `←` | Smooth scroll to previous section |
| `↑` | Scroll up 100px |
| `↓` | Scroll down 100px |

## 4. Configuration

### Separate Config Files

- `backend/config_ner.py` - NER settings (Ollama → OpenRouter)
- `backend/config_rag.py` - RAG settings (OpenRouter → Ollama)

### Environment Variables

```bash
# NER config_ner.py (Ollama primary, OpenRouter fallback)
export NER_OLLAMA_URL="https://..."
export NER_OLLAMA_MODEL="llama3.1:8b"
export NER_OPENROUTER_API_KEY="sk-or-..."
export NER_OPENROUTER_MODEL="anthropic/claude-3-haiku:free"
export NER_CONFIDENCE_THRESHOLD=0.7
export NER_CHUNK_SIZE_WORDS=1000

# RAG config_rag.py (OpenRouter primary, Ollama fallback)
export RAG_OPENROUTER_API_KEY="sk-or-..."
export RAG_OPENROUTER_MODEL="stepfun/step-3.5-flash:free"
export RAG_OLLAMA_URL="https://..."
export RAG_OLLAMA_MODEL="llama3.1:8b"
export RAG_TEMPERATURE=0.1
export RAG_TOP_K=10
```

Dictionary-backed NER does not introduce additional `NER_*` environment variables. Gazetteer behavior is driven by the CSV files in `backend/gazetteer/data/` and their matcher modules in `backend/gazetteer/`.

## 5. Testing

### Run Tests
```bash
cd PhytoQuery
phytovenv\Scripts\python -m pytest backend/tests/ -v
```

### Test Scope
- Backend tests live in `backend/tests/` and use `pytest` with async support from `backend/pytest.ini`.
- Dictionary and NER pipeline coverage includes matcher alias normalization, canonical term mapping, highlighter class mapping, and end-to-end extraction checks in `backend/tests/test_dictionary_ner_pipeline.py`.
- Existing backend tests also cover configuration, caching, sanitization, TOC generation, and RAG chunking.

## 5. Quick Start

### Build Frontend (first time)
```bash
cd frontend
bun install
bun run build
```

### Run Backend
```bash
cd PhytoQuery
phytovenv\Scripts\activate
uvicorn backend.app:app --reload
```

Or directly:
```bash
cd PhytoQuery
phytovenv\Scripts\activate
python -m backend.app
```

Open browser at `http://localhost:8000`

## 6. Tech Stack

- **Backend:** FastAPI (Python)
- **Frontend:** React + Tailwind CSS + Vite + Phosphor Icons
- **Sanitization:** nh3 (server) + DOMPurify (client)
- **Paper Source:** Europe PMC API
- **LLM:** Ollama, OpenRouter
- **Vector Store:** ChromaDB

## 7. Project Structure

```
PhytoQuery/
├── backend/
│   ├── api/                    # API endpoints
│   │   ├── paper.py
│   │   ├── search.py
│   │   ├── ner.py
│   │   ├── rag.py
│   │   ├── doi.py
│   │   └── health.py
│   ├── core/                   # Utilities
│   │   ├── caching.py
│   │   ├── sanitizer.py
│   │   ├── highlighter.py
│   │   └── http_client.py
│   ├── schemas/                # Pydantic models
│   │   └── schemas.py
│   ├── services/
│   │   ├── europe_pmc/        # Paper fetching & parsing
│   │   ├── ner_engine.py      # NER extraction
│   │   ├── rag_engine.py      # RAG chat
│   │   └── doi_resolver.py    # DOI fallback sources
│   ├── gazetteer/             # Dictionary-backed NER matchers and CSV data
│   │   ├── data/              # Gazetteer CSV dictionaries
│   │   ├── *_matcher.py       # PhraseMatcher-backed entity matchers
│   │   └── build_matcher.py   # Gazetteer build/cache helpers
│   └── tests/
├── data/                       # Runtime data
│   ├── cache/                  # Paper & NER cache
│   ├── chroma_db/              # Vector store
│   └── uploads/                # Uploaded PDFs
├── frontend/src/
│   ├── features/
│   │   ├── search/             # Search page
│   │   ├── reader/             # Paper reader
│   │   └── chat/               # RAG chat
│   ├── layout/                 # Header, Sidebar
│   ├── ui/                     # ErrorBoundary
│   ├── lib/                    # API client
│   └── types/
└── README.md
```
