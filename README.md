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
| RAG | LangChain, ChromaDB, sentence-transformers |
| PDF | pymupdf (fitz), BeautifulSoup |
| Graph | vis-network |
| Sanitization | nh3 (server), DOMPurify (client) |
| Paper Sources | Europe PMC API, OpenAlex API |
| LLM | OpenRouter / Ollama |

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

#### NER (Named Entity Recognition)
```bash
# Set OpenRouter key to enable LLM extraction
export NER_OPENROUTER_API_KEY="sk-or-..."

# Or use local Ollama
export NER_OLLAMA_URL="http://localhost:11434"
export NER_OLLAMA_MODEL="llama3.1:8b"
```

#### RAG (Chat)
```bash
export RAG_OPENROUTER_API_KEY="sk-or-..."

# Optional: local Ollama fallback
export RAG_OLLAMA_URL="http://localhost:11434"
export RAG_OLLAMA_MODEL="llama3.1:8b"
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

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `→` | Next section |
| `←` | Previous section |
| `↑` | Scroll up 100px |
| `↓` | Scroll down 100px |
| `e` | Extract entities (on paper page) |

## License

MIT