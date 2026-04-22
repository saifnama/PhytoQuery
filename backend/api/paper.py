"""Paper Router — Fetch paper data by DOI with multi-source fallback."""

from fastapi import APIRouter, Depends, Form
from backend.services.europe_pmc import EuropePMCService
from backend.services.doi_resolver import fetch_doi_abstract
from backend.services.ner_engine import ner_service, NERService
from backend.core.highlighter import Highlighter
from backend.core.caching import ner_cache
import logging

router = APIRouter(prefix="/paper", tags=["paper"])
logger = logging.getLogger(__name__)


def _normalize_identifier(identifier: str) -> str:
    id_type, clean_id = EuropePMCService.parse_identifier(identifier)
    if id_type == "doi":
        return clean_id.lower()
    return clean_id


def _coerce_cached_ner_payload(cached):
    if not cached:
        return [], {}
    if isinstance(cached, dict):
        return cached.get("entities", []), cached.get("summary", {})
    return cached, {}


def get_ner_service() -> NERService:
    return ner_service


def get_pmc_service():
    return EuropePMCService


# --- JSON Endpoints ---


@router.post("/json")
async def analyze_paper_json(
    doi: str = Form(...),
    run_ner: bool = Form(False),
    service: NERService = Depends(get_ner_service),
):
    """
    JSON endpoint for fetching paper data with sections.
    Falls back to OpenAlex/Semantic Scholar/PubMed if Europe PMC doesn't have it.
    """
    try:
        id_type, clean_id = EuropePMCService.parse_identifier(doi)
        if id_type == "doi":
            clean_id = clean_id.lower()
        paper_data = await EuropePMCService.fetch_structured_data(clean_id)

        if not paper_data["sections"]:
            # Europe PMC doesn't have it — try external sources
            logger.info(
                f"Europe PMC has no content for {clean_id}, trying external sources..."
            )
            fallback = await fetch_doi_abstract(clean_id)
            if fallback and fallback.get("abstract"):
                abstract_html = f"<section id='section-0'><h2>Abstract</h2><p>{fallback['abstract']}</p></section>"
                fallback_year = fallback.get("year")
                fallback_date = str(fallback_year) if fallback_year else ""
                return {
                    "doi": fallback.get("doi", clean_id),
                    "mode": "abstract",
                    "title": fallback.get("title", ""),
                    "html": abstract_html,
                    "sections": [
                        {"title": "Abstract", "content": fallback["abstract"]}
                    ],
                    "references": {},
                    "pmcid": "",
                    "entities": [],
                    "summary": {},
                    "is_extracted": False,
                    "fallback_source": fallback.get("source", ""),
                    "fallback_url": fallback.get("url", ""),
                    "authors": fallback.get("authors", []),
                    "year": fallback_year,
                    "journal": fallback.get("journal", ""),
                    "date": fallback_date,
                }
            if fallback and fallback.get("title"):
                fallback_year = fallback.get("year")
                fallback_date = str(fallback_year) if fallback_year else ""
                return {
                    "doi": fallback.get("doi", clean_id),
                    "mode": "abstract",
                    "title": fallback["title"],
                    "html": f"<section id='section-0'><h2>Abstract</h2><p class='text-slate-500'>Not available.</p></section>",
                    "sections": [],
                    "references": {},
                    "pmcid": "",
                    "entities": [],
                    "summary": {},
                    "is_extracted": False,
                    "fallback_source": fallback.get("source", ""),
                    "fallback_url": fallback.get("url", ""),
                    "journal": fallback.get("journal", ""),
                    "authors": fallback.get("authors", []),
                    "date": fallback_date,
                }
            return {
                "error": f"No data found for this DOI. Try viewing on publisher.",
                "sections": [],
            }

        # Europe PMC has content — but title might be missing. Try to fill it.
        if not paper_data.get("title"):
            fallback = await fetch_doi_abstract(clean_id)
            if fallback and fallback.get("title"):
                paper_data["title"] = fallback["title"]

        entities = []
        is_extracted = False
        summary = {}
        cached = service.result_cache.get(clean_id) or ner_cache.get(clean_id)
        if cached:
            entities, summary = _coerce_cached_ner_payload(cached)
            is_extracted = True
        elif run_ner:
            # Include title + sections for NER
            title = paper_data.get("title", "")
            sections_text = "\n\n".join([s["content"] for s in paper_data["sections"]])
            full_text = f"{title}\n\n{sections_text}" if title else sections_text
            summary, entities = await service.process_text(full_text)
            cache_payload = {"entities": entities, "summary": summary}
            service.result_cache[clean_id] = cache_payload
            ner_cache.set(clean_id, cache_payload)
            is_extracted = True

        # Highlight title and ALL sections if NER was run
        if is_extracted and entities:
            try:
                # Highlight title
                if paper_data.get("title"):
                    paper_data["title"] = Highlighter.highlight(
                        paper_data["title"], entities
                    )
                # Highlight all sections
                for section in paper_data["sections"]:
                    section["content"] = Highlighter.highlight(
                        section["content"], entities
                    )
                logger.info(
                    f"Successfully highlighted {len(paper_data['sections'])} sections for {clean_id}"
                )
            except Exception as e:
                logger.error(f"Highlighting failed for {clean_id}: {e}")

        return {
            "doi": paper_data.get("doi", clean_id),
            "mode": paper_data["mode"],
            "title": paper_data.get("title", ""),
            "sections": paper_data["sections"],
            "references": paper_data.get("references", {}),
            "pmcid": paper_data.get("pmcid", ""),
            "entities": entities,
            "summary": summary,
            "is_extracted": is_extracted,
            "journal": paper_data.get("journal", ""),
            "authors": paper_data.get("authors", []),
            "date": paper_data.get("date", ""),
            "fallback_source": "Europe PMC",
            "fallback_url": f"https://europepmc.org/article/{clean_id}",
        }
    except Exception as e:
        logger.error(f"Paper JSON Error: {e}")
        return {"error": str(e), "sections": []}


@router.post("/section/json")
async def switch_section_json(
    doi: str = Form(...),
    section_idx: int = Form(...),
    service: NERService = Depends(get_ner_service),
):
    """JSON endpoint for switching sections."""
    try:
        clean_id = _normalize_identifier(doi)
        paper_data = await EuropePMCService.fetch_structured_data(clean_id)
        sections = paper_data["sections"]
        if section_idx >= len(sections):
            return {"error": "Section not found"}

        current_section = sections[section_idx]

        cached = service.result_cache.get(clean_id) or ner_cache.get(clean_id)
        entities, _ = _coerce_cached_ner_payload(cached)
        if not entities:
            entities = []

        try:
            highlighted = Highlighter.highlight(current_section["content"], entities)
        except Exception as e:
            logger.error(f"Section highlighting failed: {e}")
            highlighted = current_section["content"]

        return {
            "content": current_section["content"],
            "highlighted": highlighted,
        }
    except Exception as e:
        logger.error(f"Section switch error: {e}")
        return {"error": str(e)}
