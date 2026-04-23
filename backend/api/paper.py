"""Paper Router — Fetch paper data by DOI with multi-source fallback."""

from fastapi import APIRouter, Depends, Form
from backend.services.europe_pmc import EuropePMCService
from backend.services.doi_resolver import (
    fetch_doi_abstract,
    fetch_pubmed_by_pmid,
    fetch_pmc_by_pmcid,
)
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


async def _fetch_identifier_fallback(id_type: str, clean_id: str):
    if id_type == "doi":
        return await fetch_doi_abstract(clean_id)
    if id_type == "pmid":
        return await fetch_pubmed_by_pmid(clean_id)
    if id_type == "pmcid":
        return await fetch_pmc_by_pmcid(clean_id)
    return None


# --- JSON Endpoints ---


@router.post("/json")
async def analyze_paper_json(
    doi: str = Form(...),
    run_ner: bool = Form(False),
    service: NERService = Depends(get_ner_service),
):
    """
    JSON endpoint for fetching paper data with identifier-aware fallback.

    - DOI: Europe PMC first, then OpenAlex/Semantic Scholar
    - PMID: Europe PMC first, then PubMed
    - PMCID: Europe PMC/PMC first, then direct PMC fetch
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
            fallback = await _fetch_identifier_fallback(id_type, clean_id)
            if id_type == "pmcid" and fallback and fallback.get("full_text_xml"):
                try:
                    sections, references = EuropePMCService.parse_sections_from_xml(
                        fallback["full_text_xml"], pmcid=clean_id
                    )
                    if sections:
                        fallback_year = fallback.get("year")
                        fallback_date = str(fallback_year) if fallback_year else ""
                        return {
                            "doi": fallback.get("doi", ""),
                            "mode": "full_text",
                            "title": fallback.get("title", ""),
                            "html": "",
                            "sections": sections,
                            "references": references,
                            "pmcid": fallback.get("pmcid", clean_id),
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
                except Exception as e:
                    logger.error(f"PMCID fallback XML parsing failed for {clean_id}: {e}")

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
                    "pmcid": fallback.get("pmcid", clean_id) if id_type == "pmcid" else "",
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
                    "pmcid": fallback.get("pmcid", clean_id) if id_type == "pmcid" else "",
                    "entities": [],
                    "summary": {},
                    "is_extracted": False,
                    "fallback_source": fallback.get("source", ""),
                    "fallback_url": fallback.get("url", ""),
                    "journal": fallback.get("journal", ""),
                    "authors": fallback.get("authors", []),
                    "date": fallback_date,
                }
            identifier_label = id_type.upper()
            return {
                "error": f"No data found for this {identifier_label}. Try viewing it on the source site.",
                "sections": [],
            }

        # Europe PMC has content — but title might be missing. Try to fill it.
        if not paper_data.get("title"):
            fallback = await _fetch_identifier_fallback(id_type, clean_id)
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
            # Process by sections for better entity locality
            sections = paper_data.get("sections", [])
            title = paper_data.get("title", "")
            abstract = paper_data.get("abstract", "")

            if sections and len(sections) > 0:
                # Europe PMC - has structured sections already
                # Add title as first section if present
                if title:
                    sections = [{"title": "Title", "content": title}] + sections
                summary, entities = await service.process_sections(sections)
            elif title or abstract:
                # OpenAlex/Semantic Scholar - build sections from title + abstract
                sections = []
                if title:
                    sections.append({"title": "Title", "content": title})
                if abstract:
                    sections.append({"title": "Abstract", "content": abstract})
                summary, entities = await service.process_sections(sections)
            else:
                # No content - skip NER
                summary, entities = {}, []
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
