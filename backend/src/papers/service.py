"""Papers service — fetch + NER + highlight orchestration (Phase 2b-2).

Extracted verbatim from `routers/paper.py::analyse_paper_json` so the router
stays HTTP-only. Query plans and response shapes are unchanged; only the
location moved.
"""
import logging
from typing import Any, Optional

from bs4 import BeautifulSoup
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.common.caching import ner_cache
from backend.src.common.highlighter import Highlighter
from backend.src.db import repository as _repo
from backend.src.papers.europe_pmc import EuropePMCService
from backend.src.papers.openalex import OpenAlexService
from backend.src.papers.resolver import fetch_doi_abstract, fetch_pmc_by_pmcid

logger = logging.getLogger(__name__)


def normalize_identifier(identifier: str) -> str:
    id_type, clean_id = EuropePMCService.parse_identifier(identifier)
    if id_type == "doi":
        return clean_id.lower()
    return clean_id


def coerce_cached_ner_payload(cached):
    if not cached:
        return [], {}
    if isinstance(cached, dict):
        return cached.get("entities", []), cached.get("summary", {})
    return cached, {}


def html_to_plain_text(html_content: str) -> str:
    """Strip HTML tags to get plain text for NER processing (matches PDF pipeline).

    Uses empty separator to preserve hyphenated words across inline tags (e.g. <em>),
    then normalizes whitespace. This prevents chemicals like
    '1,2-dioleoyl-sn-glycero-3-phosphocholine' from being broken by spaces.
    """
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, "html.parser")
    # Use empty separator so inline tags don't insert spaces into hyphenated words
    text = soup.get_text(separator='')
    # Normalize whitespace: collapse multiple spaces/newlines to single space
    return ' '.join(text.split())


async def fetch_identifier_fallback(id_type: str, clean_id: str):
    if id_type == "doi":
        return await fetch_doi_abstract(clean_id)
    if id_type == "pmcid":
        return await fetch_pmc_by_pmcid(clean_id)
    return None


async def fetch_paper_data(raw_identifier: str, source: str = "") -> tuple[str, str, dict]:
    """Multi-source fetch chain. Returns (id_type, clean_id, paper_data).

    `paper_data` may be `{"error": ..., "sections": []}` when nothing was
    found — the router returns it directly (legacy 200-with-error shape).
    """
    source = source.lower().strip() if source else ""
    id_type, clean_id = EuropePMCService.parse_identifier(raw_identifier)
    if id_type == "doi":
        clean_id = clean_id.lower()

    # If source is openalex, use OpenAlex service directly
    if source == "openalex":
        paper = await OpenAlexService.fetch_paper(clean_id)

        if paper:
            abstract = paper.get("abstract", "")
            pdf_url = paper.get("pdfUrl")
            paper_data = {
                "doi": paper.get("doi", clean_id),
                "mode": "abstract",
                "title": paper.get("title", ""),
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "journal": paper.get("journal", ""),
                "date": str(paper.get("year", "")),
                "abstract": abstract,
                "sections": [{"title": "Abstract", "content": abstract, "headings": []}] if abstract else [],
                "references": [],
                "pmcid": paper.get("pmcid", ""),
                "pmid": paper.get("pmid", ""),
                "fallback_source": "OpenAlex",
                "fallback_url": paper.get("url", ""),
                "entities": [],
                "pdfUrl": pdf_url,
                "isOpenAccess": paper.get("isOpenAccess", False),
            }
            if abstract:
                paper_data["html"] = f"<section id='section-0'><h2>Abstract</h2><p>{abstract}</p></section>"
            else:
                paper_data["html"] = "<section id='section-0'><h2>Abstract</h2><p class='text-slate-500'>Not available.</p></section>"
                paper_data["sections"] = [{"title": "Abstract", "content": "Not available.", "headings": []}]
        else:
            paper_data = {
                "doi": clean_id,
                "mode": "abstract",
                "title": "",
                "authors": [],
                "year": None,
                "journal": "",
                "date": "",
                "abstract": "",
                "sections": [],
                "references": [],
                "pmcid": "",
                "pmid": "",
                "fallback_source": "OpenAlex",
                "fallback_url": "",
                "error": "No data found for this DOI in OpenAlex.",
                "entities": [],
            }
    else:
        # Europe PMC (default) path
        paper_data = await EuropePMCService.fetch_structured_data(clean_id)

        # If Europe PMC has no content, try fallback sources (unless source is "europepmc")
        if not paper_data["sections"]:
            fallback = None
            if source != "europepmc":
                # Only try external sources if not in exclusive Europe PMC mode
                fallback = await fetch_identifier_fallback(id_type, clean_id)
            if id_type == "pmcid" and fallback and fallback.get("full_text_xml"):
                try:
                    sections, references = EuropePMCService.parse_sections_from_xml(
                        fallback["full_text_xml"], pmcid=clean_id
                    )
                    if sections:
                        fallback_year = fallback.get("year")
                        fallback_date = str(fallback_year) if fallback_year else ""
                        paper_data = {
                            "doi": fallback.get("doi", ""),
                            "mode": "full_text",
                            "title": fallback.get("title", ""),
                            "html": "",
                            "sections": sections,
                            "references": references,
                            "pmcid": fallback.get("pmcid", clean_id),
                            "fallback_source": fallback.get("source", ""),
                            "fallback_url": fallback.get("url", ""),
                            "authors": fallback.get("authors", []),
                            "year": fallback_year,
                            "journal": fallback.get("journal", ""),
                            "date": fallback_date,
                        }
                except Exception as e:
                    logger.error(f"PMCID fallback XML parsing failed for {clean_id}: {e}")

            if not paper_data.get("sections") and fallback and fallback.get("abstract"):
                abstract_text = fallback["abstract"]
                abstract_html = f"<section id='section-0'><h2>Abstract</h2><p>{abstract_text}</p></section>"
                fallback_year = fallback.get("year")
                fallback_date = str(fallback_year) if fallback_year else ""
                paper_data = {
                    "doi": fallback.get("doi", clean_id),
                    "mode": "abstract",
                    "title": fallback.get("title", ""),
                    "html": abstract_html,
                    "sections": [
                        {"title": "Abstract", "content": abstract_text}
                    ],
                    "references": {},
                    "pmcid": fallback.get("pmcid", clean_id) if id_type == "pmcid" else "",
                    "abstract": abstract_text,
                    "fallback_source": fallback.get("source", ""),
                    "fallback_url": fallback.get("url", ""),
                    "authors": fallback.get("authors", []),
                    "year": fallback_year,
                    "journal": fallback.get("journal", ""),
                    "date": fallback_date,
                    "pdfUrl": fallback.get("pdfUrl"),
                    "openAccessPdf": fallback.get("openAccessPdf"),
                    "isOpenAccess": fallback.get("isOpenAccess", False),
                }
            if not paper_data.get("sections") and fallback and fallback.get("title"):
                fallback_year = fallback.get("year")
                fallback_date = str(fallback_year) if fallback_year else ""
                paper_data = {
                    "doi": fallback.get("doi", clean_id),
                    "mode": "abstract",
                    "title": fallback["title"],
                    "html": f"<section id='section-0'><h2>Abstract</h2><p class='text-slate-500'>Not available.</p></section>",
                    "sections": [],
                    "references": {},
                    "pmcid": fallback.get("pmcid", clean_id) if id_type == "pmcid" else "",
                    "fallback_source": fallback.get("source", ""),
                    "fallback_url": fallback.get("url", ""),
                    "journal": fallback.get("journal", ""),
                    "authors": fallback.get("authors", []),
                    "year": fallback_year,
                    "date": fallback_date,
                    "pdfUrl": fallback.get("pdfUrl"),
                    "openAccessPdf": fallback.get("openAccessPdf"),
                    "isOpenAccess": fallback.get("isOpenAccess", False),
                }
            if not paper_data.get("title") and not paper_data.get("sections"):
                identifier_label = id_type.upper()
                return id_type, clean_id, {
                    "error": f"No data found for this {identifier_label}. Try viewing it on the source site.",
                    "sections": [],
                }

        # For Europe PMC exclusive mode: if no sections but title exists, show "Abstract not available" placeholder
        if source == "europepmc" and not paper_data.get("sections") and paper_data.get("title"):
            paper_data["mode"] = "abstract"
            paper_data["html"] = "<section id='section-0'><h2>Abstract</h2><p class='text-slate-500'>Not available.</p></section>"
            paper_data["fallback_source"] = "Europe PMC"
            paper_data["abstract"] = ""

        # Europe PMC has content — but title might be missing. Try to fill it ONLY if allowed
        if not paper_data.get("title") and source != "europepmc":
            fallback = await fetch_identifier_fallback(id_type, clean_id)
            if fallback and fallback.get("title"):
                paper_data["title"] = fallback["title"]

    return id_type, clean_id, paper_data


async def annotate_paper(
    paper_data: dict,
    *,
    raw_doi: str,
    clean_id: str,
    run_ner: bool,
    service,
    db: AsyncSession,
) -> tuple[list, dict, bool]:
    """NER (cache → live → DB fallback) + highlight. Mutates paper_data in place.

    Highlighting priority:
      1. NER in-memory / ner_cache hit  → entities highlighted at NER-run time
      2. run_ner=True                   → live NER, result cached
      3. paper_entities DB table        → pre-extracted entities used to highlight
         HTML so the entity-index <> navigation finds real DOM nodes
    Returns (entities, summary, is_extracted).
    """
    entities = []
    is_extracted = False
    summary = {}
    cached = service.result_cache.get(clean_id) or ner_cache.get(clean_id)
    if cached:
        entities, summary = coerce_cached_ner_payload(cached)
        is_extracted = True
    elif run_ner:
        # Process by sections for better entity locality
        sections = paper_data.get("sections", [])
        title = paper_data.get("title", "")
        abstract = paper_data.get("abstract", "")

        if sections and len(sections) > 0:
            # Europe PMC - has structured sections already
            # Build a temporary list for NER: prepend title if present (title not added to paper_data["sections"])
            ner_sections = [{"title": "Title", "content": title}] + sections if title else sections
            # Strip HTML from sections for clean NER input (matches PDF pipeline)
            plain_text_sections = []
            for s in ner_sections:
                plain_content = html_to_plain_text(s.get("content", ""))
                if plain_content.strip():
                    plain_text_sections.append({"title": s.get("title", ""), "content": plain_content})
            summary, entities = await service.process_sections(plain_text_sections)
        elif title or abstract:
            # OpenAlex/Semantic Scholar - build NER sections list: title + existing sections
            ner_sections = []
            if title:
                plain_title = html_to_plain_text(title)
                if plain_title.strip():
                    ner_sections.append({"title": "Title", "content": plain_title})
            for s in paper_data.get("sections", []):
                plain_content = html_to_plain_text(s.get("content", ""))
                if plain_content.strip():
                    ner_sections.append({"title": s.get("title", ""), "content": plain_content})
            summary, entities = await service.process_sections(ner_sections)
        else:
            # No content - skip NER
            summary, entities = {}, []
        cache_payload = {"entities": entities, "summary": summary}
        service.result_cache[clean_id] = cache_payload
        ner_cache.set(clean_id, cache_payload)
        is_extracted = True

    # ── DB fallback: use pre-extracted entities from SQLite ──────────────
    # This path is taken when the paper was imported via the ingestion
    # pipeline but the NER cache is cold (e.g. after a server restart) and
    # the caller didn't ask for a live NER run (run_ner=False).
    # We load the DB rows and use them to highlight the HTML so that the
    # entity-index sidebar and its <> navigation can find real DOM nodes.
    if not is_extracted:
        try:
            db_paper_id = await _repo.paper_id_by_doi(db, clean_id, raw_doi)
            if db_paper_id:
                db_entities = await _repo.entities_for_paper(db, db_paper_id)
                if db_entities:
                    entities = db_entities
                    is_extracted = True
                    cache_payload = {"entities": entities, "summary": summary}
                    service.result_cache[clean_id] = cache_payload
                    ner_cache.set(clean_id, cache_payload)
                    logger.info(
                        f"Loaded {len(entities)} pre-extracted entities from DB for {clean_id}"
                    )
        except Exception as db_err:
            logger.warning(f"DB entity fallback failed for {clean_id}: {db_err}")

    # Highlight title and ALL sections if NER was run or pre-extracted entities loaded
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
            # Rebuild html from highlighted sections to match frontend's fallback construction
            if paper_data.get("sections"):
                html_parts = []
                for idx, s in enumerate(paper_data["sections"]):
                    section_html = f'<section id="section-{idx}"><h2>{s.get("title", "")}</h2>{s.get("content", "")}</section>'
                    html_parts.append(section_html)
                paper_data["html"] = "".join(html_parts)
            elif paper_data.get("html"):
                paper_data["html"] = Highlighter.highlight(
                    paper_data["html"], entities
                )
        except Exception as e:
            logger.error(f"Highlighting failed for {clean_id}: {e}")

    return entities, summary, is_extracted
