"""Paper Router — Fetch paper data by DOI with multi-source fallback."""

import json
import urllib.parse
from fastapi import APIRouter, Depends, Form, HTTPException, Query
from fastapi.responses import Response
from backend.services.europe_pmc import EuropePMCService
from backend.services.openalex import OpenAlexService
from backend.services.doi_resolver import (
    fetch_doi_abstract,
    fetch_pmc_by_pmcid,
)
from backend.services.ner_engine import ner_service, NERService
from backend.core.highlighter import Highlighter
from backend.core.caching import ner_cache
from backend.core.http_client import HttpClientManager
from bs4 import BeautifulSoup
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc
from backend.db.database import get_db
from backend.db.models import Paper, PaperEntity

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


def _maybe_json_load(value):
    """JSON columns may surface as either parsed objects or raw strings depending
    on the SQLite driver / type-affinity path. Normalize to Python objects."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return value


def _entity_row_to_dict(pe: PaperEntity) -> dict:
    """Convert a ``PaperEntity`` row into the API entity dict shape.

    The frontend depends on the ``{label, text, count, metadata}`` keys;
    ``canonical`` and ``aliases`` are kept as a superset for callers that
    need them. There is no longer a separate ``Entity`` catalog table to
    join against — entity-type metadata (SMILES for chemicals, taxonomy
    for species, etc.) lives either in the ``metadata`` JSON column on
    this row or in the gazetteer CSV files (resolved at render time on
    the frontend). We pass through whatever happens to be in the JSON
    payload here; if there is none, ``metadata`` is an empty dict.
    """
    meta_raw = _maybe_json_load(pe.meta)
    metadata: dict = meta_raw if isinstance(meta_raw, dict) else {}

    aliases = metadata.pop("aliases", None) if isinstance(metadata, dict) else None
    if not isinstance(aliases, list):
        aliases = []

    # Flatten metadata into the top-level entity dict. The frontend Entity
    # type reads flat fields like entity.accepted_scientific_name,
    # entity.smiles, etc. — not nested under entity.metadata.
    return {
        "label": pe.label,
        "text": pe.canonical_text,
        "canonical": pe.canonical_text,
        "count": pe.frequency,
        "aliases": aliases,
        **metadata,
    }


def get_ner_service() -> NERService:
    return ner_service


def get_pmc_service():
    return EuropePMCService


def _extract_filename_from_disposition(content_disposition: str | None) -> str | None:
    if not content_disposition:
        return None
    import re

    match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if match:
        return match.group(1).strip()
    return None


def _html_to_plain_text(html_content: str) -> str:
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


async def _fetch_identifier_fallback(id_type: str, clean_id: str):
    if id_type == "doi":
        return await fetch_doi_abstract(clean_id)
    if id_type == "pmcid":
        return await fetch_pmc_by_pmcid(clean_id)
    return None


# --- JSON Endpoints ---


@router.post("/json")
async def analyze_paper_json(
    doi: str = Form(...),
    run_ner: bool = Form(False),
    source: str = Form(""),  # "europepmc", "openalex", or "" - case insensitive
    service: NERService = Depends(get_ner_service),
):
    """JSON endpoint for fetching paper data with identifier-aware fallback.
    
    If source="openalex", skip Europe PMC fallback - only return OpenAlex metadata.
    """
    # Normalize source: "Europe PMC" → "europepmc", "OpenAlex" → "openalex"
    source = source.lower().strip() if source else ""
    
    try:
        id_type, clean_id = EuropePMCService.parse_identifier(doi)
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
                }
                if abstract:
                    paper_data["html"] = f"<section id='section-0'><h2>Abstract</h2><p>{abstract}</p></section>"
                elif paper_data.get("title"):
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
                    fallback = await _fetch_identifier_fallback(id_type, clean_id)
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
                    }
                if not paper_data.get("title") and not paper_data.get("sections"):
                    identifier_label = id_type.upper()
                    return {
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
                # Build a temporary list for NER: prepend title if present (title not added to paper_data["sections"])
                ner_sections = [{"title": "Title", "content": title}] + sections if title else sections
                # Strip HTML from sections for clean NER input (matches PDF pipeline)
                plain_text_sections = []
                for s in ner_sections:
                    plain_content = _html_to_plain_text(s.get("content", ""))
                    if plain_content.strip():
                        plain_text_sections.append({"title": s.get("title", ""), "content": plain_content})
                summary, entities = await service.process_sections(plain_text_sections)
            elif title or abstract:
                # OpenAlex/Semantic Scholar - build NER sections list: title + existing sections
                ner_sections = []
                if title:
                    plain_title = _html_to_plain_text(title)
                    if plain_title.strip():
                        ner_sections.append({"title": "Title", "content": plain_title})
                for s in paper_data.get("sections", []):
                    plain_content = _html_to_plain_text(s.get("content", ""))
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
                # Rebuild html from highlighted sections to match frontend's fallback construction
                if paper_data.get("sections"):
                    html_parts = []
                    for idx, s in enumerate(paper_data["sections"]):
                        section_html = f'<section id="section-{idx}"><h2>{s.get("title", "")}</h2>{s.get("content", "")}</section>'
                        html_parts.append(section_html)
                    paper_data["html"] = "".join(html_parts)
            except Exception as e:
                logger.error(f"Highlighting failed for {clean_id}: {e}")

        return {
            "doi": paper_data.get("doi", clean_id),
            "html": paper_data.get("html", ""),
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
            "fallback_source": paper_data.get("fallback_source", "Europe PMC"),
            "fallback_url": paper_data.get(
                "fallback_url", f"https://europepmc.org/article/{clean_id}"
            ),
            "pdfUrl": paper_data.get("pdfUrl"),
            "openAccessPdf": paper_data.get("openAccessPdf"),
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


@router.get("/pdf")
async def download_paper_pdf(identifier: str = Query(...)):
    """Resolve and stream a paper PDF when one is available."""
    try:
        clean_id = _normalize_identifier(identifier)
        pdf_info = await EuropePMCService.resolve_pdf_url(clean_id)
        if not pdf_info:
            raise HTTPException(
                status_code=404,
                detail="No downloadable PDF found for this paper.",
            )

        client = await HttpClientManager.get_client()
        upstream = await client.get(
            pdf_info["url"], follow_redirects=True, timeout=60.0
        )
        upstream.raise_for_status()

        if not upstream.content:
            raise HTTPException(status_code=404, detail="PDF response was empty.")

        content_disposition = upstream.headers.get("content-disposition")
        filename = (
            _extract_filename_from_disposition(content_disposition)
            or pdf_info.get("filename")
            or "paper.pdf"
        )

        return Response(
            content=upstream.content,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "X-PDF-Source": pdf_info.get("source", ""),
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF download error for {identifier}: {e}")
        raise HTTPException(status_code=502, detail="Failed to fetch paper PDF.")


@router.get("/pdf-proxy")
async def proxy_pdf(url: str = Query(...)):
    """Proxy PDF download from external URL (bypasses CORS)."""
    try:
        decoded_url = urllib.parse.unquote(url)
        logger.info(f"[pdf-proxy] Fetching: {decoded_url[:120]}")
        
        client = await HttpClientManager.get_client()
        # Use browser-like headers to avoid publisher blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf",
            "Accept-Language": "en-US,en;q=0.9",
        }
        resp = await client.get(decoded_url, follow_redirects=True, timeout=60.0, headers=headers)
        resp.raise_for_status()

        content = resp.content
        if not content:
            raise HTTPException(status_code=404, detail="Empty PDF response.")

        content_disposition = resp.headers.get("content-disposition")
        filename = _extract_filename_from_disposition(content_disposition) or "paper.pdf"
        media_type = resp.headers.get("content-type", "application/pdf")

        return Response(
            content=content,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{filename}"'},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF proxy error for url={url[:100]}: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to fetch PDF: {str(e)}")


@router.get("/db/list")
async def list_papers(
    limit: int = Query(50, ge=1, le=500), 
    offset: int = Query(0, ge=0), 
    country: str = Query(None),
    query: str = Query(None),
    year: int = Query(None),
    db: AsyncSession = Depends(get_db)
):
    """Fetch a paginated list of papers from the local SQLite database."""
    try:
        from sqlalchemy import func
        
        # Base queries
        select_stmt = select(Paper)
        
        if country:
            country_expr = func.json_extract(PaperEntity.meta, "$.country")
            select_stmt = select_stmt.join(PaperEntity).where(PaperEntity.label == "LOCATION").where(country_expr == country)
            
        if query:
            select_stmt = select_stmt.where(
                Paper.title.ilike(f"%{query}%") | Paper.journal.ilike(f"%{query}%")
            )

        if year is not None:
            select_stmt = select_stmt.where(Paper.year == year)
            
        # Distinct/Group by to avoid duplicates
        select_stmt = select_stmt.group_by(Paper.id).order_by(desc(Paper.id))
        
        # Execute to get list
        result = await db.execute(select_stmt.limit(limit).offset(offset))
        papers = result.scalars().all()
        
        # Count query
        if country or query or year is not None:
            subq = select_stmt.limit(None).offset(None).subquery()
            count_stmt = select(func.count()).select_from(subq)
        else:
            count_stmt = select(func.count(Paper.id))
            
        count_result = await db.execute(count_stmt)
        total_count = count_result.scalar() or 0
        
        paper_list = [
            {
                "id": p.id, 
                "doi": p.doi, 
                "title": p.title, 
                "journal": p.journal, 
                "year": p.year,
                "is_open_access": p.is_open_access,
                "entity_count": p.entity_count
            } 
            for p in papers
        ]
        
        return {"total": total_count, "papers": paper_list}
    except Exception as e:
        logger.error(f"Error fetching paper list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/db/{doi:path}/entities")
async def get_paper_entities(doi: str, db: AsyncSession = Depends(get_db)):
    """Fetch pre-extracted entities for a paper from the SQLite database
    using its DOI.

    Returns ``{label, text, canonical, count, metadata, aliases}`` per
    entity. ``metadata`` is whatever was stored in the ``paper_entities.
    metadata`` JSON column at ingest time (chemical/species enrichment
    happens on the frontend via the gazetteer CSVs at render time).
    """
    try:
        clean_id = _normalize_identifier(doi)

        # 1. Find paper ID by DOI (try normalized first, then raw).
        result = await db.execute(select(Paper.id).where(Paper.doi == clean_id))
        paper_id = result.scalar_one_or_none()

        if not paper_id and clean_id != doi:
            result = await db.execute(select(Paper.id).where(Paper.doi == doi))
            paper_id = result.scalar_one_or_none()

        if not paper_id:
            return {"entities": []}

        # 2. Single-table query — entity-type metadata is on the row.
        result = await db.execute(
            select(PaperEntity).where(PaperEntity.paper_id == paper_id)
        )
        rows = result.scalars().all()

        formatted_entities = [_entity_row_to_dict(pe) for pe in rows]

        return {"paper_id": paper_id, "doi": clean_id, "entities": formatted_entities}
    except Exception as e:
        logger.error(f"Error fetching entities for {doi}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
