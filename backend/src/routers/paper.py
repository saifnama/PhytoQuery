"""Paper Router — Fetch paper data by DOI with multi-source fallback."""

import ipaddress
import socket
import urllib.parse
from fastapi import APIRouter, Depends, Form, HTTPException, Query
from fastapi.responses import Response
from backend.src.papers import service as paper_service
from backend.src.papers.europe_pmc import EuropePMCService
from backend.src.ner.service import NERService
from backend.src.dependencies import get_ner_service
from backend.src.common.highlighter import Highlighter
from backend.src.common.caching import ner_cache
from backend.src.common.http_client import HttpClientManager
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from backend.src.db.session import get_db

router = APIRouter(prefix="/paper", tags=["paper"])
logger = logging.getLogger(__name__)

# pdf-proxy fetch policy: open scholarly infrastructure only.
# Intentionally narrow — only hosts where open-access PDFs are expected.
# Private-network targets are always blocked by the IP check below.
# 20 MB streaming cap (never buffer unbounded).
_PDF_PROXY_HOST_SUFFIXES = (
    # Europe PMC / EBI (open-access full text)
    "europepmc.org",
    "ebi.ac.uk",
    # NIH / PubMed Central
    "ncbi.nlm.nih.gov",
    "nih.gov",
    # OpenAlex (open metadata + OA PDFs)
    "openalex.org",
    # Unpaywall (legal OA PDF resolver)
    "unpaywall.org",
    # DOI canonical resolvers (redirect to publisher; blocked at IP level
    # if the resolved publisher is private — second layer of defence)
    "doi.org",
    "dx.doi.org",
    # Open preprint servers
    "arxiv.org",
    "biorxiv.org",
    "medrxiv.org",
)
_PDF_PROXY_MAX_BYTES = 20 * 1024 * 1024


def _assert_proxy_url_allowed(raw_url: str) -> str:
    """Validate a pdf-proxy target. Returns the URL or raises 422/403."""
    parsed = urllib.parse.urlparse(urllib.parse.unquote(raw_url))
    if parsed.scheme not in ("https", "http"):
        raise HTTPException(status_code=422, detail="Only http(s) URLs may be proxied.")
    host = (parsed.hostname or "").lower().rstrip(".")
    if not host or not any(host == s or host.endswith("." + s) for s in _PDF_PROXY_HOST_SUFFIXES):
        raise HTTPException(status_code=403, detail="Host is not an allowed scholarly source.")
    try:
        for family, _, _, _, sockaddr in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(sockaddr[0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved:
                raise HTTPException(status_code=403, detail="Private-network targets are blocked.")
                break
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=502, detail="Could not resolve target host.")
    return parsed.geturl()


def _extract_filename_from_disposition(content_disposition: str | None) -> str | None:
    if not content_disposition:
        return None
    import re

    match = re.search(r'filename="?([^";]+)"?', content_disposition)
    if match:
        return match.group(1).strip()
    return None


# --- JSON Endpoints ---


@router.post("/json")
async def analyse_paper_json(
    doi: str = Form(...),
    run_ner: bool = Form(False),
    source: str = Form(""),  # "europepmc", "openalex", or "" - case insensitive
    service: NERService = Depends(get_ner_service),
    db: AsyncSession = Depends(get_db),
):
    """JSON endpoint for fetching paper data with identifier-aware fallback.

    Highlighting priority:
      1. NER in-memory / ner_cache hit  → entities highlighted at NER-run time
      2. run_ner=True                   → live NER, result cached
      3. paper_entities DB table        → pre-extracted entities used to highlight
         HTML so the entity-index <> navigation finds real DOM nodes

    If source="openalex", skip Europe PMC fallback - only return OpenAlex metadata.
    """
    try:
        id_type, clean_id, paper_data = await paper_service.fetch_paper_data(doi, source)
        if paper_data.get("error"):
            raise HTTPException(status_code=404, detail=paper_data["error"])

        entities, summary, is_extracted = await paper_service.annotate_paper(
            paper_data, raw_doi=doi, clean_id=clean_id,
            run_ner=run_ner, service=service, db=db,
        )

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
            "isOpenAccess": paper_data.get("isOpenAccess"),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Paper JSON Error: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.post("/section/json")
async def switch_section_json(
    doi: str = Form(...),
    section_idx: int = Form(...),
    service: NERService = Depends(get_ner_service),
):
    """JSON endpoint for switching sections."""
    try:
        clean_id = paper_service.normalize_identifier(doi)
        paper_data = await EuropePMCService.fetch_structured_data(clean_id)
        sections = paper_data["sections"]
        if section_idx >= len(sections):
            return {"error": "Section not found"}

        current_section = sections[section_idx]

        cached = service.result_cache.get(clean_id) or ner_cache.get(clean_id)
        entities, _ = paper_service.coerce_cached_ner_payload(cached)
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
        clean_id = paper_service.normalize_identifier(identifier)
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
        if len(upstream.content) > _PDF_PROXY_MAX_BYTES:
            raise HTTPException(status_code=413, detail="PDF exceeds 20 MB proxy cap.")

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
    """Proxy PDF download from an allowed scholarly URL (bypasses CORS)."""
    try:
        decoded_url = _assert_proxy_url_allowed(url)
        logger.info(f"[pdf-proxy] Fetching: {decoded_url[:120]}")

        client = await HttpClientManager.get_client()
        # Use browser-like headers to avoid publisher blocks
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "application/pdf",
            "Accept-Language": "en-US,en;q=0.9",
        }
        chunks: list[bytes] = []
        total = 0
        async with client.stream(
            "GET", decoded_url, follow_redirects=True, timeout=60.0, headers=headers
        ) as resp:
            resp.raise_for_status()
            async for chunk in resp.aiter_bytes(65536):
                total += len(chunk)
                if total > _PDF_PROXY_MAX_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds 20 MB proxy cap.")
                chunks.append(chunk)
        content = b"".join(chunks)
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
        from backend.src.db import repository as _repo

        total_count, paper_list = await _repo.list_papers(
            db, limit=limit, offset=offset, country=country, query=query, year=year
        )
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
    happens on the frontend via the dictionary CSVs at render time).
    """
    try:
        from backend.src.db import repository as _repo

        clean_id = paper_service.normalize_identifier(doi)
        paper_id = await _repo.paper_id_by_doi(db, clean_id, doi)
        if not paper_id:
            return {"entities": []}
        formatted_entities = await _repo.entities_for_paper(db, paper_id)
        return {"paper_id": paper_id, "doi": clean_id, "entities": formatted_entities}
    except Exception as e:
        logger.error(f"Error fetching entities for {doi}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
