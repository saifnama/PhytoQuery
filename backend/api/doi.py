"""
DOI Abstract Router
---------------------
Fallback endpoint to fetch abstracts for DOIs not found in Europe PMC.
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from backend.services.doi_resolver import fetch_doi_abstract
from backend.services.europe_pmc import EuropePMCService
from backend.core.caching import doi_cache

router = APIRouter(prefix="/doi", tags=["doi"])


def _normalize_doi(doi: str) -> str:
    normalized = doi.strip().lower()
    if normalized.startswith("https://doi.org/"):
        normalized = normalized.replace("https://doi.org/", "")
    elif normalized.startswith("http://doi.org/"):
        normalized = normalized.replace("http://doi.org/", "")
    elif normalized.startswith("doi:"):
        normalized = normalized[4:].strip()
    return normalized


def _strip_cache_metadata(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if not key.startswith("_")}


@router.get("/abstract")
async def get_doi_abstract(
    doi: str = Query(..., description="DOI to fetch abstract for"),
):
    """
    Fetch abstract for a DOI when Europe PMC doesn't have it.
    Uses DOI-oriented fallback: OpenAlex → Semantic Scholar.
    Results are cached for future requests.
    """
    id_type, clean_id = EuropePMCService.parse_identifier(doi)
    if id_type != "doi":
        raise HTTPException(status_code=400, detail="This endpoint only accepts DOI input")

    normalized_doi = _normalize_doi(clean_id)
    if not normalized_doi:
        raise HTTPException(status_code=400, detail="Invalid DOI")

    # Check cache first
    cached = doi_cache.get(normalized_doi)
    if cached:
        return JSONResponse(content=_strip_cache_metadata(cached))

    # Fetch from external sources
    result = await fetch_doi_abstract(normalized_doi)
    if not result:
        raise HTTPException(status_code=404, detail=f"No abstract found for DOI: {normalized_doi}")

    # Cache the result
    doi_cache.set(normalized_doi, result)

    return JSONResponse(content=result)
