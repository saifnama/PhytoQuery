"""
DOI Abstract Router
---------------------
Fallback endpoint to fetch abstracts for DOIs not found in Europe PMC.
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
from backend.services.doi_resolver import fetch_doi_abstract
from backend.core.caching import doi_cache

router = APIRouter(prefix="/doi", tags=["doi"])


@router.get("/abstract")
async def get_doi_abstract(
    doi: str = Query(..., description="DOI to fetch abstract for"),
):
    """
    Fetch abstract for a DOI when Europe PMC doesn't have it.
    Uses multi-source fallback: OpenAlex → Semantic Scholar → PubMed.
    Results are cached for future requests.
    """
    # Check cache first
    cached = doi_cache.get(doi)
    if cached:
        return JSONResponse(content=cached)

    # Fetch from external sources
    result = await fetch_doi_abstract(doi)
    if not result:
        raise HTTPException(status_code=404, detail=f"No abstract found for DOI: {doi}")

    # Cache the result
    doi_cache.set(doi, result)

    return JSONResponse(content=result)
