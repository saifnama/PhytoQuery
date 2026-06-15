"""NER Router — Standalone entity extraction."""

from fastapi import APIRouter, Depends, HTTPException
from typing import List, Dict, Any
from backend.services.ner_engine import ner_service, NERService
from backend.services.europe_pmc import EuropePMCService
from backend.schemas.schemas import NERRequest, NERResponse, Entity
from backend.core.caching import ner_cache
import logging


# Dependency providers
def get_ner_service() -> NERService:
    return ner_service


router = APIRouter(prefix="/ner", tags=["NER"])
logger = logging.getLogger(__name__)


# --- Standalone NER Endpoint ---


@router.post("/process")
async def process_text_ner(
    request: dict,
    service: NERService = Depends(get_ner_service),
):
    """
    Standalone NER endpoint: takes raw text or sections, returns extracted entities.
    Can be used on any text, not just papers.

    Request body option 1 (plain text):
    {
        "text": "raw text to extract entities from"
    }

    Request body option 2 (sections - better for papers):
    {
        "sections": [
            {"title": "Abstract", "content": "..."},
            {"title": "Methods", "content": "..."}
        ]
    }

    Response:
    {
        "summary": { ... },
        "entities": [ ... ]
    }
    """
    sections = request.get("sections", [])
    if sections:
        summary, entities = await service.process_sections(sections)
    else:
        text = request.get("text", "")
        if not text:
            return {"error": "No text provided", "entities": [], "summary": {}}
        summary, entities = await service.process_text(text)

    return {
        "summary": summary,
        "entities": entities,
    }


# --- Legacy JSON Endpoint ---


@router.post("/doi/json", response_model=NERResponse)
async def process_doi_json(
    request: NERRequest, service: NERService = Depends(get_ner_service)
):
    doi = request.doi
    _, clean_id = EuropePMCService.parse_identifier(doi)
    cache_key = f"doi_json::{clean_id}"

    cached = ner_cache.get(cache_key)
    if cached:
        return NERResponse(
            **{k: v for k, v in cached.items() if not k.startswith("_")}
        )

    text, mode = await EuropePMCService.fetch_paper_data(clean_id)
    if not text:
        raise HTTPException(status_code=404, detail="Paper not found")

    # Strip XML tags for full_text mode so NER sees plain text (matches PDF pipeline)
    if mode == "full_text":
        text = EuropePMCService.clean_xml(text)

    summary, entities_data = await service.process_text(text)
    entities = [Entity(**e) for e in entities_data]

    response = NERResponse(doi=clean_id, mode=mode, text=text[:1000], entities=entities)
    ner_cache.set(cache_key, response.dict())
    return response


# --- Clear NER Cache ---

@router.delete("/cache/{doi}")
async def clear_ner_cache(doi: str):
    """Clear NER cache for a specific DOI (both disk cache and in-memory)."""
    try:
        _, clean_id = EuropePMCService.parse_identifier(doi)
    except Exception:
        clean_id = doi

    ner_cache.delete(doi)
    ner_cache.delete(clean_id)
    ner_cache.delete(f"doi_json::{clean_id}")
    ner_cache.delete(f"doi_json::{doi}")
    ner_service.result_cache.pop(doi, None)
    ner_service.result_cache.pop(clean_id, None)
    return {"status": "cleared", "doi": doi}
