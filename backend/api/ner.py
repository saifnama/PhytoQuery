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
    Standalone NER endpoint: takes raw text, returns extracted entities.
    Can be used on any text, not just papers.

    Request body:
    {
        "text": "raw text to extract entities from"
    }

    Response:
    {
        "summary": { ... },
        "entities": [ ... ]
    }
    """
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
    cached = ner_cache.get(doi)
    if cached:
        return NERResponse(**cached)

    text, mode = await EuropePMCService.fetch_paper_data(doi)
    if not text:
        raise HTTPException(status_code=404, detail="Paper not found")

    summary, entities_data = await service.process_text(text)
    entities = [Entity(**e) for e in entities_data]

    response = NERResponse(doi=doi, mode=mode, text=text[:1000], entities=entities)
    ner_cache.set(doi, {"entities": response.dict(), "summary": summary})
    return response


# --- Molecular Structure Image Endpoint ---


@router.get("/molecule/image")
async def get_molecule_image(
    smiles: str,
    width: int = 300,
    height: int = 200,
):
    """
    Generate a molecular structure image from a SMILES string.
    
    Args:
        smiles: SMILES string representing the molecule
        width: Image width in pixels (default 300)
        height: Image height in pixels (default 200)
    
    Returns:
        JSON with base64-encoded PNG image
    """
    from backend.core.rdkit_utils import smiles_to_image_base64
    
    image_b64 = smiles_to_image_base64(
        smiles=smiles,
        width=width,
        height=height
    )
    
    if not image_b64:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    return {
        "smiles": smiles,
        "image": image_b64,
        "width": width,
        "height": height
    }


@router.get("/molecule/info")
async def get_molecule_info(smiles: str):
    """
    Get molecular information from a SMILES string.
    
    Args:
        smiles: SMILES string
    
    Returns:
        JSON with molecular weight, formula, etc.
    """
    from backend.core.rdkit_utils import get_mol_info
    
    info = get_mol_info(smiles)
    
    if not info:
        raise HTTPException(status_code=400, detail="Invalid SMILES string")
    
    return {
        "smiles": smiles,
        **info
    }
