"""
NER PDF Upload API - Extract entities from uploaded PDFs
"""
import os
import re
import logging
from typing import Any, Dict, List

import fitz  # PyMuPDF
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ner", tags=["ner"])

# Upload directory for NER-processed PDFs
NER_UPLOAD_DIR = os.path.join(os.getcwd(), "data", "ner_uploads")


async def extract_metadata_from_pdf(doc: fitz.Document) -> Dict[str, Any]:
    """Extract metadata from PDF using PyMuPDF."""
    meta = doc.metadata
    title = meta.get("title", "")
    author = meta.get("author", "")
    subject = meta.get("subject", "")
    creator = meta.get("creator", "")
    producer = meta.get("producer", "")
    
    # Try to extract DOI from text
    doi_pattern = r"10\.\d{4,}/[^\s]+"
    doi = ""
    
    search_text = f"{title} {author} {subject} {creator} {producer}"
    doi_match = re.search(doi_pattern, search_text)
    if doi_match:
        doi = doi_match.group()
    
    if not doi and doc.page_count > 0:
        page1_text = doc[0].get_text("text", sort=True)
        doi_match = re.search(doi_pattern, page1_text)
        if doi_match:
            doi = doi_match.group()[:100]
    
    return {
        "title": title or "Untitled",
        "doi": doi,
    }


async def extract_text_from_pdf(doc: fitz.Document) -> str:
    """Extract full text from PDF using PyMuPDF."""
    text_parts = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        text = page.get_text()
        if text.strip():
            text_parts.append(text)

    return "\n\n".join(text_parts)


def extract_entities_fast(text: str) -> tuple[Dict[str, List[str]], Dict[str, Dict[str, int]], Dict[str, Dict[str, Dict[str, Any]]]]:
    """Fast dictionary-only entity extraction (no LLM). Returns (entities, counts, canonical_data)."""
    from backend.gazetteer import plant_part_matcher, chemical_matcher, species_matcher
    from backend.gazetteer import analytical_technique_matcher, extraction_method_matcher
    from backend.gazetteer import development_stage_matcher, season_matcher, bioactivity_matcher

    # Dictionary-only matching (FAST)
    plant_parts = plant_part_matcher.match_plant_parts(text)
    chemicals = chemical_matcher.match_chemicals(text)
    species = species_matcher.match_species(text)
    analytical = analytical_technique_matcher.match_analytical_techniques(text)
    extraction = extraction_method_matcher.match_extraction_methods(text)
    development = development_stage_matcher.match_development_stages(text)
    seasons = season_matcher.match_seasons(text)
    bioactivities = bioactivity_matcher.match_bioactivities(text)

    # Group by type with case-insensitive deduplication, track counts
    entities: Dict[str, List[str]] = {}
    seen_lower: Dict[str, set] = {}
    count_map: Dict[str, Dict[str, int]] = {}  # label -> text_lower -> count
    canonical_data: Dict[str, Dict[str, Dict[str, Any]]] = {}  # label -> canonical_lower -> {canonical, aliases}

    def add_simple_ents(ents: list, label: str):
        if label not in entities:
            entities[label] = []
            seen_lower[label] = set()
            count_map[label] = {}
        for e in ents:
            txt = e.get("span", e.get("text", ""))
            if txt:
                txt_lower = txt.lower()
                if txt_lower not in seen_lower[label]:
                    seen_lower[label].add(txt_lower)
                    entities[label].append(txt)
                    count_map[label][txt_lower] = 1
                else:
                    # Increment count
                    count_map[label][txt_lower] = count_map[label].get(txt_lower, 0) + 1

    def add_canonical_ents(ents: list, label: str):
        """Group dictionary-backed entities by canonical, use CSV canonical name as display text."""
        if label not in entities:
            entities[label] = []
            seen_lower[label] = set()
            count_map[label] = {}
            canonical_data[label] = {}

        # canonical -> matched_text -> count
        alias_counts: Dict[str, Dict[str, int]] = {}
        canonical_aliases_map: Dict[str, List[str]] = {}  # canonical -> all aliases from metadata

        for e in ents:
            txt = e.get("span", e.get("text", ""))
            canonical = e.get("canonical", txt)
            if txt and canonical:
                alias_counts.setdefault(canonical, {})
                alias_counts[canonical][txt] = alias_counts[canonical].get(txt, 0) + 1
                # Store aliases from metadata for enrichment
                if canonical not in canonical_aliases_map and e.get("aliases"):
                    canonical_aliases_map[canonical] = e.get("aliases", [])

        for canonical, matched_texts in alias_counts.items():
            if not matched_texts:
                continue

            # Use CSV canonical name as display text (like DOI NER)
            display_text = canonical
            total_count = sum(matched_texts.values())
            canonical_lower = canonical.lower()

            if canonical_lower not in seen_lower[label]:
                seen_lower[label].add(canonical_lower)
                entities[label].append(display_text)
                # Store count by canonical for lookup
                count_map[label][canonical_lower] = total_count
                # Merge matched text variants with metadata aliases
                all_variants = list(matched_texts.keys())
                meta_aliases = canonical_aliases_map.get(canonical, [])
                merged = list(dict.fromkeys(all_variants + meta_aliases))
                canonical_data[label][canonical_lower] = {
                    "canonical": canonical,
                    "display_text": display_text,
                    "aliases": merged,
                }
            else:
                # Merge counts if same canonical appears again (shouldn't happen with proper dedup)
                count_map[label][canonical_lower] = count_map[label].get(canonical_lower, 0) + total_count
                # Merge aliases
                existing = canonical_data[label].get(canonical_lower, {})
                existing_aliases = set(existing.get("aliases", []))
                existing_aliases.update(matched_texts.keys())
                meta_aliases = canonical_aliases_map.get(canonical, [])
                existing_aliases.update(meta_aliases)
                existing["aliases"] = list(existing_aliases)

    add_canonical_ents(plant_parts, "PLANT_PART")
    add_canonical_ents(chemicals, "CHEMICAL")
    add_canonical_ents(species, "SPECIES")
    add_canonical_ents(analytical, "ANALYTICAL_TECHNIQUE")
    add_canonical_ents(extraction, "EXTRACTION_METHOD")
    add_canonical_ents(development, "DEVELOPMENT_STAGE")
    add_canonical_ents(seasons, "SEASON")
    add_canonical_ents(bioactivities, "BIOACTIVITY")

    return entities, count_map, canonical_data


@router.post("/upload/json")
async def upload_pdf_for_ner(
    file: UploadFile = File(...),
) -> JSONResponse:
    """
    Upload PDF for NER extraction.
    
    1. Extracts metadata (title, author, DOI)
    2. Extracts full text
    3. Runs NER on text
    4. Returns metadata + entities
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")
    
    # Create upload directory
    os.makedirs(NER_UPLOAD_DIR, exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join(NER_UPLOAD_DIR, file.filename)
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
    except Exception as e:
        logger.error(f"Failed to save PDF: {e}")
        raise HTTPException(status_code=500, detail="Failed to save file")
    
    try:
        # Open with PyMuPDF
        doc = fitz.open(file_path)
        
        # Extract metadata
        metadata = await extract_metadata_from_pdf(doc)
        
        # Extract full text
        text = await extract_text_from_pdf(doc)
        
        # Run FAST dictionary-only NER (no LLM, no slow processing)
        entities_by_type, entity_counts, canonical_data = extract_entities_fast(text)

        doc.close()

        # Calculate total entity count (sum of all occurrences)
        total_entities = sum(sum(cnt.values()) for cnt in entity_counts.values())

        # Build entities with counts
        entities_with_counts: Dict[str, List[Dict[str, Any]]] = {}
        for label, texts in entities_by_type.items():
            entities_with_counts[label] = []
            for txt in texts:
                txt_lower = txt.lower()
                cnt = entity_counts.get(label, {}).get(txt_lower, 1)
                entry: Dict[str, Any] = {"text": txt, "count": cnt}
                # Add canonical/aliases for dictionary-backed entities
                meta = canonical_data.get(label, {}).get(txt_lower)
                if meta:
                    entry["canonical"] = meta["canonical"]
                    entry["aliases"] = meta["aliases"]
                entities_with_counts[label].append(entry)

        return JSONResponse({
            "filename": file.filename,
            "metadata": {
                "title": metadata.get("title", "Untitled"),
                "doi": metadata.get("doi", ""),
            },
            "entity_count": total_entities,
            "entities": entities_by_type,
            "entity_counts": entities_with_counts,
        })
        
    except Exception as e:
        logger.error(f"PDF NER failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")
