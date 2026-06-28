"""
NER PDF Upload API - Extract entities from uploaded PDFs
"""
import base64
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import fitz  # PyMuPDF
from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from backend.core.session import attach_session_cookie, get_or_set_session_id, get_session_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ner", tags=["ner"])

# Upload directory for NER-processed PDFs
NER_UPLOAD_DIR = os.path.join(os.getcwd(), "data", "ner_uploads")
PDF_SIGNING_SECRET_FILE = os.path.join(NER_UPLOAD_DIR, ".pdf-signing-secret")
PDF_URL_TTL_SECONDS = 60 * 60 * 24 * 30


def _build_stored_pdf_name(original_filename: str) -> str:
    stem = Path(original_filename).stem or "paper"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "paper"
    return f"{safe_stem}_{uuid4().hex}.pdf"


@lru_cache(maxsize=1)
def _get_signing_secret() -> bytes:
    env_secret = os.getenv("PHYTOQUERY_PDF_SIGNING_SECRET")
    if env_secret:
        return env_secret.encode("utf-8")

    os.makedirs(NER_UPLOAD_DIR, exist_ok=True)
    if os.path.isfile(PDF_SIGNING_SECRET_FILE):
        with open(PDF_SIGNING_SECRET_FILE, "rb") as secret_file:
            return secret_file.read().strip()

    secret = secrets.token_hex(32).encode("utf-8")
    with open(PDF_SIGNING_SECRET_FILE, "wb") as secret_file:
        secret_file.write(secret)
    return secret


def _build_pdf_token(stored_filename: str, owner_id: str, expires_at: int) -> str:
    payload = f"{stored_filename}:{owner_id}:{expires_at}".encode("utf-8")
    digest = hmac.new(_get_signing_secret(), payload, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode("utf-8").rstrip("=")


def _build_signed_pdf_url(stored_filename: str, owner_id: str) -> str:
    expires_at = int(time.time()) + PDF_URL_TTL_SECONDS
    token = _build_pdf_token(stored_filename, owner_id, expires_at)
    return f"/ner/uploaded/{stored_filename}?expires={expires_at}&token={token}"


def _verify_pdf_token(stored_filename: str, owner_id: str, expires_at: int, token: str) -> bool:
    if expires_at < int(time.time()):
        return False
    expected = _build_pdf_token(stored_filename, owner_id, expires_at)
    return hmac.compare_digest(expected, token)


def _metadata_path(stored_filename: str) -> str:
    return os.path.join(NER_UPLOAD_DIR, f"{stored_filename}.meta.json")


def _write_upload_metadata(stored_filename: str, user_id: str) -> None:
    with open(_metadata_path(stored_filename), "w", encoding="utf-8") as meta_file:
        json.dump({"user_id": user_id}, meta_file)


def _read_upload_metadata(stored_filename: str) -> Dict[str, Any]:
    path = _metadata_path(stored_filename)
    if not os.path.isfile(path):
        return {}
    with open(path, "r", encoding="utf-8") as meta_file:
        return json.load(meta_file)


def _cleanup_upload_artifacts(file_path: str, stored_filename: str) -> None:
    for path in (file_path, _metadata_path(stored_filename)):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                logger.warning("Failed to remove upload artifact: %s", path)


async def extract_metadata_from_pdf(doc: fitz.Document) -> Dict[str, Any]:
    """Extract metadata from PDF using PyMuPDF."""
    meta = doc.metadata
    title = meta.get("title", "")
    author = meta.get("author", "")
    subject = meta.get("subject", "")
    creator = meta.get("creator", "")
    producer = meta.get("producer", "")

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
    from backend.gazetteer import analytical_technique_matcher, bioactivity_matcher
    from backend.gazetteer import chemical_matcher, development_stage_matcher, extraction_method_matcher
    from backend.gazetteer import plant_part_matcher, season_matcher, species_matcher

    # Collect all entities with their output labels
    matchers = [
        (chemical_matcher.match_chemicals, "CHEMICAL"),
        (plant_part_matcher.match_plant_parts, "PLANT_PART"),
        (species_matcher.match_species, "SPECIES"),
        (analytical_technique_matcher.match_analytical_techniques, "ANALYTICAL_TECHNIQUE"),
        (extraction_method_matcher.match_extraction_methods, "EXTRACTION_METHOD"),
        (development_stage_matcher.match_development_stages, "DEVELOPMENT_STAGE"),
        (season_matcher.match_seasons, "SEASON"),
        (bioactivity_matcher.match_bioactivities, "BIOACTIVITY"),
    ]
    all_labeled = []
    for fn, label in matchers:
        for e in fn(text):
            all_labeled.append((e, label))

    # Cross-type dedup: longest span wins regardless of type
    kept = []
    for e, label in sorted(all_labeled, key=lambda x: (x[0].get("start") or 0, -(x[0].get("end") or 0))):
        s, en = e.get("start"), e.get("end")
        if s is None or en is None:
            kept.append((e, label))
            continue
        if not any(s < k[0]["end"] and en > k[0]["start"] for k in kept if "start" in k[0] and "end" in k[0]):
            kept.append((e, label))

    grouped: Dict[str, list] = {}
    for e, label in kept:
        grouped.setdefault(label, []).append(e)

    entities: Dict[str, List[str]] = {}
    seen_lower: Dict[str, set] = {}
    count_map: Dict[str, Dict[str, int]] = {}
    canonical_data: Dict[str, Dict[str, Dict[str, Any]]] = {}

    def add_canonical_ents(ents: list, label: str):
        if label not in entities:
            entities[label] = []
            seen_lower[label] = set()
            count_map[label] = {}
            canonical_data[label] = {}

        alias_counts: Dict[str, Dict[str, int]] = {}
        canonical_aliases_map: Dict[str, List[str]] = {}

        for entity in ents:
            txt = entity.get("span", entity.get("text", ""))
            canonical = entity.get("canonical", txt)
            if txt and canonical:
                alias_counts.setdefault(canonical, {})
                alias_counts[canonical][txt] = alias_counts[canonical].get(txt, 0) + 1
                if canonical not in canonical_aliases_map and entity.get("aliases"):
                    canonical_aliases_map[canonical] = entity.get("aliases", [])

        for canonical, matched_texts in alias_counts.items():
            if not matched_texts:
                continue

            total_count = sum(matched_texts.values())
            canonical_lower = canonical.lower()

            if canonical_lower not in seen_lower[label]:
                seen_lower[label].add(canonical_lower)
                entities[label].append(canonical)
                count_map[label][canonical_lower] = total_count
                all_variants = list(matched_texts.keys())
                meta_aliases = canonical_aliases_map.get(canonical, [])
                canonical_data[label][canonical_lower] = {
                    "canonical": canonical,
                    "display_text": canonical,
                    "aliases": list(dict.fromkeys(all_variants + meta_aliases)),
                }
            else:
                count_map[label][canonical_lower] = count_map[label].get(canonical_lower, 0) + total_count
                existing = canonical_data[label].get(canonical_lower, {})
                existing_aliases = set(existing.get("aliases", []))
                existing_aliases.update(matched_texts.keys())
                existing_aliases.update(canonical_aliases_map.get(canonical, []))
                existing["aliases"] = list(existing_aliases)

    for label, ents in grouped.items():
        add_canonical_ents(ents, label)

    return entities, count_map, canonical_data


@router.post("/upload/json")
async def upload_pdf_for_ner(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Upload PDF for NER extraction.

    1. Extracts metadata (title, DOI)
    2. Extracts full text
    3. Runs NER on text
    4. Returns metadata + entities + stored PDF URL
    """
    original_filename = file.filename or "paper.pdf"
    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted")

    os.makedirs(NER_UPLOAD_DIR, exist_ok=True)

    user_id = get_or_set_session_id(request, response)
    stored_filename = _build_stored_pdf_name(original_filename)
    file_path = os.path.join(NER_UPLOAD_DIR, stored_filename)

    try:
        content = await file.read()
        with open(file_path, "wb") as output_file:
            output_file.write(content)
        _write_upload_metadata(stored_filename, user_id)
    except Exception as exc:
        logger.error("Failed to save PDF: %s", exc)
        _cleanup_upload_artifacts(file_path, stored_filename)
        raise HTTPException(status_code=500, detail="Failed to save file") from exc

    doc: Optional[fitz.Document] = None
    try:
        doc = fitz.open(file_path)
        metadata = await extract_metadata_from_pdf(doc)
        text = await extract_text_from_pdf(doc)
        entities_by_type, entity_counts, canonical_data = extract_entities_fast(text)

        total_entities = sum(sum(counts.values()) for counts in entity_counts.values())
        entities_with_counts: Dict[str, List[Dict[str, Any]]] = {}
        for label, texts in entities_by_type.items():
            entities_with_counts[label] = []
            for txt in texts:
                txt_lower = txt.lower()
                count = entity_counts.get(label, {}).get(txt_lower, 1)
                entry: Dict[str, Any] = {"text": txt, "count": count}
                meta = canonical_data.get(label, {}).get(txt_lower)
                if meta:
                    entry["canonical"] = meta["canonical"]
                    entry["aliases"] = meta["aliases"]
                entities_with_counts[label].append(entry)

        return {
            "filename": original_filename,
            "stored_filename": stored_filename,
            "pdf_url": _build_signed_pdf_url(stored_filename, user_id),
            "metadata": {
                "title": metadata.get("title", "Untitled"),
                "doi": metadata.get("doi", ""),
            },
            "entity_count": total_entities,
            "entities": entities_by_type,
            "entity_counts": entities_with_counts,
        }
    except Exception as exc:
        logger.error("PDF NER failed: %s", exc)
        _cleanup_upload_artifacts(file_path, stored_filename)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(exc)}") from exc
    finally:
        if doc is not None:
            doc.close()


@router.get("/uploaded/{stored_filename}")
async def view_uploaded_pdf(
    request: Request,
    response: Response,
    stored_filename: str,
    expires: int = Query(...),
    token: str = Query(...),
) -> FileResponse:
    safe_name = os.path.basename(stored_filename)
    if safe_name != stored_filename or not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid PDF path")

    file_path = os.path.join(NER_UPLOAD_DIR, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    metadata = _read_upload_metadata(safe_name)
    owner_id = metadata.get("user_id", "default")
    session_id = get_session_id(request)
    if session_id != owner_id or not _verify_pdf_token(safe_name, owner_id, expires, token):
        raise HTTPException(status_code=403, detail="PDF access denied")

    file_response = FileResponse(file_path, media_type="application/pdf")
    attach_session_cookie(file_response, request, session_id)
    return file_response


@router.delete("/uploaded/{stored_filename}")
async def delete_uploaded_pdf(
    request: Request,
    response: Response,
    stored_filename: str,
) -> Dict[str, str]:
    """Delete an uploaded PDF and its metadata."""
    safe_name = os.path.basename(stored_filename)
    if safe_name != stored_filename or not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Invalid PDF path")

    file_path = os.path.join(NER_UPLOAD_DIR, safe_name)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="PDF not found")

    metadata = _read_upload_metadata(safe_name)
    owner_id = metadata.get("user_id", "default")
    session_id = get_session_id(request)
    if session_id != owner_id:
        raise HTTPException(status_code=403, detail="PDF access denied")

    _cleanup_upload_artifacts(file_path, safe_name)
    return {"status": "success", "message": "PDF deleted"}
