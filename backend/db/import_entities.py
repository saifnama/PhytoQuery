"""
Importer for the two-table schema (``papers`` + ``paper_entities``).

Reads the curated Excel sheet at ``EXCEL_PATH``, groups rows by DOI,
upserts a ``Paper`` row per DOI (falling back to OpenAlex for missing
metadata), and then for each ``(label, canonical_text)`` on each row
UPSERTs into ``paper_entities``:

  * On first insert: ``frequency = 1`` and ``metadata`` is the merged
    gazetteer + Excel metadata dict, or NULL if nothing useful.
  * On conflict (re-import of an already-seen ``(paper_id, label,
    canonical_text)`` tuple): ``frequency += 1`` and ``metadata`` is
    kept as-is (``COALESCE(existing, new)`` — first non-NULL wins).

Labels imported (matches the user's scope decision):
    CHEMICAL, SPECIES, PLANT PART, ANALYTICAL TECHNIQUE,
    EXTRACTION METHOD, DEVELOPMENT STAGE, LOCATION
BIOACTIVITY, DISEASE, SEASON are intentionally NOT imported.

Run via:  ``python -m backend.db.import_entities``
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any, Dict, Optional

import pandas as pd
import httpx
from sqlalchemy import select, func
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

# Add project root to path so we can import 'backend'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.db.database import Base, engine, AsyncSessionLocal
from backend.db.models import Paper, PaperEntity
from backend.gazetteer.chemical_matcher import get_matcher as get_chemical_matcher
from backend.gazetteer.species_matcher import get_matcher as get_species_matcher


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXCEL_PATH = r"C:\Users\saif\saifnama_lab\csv data\testing\2_paper_data_for_NER.xlsx"


# Keys we should never carry into the metadata JSON column — gazetteer
# internals + Excel scratch fields that would only add noise.
DROPPED_FIELDS = {
    "text", "span", "type", "label", "score",
    "linked_to", "canonical", "start", "end", "aliases",
}


# ───────────────────────────── OpenAlex fallback ─────────────────────────────

async def fetch_paper_metadata(doi: str):
    """Fetch Title, Journal, Year, OA-flag from OpenAlex when the Excel
    sheet doesn't carry them."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            url = f"https://api.openalex.org/works/https://doi.org/{doi}"
            response = await client.get(url)
            if response.status_code == 200:
                data = response.json()
                return {
                    "title": data.get("title"),
                    "journal": data.get("host_venue", {}).get("display_name")
                        or data.get("primary_location", {}).get("source", {}).get("display_name"),
                    "year": data.get("publication_year"),
                    "is_oa": data.get("open_access", {}).get("is_oa", False),
                }
    except Exception as e:
        logger.warning(f"Metadata fetch failed for {doi}: {e}")
    return None


# ─────────────────────────────── DB lifecycle ───────────────────────────────

async def init_db():
    """Create tables if missing. No-op against the existing live DB —
    SQLAlchemy's ``create_all`` uses ``CREATE TABLE IF NOT EXISTS``."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized.")


# ──────────────────────── Excel row → entity descriptors ────────────────────

def _cell(row, col) -> Optional[str]:
    """Return a stripped string for ``row[col]`` or None for blanks/NaN."""
    if col not in row.index:
        return None
    val = row[col]
    if pd.isna(val):
        return None
    s = str(val).strip()
    if not s or s.lower() == "nan":
        return None
    return s


def get_entity_mappings(row):
    """Map a single Excel row into a list of entity descriptors of shape::

        {"label": "SPECIES",
         "canonical_text": "Lavandula stoechas",
         "excel_meta": {"family": "Lamiaceae", "common_name": "..."}}

    Casing rule: SPECIES keeps the as-written form (matches the DB
    convention of "Lantana camara" etc.); other labels use lowercased
    canonical_text (matches "alpha-pinene", "gc-ms", ...).
    """
    entities = []

    def add_entity(label: str, text_col: str, meta_cols: Optional[Dict[str, str]] = None):
        text = _cell(row, text_col)
        if not text:
            return
        excel_meta: Dict[str, Any] = {}
        if meta_cols:
            for meta_key, meta_col in meta_cols.items():
                val = _cell(row, meta_col)
                if val:
                    excel_meta[meta_key] = val
        if label == "SPECIES":
            canonical = text  # preserve as-written (matches existing DB rows)
        else:
            canonical = text.lower()
        entities.append({
            "label": label,
            "canonical_text": canonical,
            "excel_meta": excel_meta,
        })

    # Per the user's scope decision: BIOACTIVITY, DISEASE, SEASON are NOT
    # imported. The Excel sheet may still have those columns; we ignore them.
    add_entity("SPECIES", "Scientific_Name",
               {"family": "Family", "common_name": "Common name"})
    add_entity("CHEMICAL", "Chemical")
    add_entity("PLANT PART", "Plant Part")
    add_entity("DEVELOPMENT STAGE", "Development stage")
    add_entity("EXTRACTION METHOD", "Extraction Method")
    add_entity("ANALYTICAL TECHNIQUE", "Analytical Technique")

    # LOCATION — display from "Location" col, fall back to "Country".
    loc_text = _cell(row, "Location")
    country = _cell(row, "Country")
    if not loc_text and country:
        loc_text = country
    if loc_text:
        loc_meta: Dict[str, Any] = {}
        if country:
            loc_meta["country"] = country
        entities.append({
            "label": "LOCATION",
            "canonical_text": loc_text.title(),
            "excel_meta": loc_meta,
        })

    return entities


# ─────────────────────── gazetteer enrichment + metadata merge ────────────

def _enrich_from_gazetteer(label: str, canonical_text: str,
                           chem_matcher, species_matcher) -> Dict[str, Any]:
    """Look the entity up in the appropriate gazetteer (CHEMICAL or SPECIES).
    Returns the gazetteer's metadata dict, or ``{}`` for unknown labels /
    unknown terms. Tries the canonical_text first, then a normalized
    lowercase variant.
    """
    if label not in ("CHEMICAL", "SPECIES"):
        return {}

    matcher = chem_matcher if label == "CHEMICAL" else species_matcher
    canonical_lower = canonical_text.lower().strip()

    try:
        result = matcher.lookup(canonical_text)
        if result is None and canonical_lower != canonical_text:
            result = matcher.lookup(canonical_lower)
        return result or {}
    except Exception as e:
        logger.warning(
            f"Gazetteer lookup failed for {label} '{canonical_text}': {e}"
        )
        return {}


def _build_metadata(gazetteer_meta: Dict[str, Any],
                    excel_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Merge gazetteer + Excel metadata into a single dict for the
    ``paper_entities.metadata`` JSON column. Excel wins on conflicts
    (manually curated). Returns ``None`` if there is nothing useful to
    store — that keeps the column SQL NULL.
    """
    merged: Dict[str, Any] = {}
    if gazetteer_meta:
        merged.update(gazetteer_meta)
    if excel_meta:
        for k, v in excel_meta.items():
            if v is not None and v != "":
                merged[k] = v

    cleaned = {
        k: v for k, v in merged.items()
        if k not in DROPPED_FIELDS
        and v is not None
        and v != ""
    }
    return cleaned if cleaned else None


# ──────────────────────── paper_entities UPSERT ────────────────────────────

async def upsert_paper_entity(
    session,
    paper_id: int,
    label: str,
    canonical_text: str,
    metadata: Optional[Dict[str, Any]],
) -> None:
    """Insert one ``paper_entities`` row, or on conflict bump frequency
    and preserve existing metadata (first non-NULL payload wins).

    The UNIQUE INDEX on (paper_id, label, canonical_text) is what powers
    the ON CONFLICT clause.
    """
    metadata_json = json.dumps(metadata) if metadata else None

    stmt = sqlite_insert(PaperEntity).values(
        paper_id=paper_id,
        label=label,
        canonical_text=canonical_text,
        frequency=1,
        meta=metadata_json,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[
            PaperEntity.paper_id,
            PaperEntity.label,
            PaperEntity.canonical_text,
        ],
        set_={
            "frequency": PaperEntity.frequency + 1,
            # COALESCE keeps the existing metadata if non-NULL; only
            # backfills with the new payload when the row had no
            # metadata yet.
            "meta": func.coalesce(PaperEntity.meta, stmt.excluded.meta),
        },
    )
    await session.execute(stmt)


# ─────────────────────────────── Main flow ───────────────────────────────────

async def import_data():
    await init_db()

    # Gazetteers each load a .pkl cache on first call — keep ONE instance
    # for the entire run.
    logger.info("Loading gazetteer matchers (chemical + species)...")
    chem_matcher = get_chemical_matcher()
    species_matcher = get_species_matcher()
    logger.info("Gazetteer matchers ready.")

    logger.info(f"Reading Excel file: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)
    df = df.dropna(subset=["DOI number"])

    async with AsyncSessionLocal() as session:
        grouped = df.groupby("DOI number")

        for doi, group in grouped:
            doi_str = str(doi).strip()

            # 1. Get or create Paper
            result = await session.execute(select(Paper).where(Paper.doi == doi_str))
            paper = result.scalars().first()

            if not paper:
                first_row = group.iloc[0]

                excel_title = str(first_row.get("Title")) if pd.notna(first_row.get("Title")) else None
                excel_journal = str(first_row.get("Journal")) if pd.notna(first_row.get("Journal")) else None
                excel_year = (
                    int(first_row.get("Year of data collection"))
                    if pd.notna(first_row.get("Year of data collection"))
                    else None
                )
                excel_oa = False
                if "Open Access" in first_row.index and pd.notna(first_row.get("Open Access")):
                    val = str(first_row.get("Open Access")).lower()
                    excel_oa = val in ("yes", "true", "1", "open")

                api_meta = await fetch_paper_metadata(doi_str)

                # Specific override for the Togo paper, requested by the
                # user when seeding the DB.
                if doi_str == "10.1080/0972060X.2011.10643597":
                    title = (
                        "Chemical Composition and Cytotoxic Activity of "
                        "Essential Oil of Chromolaena odorata L. Growing in Togo"
                    )
                    journal = "Journal of Essential Oil Bearing Plants"
                    year = 2011
                    is_oa = True
                else:
                    title = excel_title or (api_meta.get("title") if api_meta else None)
                    journal = excel_journal or (api_meta.get("journal") if api_meta else None)
                    year = excel_year or (api_meta.get("year") if api_meta else None)
                    is_oa = excel_oa or (api_meta.get("is_oa") if api_meta else False)

                paper = Paper(
                    doi=doi_str,
                    title=title,
                    journal=journal,
                    year=year,
                    is_open_access=is_oa,
                    entity_count=0,
                )
                session.add(paper)
                await session.flush()  # populate paper.id

            # 2. UPSERT each entity descriptor for every row of this DOI.
            row_mentions = 0
            for _, row in group.iterrows():
                mappings = get_entity_mappings(row)
                for em in mappings:
                    label = em["label"]
                    canonical = em["canonical_text"]
                    gazetteer = _enrich_from_gazetteer(
                        label, canonical, chem_matcher, species_matcher
                    )
                    metadata = _build_metadata(gazetteer, em["excel_meta"])
                    await upsert_paper_entity(
                        session, paper.id, label, canonical, metadata
                    )
                    row_mentions += 1

            # 3. Refresh entity_count for this paper from the live table.
            paper.entity_count = await session.scalar(
                select(func.count())
                .select_from(PaperEntity)
                .where(PaperEntity.paper_id == paper.id)
            )

            logger.info(
                f"Imported DOI: {doi_str} — processed {row_mentions} row-mentions, "
                f"paper now linked to {paper.entity_count} distinct entities."
            )

        await session.commit()
        logger.info("Import complete.")


if __name__ == "__main__":
    asyncio.run(import_data())
