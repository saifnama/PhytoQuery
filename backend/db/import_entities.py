"""
Importer for the new 3-table star schema (papers, entities, paper_entities).

Reads the curated Excel sheet at ``EXCEL_PATH``, groups rows by DOI, upserts
a ``Paper`` row per DOI (falling back to OpenAlex for missing metadata), and
then for each (label, canonical_text) on the row:

    1. Enriches CHEMICAL / SPECIES via the gazetteer matchers.
    2. Overlays Excel-derived metadata on top of the gazetteer dict
       (Excel wins — it is manually curated).
    3. Get-or-creates the canonical row in ``entities`` (UNIQUE label+canonical).
    4. UPSERTs the junction row in ``paper_entities``. Because the Excel rows
       carry no mention_count of their own, re-importing the same row simply
       increments mention_count by 1.

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
from backend.db.models import Paper, Entity, PaperEntity
from backend.gazetteer.chemical_matcher import get_matcher as get_chemical_matcher
from backend.gazetteer.species_matcher import get_matcher as get_species_matcher


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXCEL_PATH = r"C:\Users\saif\saifnama_lab\csv data\testing\2_paper_data_for_NER.xlsx"


# ─── Columns we promote out of the merged metadata dict into typed Entity cols ──
PROMOTED_FIELDS = {
    "family",
    "common_name",
    "accepted_scientific_name",
    "taxon_id",
    "name_type",
    "inchikey",
    "smiles",
    "molecular_formula",
    "preferred_name",
    "country",
    "state",
    "source_db",
    "source_url",
}

# Keys we should never carry into metadata_json — they are either promoted
# above, stored in aliases_json, or are gazetteer-internal artefacts that
# would only add noise.
DROPPED_FIELDS = {
    "text",
    "span",
    "type",
    "label",
    "score",
    "linked_to",
    "canonical",
    "start",
    "end",
    "aliases",
}


# ───────────────────────────── OpenAlex fallback ─────────────────────────────

async def fetch_paper_metadata(doi: str):
    """Fetch Title, Journal, and Year from OpenAlex as a fallback."""
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
    async with engine.begin() as conn:
        # Create all tables
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
    """
    Map a single Excel row into a list of entity descriptors of the shape::

        {
            "label":        "SPECIES",
            "display_text": "Lavandula Stoechas",   # preferred casing
            "excel_meta":   {"family": "Lamiaceae", "common_name": "..."}
        }

    Casing rules:
        - SPECIES display_text uses Title Case (matches existing behavior).
        - All other labels use lowercased display_text.
        - canonical_text is derived later as display_text.lower().strip().
    """
    entities = []

    def add_entity(label, text_col, meta_cols=None):
        text = _cell(row, text_col)
        if not text:
            return
        excel_meta = {}
        if meta_cols:
            for meta_key, meta_col in meta_cols.items():
                val = _cell(row, meta_col)
                if val:
                    excel_meta[meta_key] = val

        # Standardize capitalization based on label
        if label == "SPECIES":
            display = text.title()
        else:
            display = text.lower()

        entities.append({
            "label": label,
            "display_text": display,
            "excel_meta": excel_meta,
        })

    # 1. Species (with Family and Common Name)
    add_entity("SPECIES", "Scientific_Name",
               {"family": "Family", "common_name": "Common name"})

    # 2. Chemical
    add_entity("CHEMICAL", "Chemical")

    # 3. Plant Part
    add_entity("PLANT PART", "Plant Part")

    # 4. Development Stage
    add_entity("DEVELOPMENT STAGE", "Development stage")

    # 5. Extraction Method
    add_entity("EXTRACTION METHOD", "Extraction Method")

    # 6. Analytical Technique
    add_entity("ANALYTICAL TECHNIQUE", "Analytical Technique")

    # 7. Bioactivity
    add_entity("BIOACTIVITY", "Bioactivity")

    # 8. Disease
    add_entity("DISEASE", "Disease")

    # 9. Season
    add_entity("SEASON", "Season")

    # 10. Location — display from "Location" col (fallback to "Country");
    #     country/state are typed columns on Entity.
    loc_text = _cell(row, "Location")
    country = _cell(row, "Country")
    if not loc_text and country:
        loc_text = country

    if loc_text:
        excel_meta = {}
        if country:
            excel_meta["country"] = country
        # No "State" column in the current Excel sheet; state stays NULL.
        entities.append({
            "label": "LOCATION",
            "display_text": loc_text.title(),
            "excel_meta": excel_meta,
        })

    return entities


# ──────────────────────── Entity get-or-create + enrich ─────────────────────

def _enrich_from_gazetteer(label: str, display_text: str,
                           chem_matcher, species_matcher) -> Dict[str, Any]:
    """
    Look the entity up in the appropriate gazetteer (CHEMICAL or SPECIES) and
    return its dict. Returns {} for unknown labels or unknown terms.

    Tries display_text first, then the lowercased canonical, so that gazetteer
    failures degrade gracefully without aborting the row.
    """
    if label not in ("CHEMICAL", "SPECIES"):
        return {}

    matcher = chem_matcher if label == "CHEMICAL" else species_matcher
    canonical_lower = display_text.lower().strip()

    try:
        result = matcher.lookup(display_text)
        if result is None and canonical_lower != display_text:
            result = matcher.lookup(canonical_lower)
        return result or {}
    except Exception as e:
        logger.warning(
            f"Gazetteer lookup failed for {label} '{display_text}': {e}"
        )
        return {}


async def get_or_create_entity(session, label: str, display_text: str,
                               excel_meta: Dict[str, Any],
                               chem_matcher, species_matcher) -> Entity:
    """
    Find-or-insert a canonical Entity row. Enriches CHEMICAL / SPECIES from the
    gazetteer; overlays Excel metadata on top (Excel wins on conflicts because
    it is manually curated). Returns an Entity with a valid ``id``.
    """
    canonical_text = display_text.lower().strip()

    # 1. Look for an existing canonical row first.
    existing = await session.execute(
        select(Entity).where(
            Entity.label == label,
            Entity.canonical_text == canonical_text,
        )
    )
    entity = existing.scalars().first()
    if entity is not None:
        return entity

    # 2. Build the enriched payload: gazetteer first, then Excel overlay.
    gazetteer = _enrich_from_gazetteer(label, display_text,
                                       chem_matcher, species_matcher)
    merged: Dict[str, Any] = {}
    if gazetteer:
        merged.update(gazetteer)
    # Excel overlays — manually curated, so it wins on conflicts.
    for k, v in (excel_meta or {}).items():
        if v is not None and v != "":
            merged[k] = v

    # 3. Promote typed columns out of merged.
    promoted = {field: merged.get(field) for field in PROMOTED_FIELDS}

    # 4. aliases_json comes off the "aliases" key, if present.
    aliases = merged.get("aliases")
    aliases_json = json.dumps(aliases) if aliases else None

    # 5. metadata_json keeps anything left over (not promoted, not aliases,
    #    not gazetteer-internal noise).
    leftovers = {
        k: v for k, v in merged.items()
        if k not in PROMOTED_FIELDS
        and k not in DROPPED_FIELDS
        and v is not None
        and v != ""
    }
    metadata_json = leftovers if leftovers else None

    entity = Entity(
        label=label,
        canonical_text=canonical_text,
        display_text=display_text,
        family=promoted.get("family"),
        common_name=promoted.get("common_name"),
        accepted_scientific_name=promoted.get("accepted_scientific_name"),
        taxon_id=promoted.get("taxon_id"),
        name_type=promoted.get("name_type"),
        inchikey=promoted.get("inchikey"),
        smiles=promoted.get("smiles"),
        molecular_formula=promoted.get("molecular_formula"),
        preferred_name=promoted.get("preferred_name"),
        country=promoted.get("country"),
        state=promoted.get("state"),
        source_db=promoted.get("source_db"),
        source_url=promoted.get("source_url"),
        aliases_json=aliases_json,
        metadata_json=metadata_json,
    )
    session.add(entity)
    # Flush so we get entity.id without committing yet.
    await session.flush()
    return entity


# ──────────────────────── paper_entities junction UPSERT ────────────────────

async def upsert_paper_entity(session, paper_id: int, entity_id: int) -> None:
    """
    Insert (paper_id, entity_id, 1) or, if the row already exists, increment
    its mention_count by 1.

    Re-running the importer on the same Excel sheet therefore bumps the
    mention_count by +1 each time. The Excel rows do not carry a mention_count
    of their own; one row = one mention.
    """
    stmt = sqlite_insert(PaperEntity).values(
        paper_id=paper_id,
        entity_id=entity_id,
        mention_count=1,
    )
    stmt = stmt.on_conflict_do_update(
        index_elements=[PaperEntity.paper_id, PaperEntity.entity_id],
        set_={"mention_count": PaperEntity.mention_count + 1},
    )
    await session.execute(stmt)


# ─────────────────────────────── Main flow ───────────────────────────────────

async def import_data():
    await init_db()

    # Initialize gazetteer matchers ONCE — they each load a .pkl cache on the
    # first call, which is slow. Per-row initialization would be disastrous.
    logger.info("Loading gazetteer matchers (chemical + species)...")
    chem_matcher = get_chemical_matcher()
    species_matcher = get_species_matcher()
    logger.info("Gazetteer matchers ready.")

    logger.info(f"Reading Excel file: {EXCEL_PATH}")
    df = pd.read_excel(EXCEL_PATH)

    # Clean DOIs (drop empty rows)
    df = df.dropna(subset=['DOI number'])

    async with AsyncSessionLocal() as session:
        # Group by DOI so we process one paper at a time
        grouped = df.groupby('DOI number')

        for doi, group in grouped:
            doi_str = str(doi).strip()

            # 1. Create or get Paper
            result = await session.execute(select(Paper).where(Paper.doi == doi_str))
            paper = result.scalars().first()

            if not paper:
                # Use the first row for paper metadata
                first_row = group.iloc[0]

                # Try to get metadata from Excel
                excel_title = str(first_row.get('Title')) if pd.notna(first_row.get('Title')) else None
                excel_journal = str(first_row.get('Journal')) if pd.notna(first_row.get('Journal')) else None
                excel_year = int(first_row.get('Year of data collection')) if pd.notna(first_row.get('Year of data collection')) else None
                excel_oa = False
                if 'Open Access' in first_row.index and pd.notna(first_row.get('Open Access')):
                    val = str(first_row.get('Open Access')).lower()
                    excel_oa = val in ['yes', 'true', '1', 'open']

                # FALLBACK: Fetch from API if data is missing
                api_meta = await fetch_paper_metadata(doi_str)

                # SPECIFIC OVERRIDE for the Togo paper requested by the user
                if doi_str == "10.1080/0972060X.2011.10643597":
                    title = "Chemical Composition and Cytotoxic Activity of Essential Oil of Chromolaena odorata L. Growing in Togo"
                    journal = "Journal of Essential Oil Bearing Plants"
                    year = 2011
                    is_oa = True  # As per user request "and open access is true then?"
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
                await session.flush()  # To get the paper.id

            # 2. Process every entity on every row for this DOI.
            row_mentions = 0
            for _, row in group.iterrows():
                mappings = get_entity_mappings(row)

                for em in mappings:
                    entity = await get_or_create_entity(
                        session,
                        label=em["label"],
                        display_text=em["display_text"],
                        excel_meta=em["excel_meta"],
                        chem_matcher=chem_matcher,
                        species_matcher=species_matcher,
                    )
                    await upsert_paper_entity(session, paper.id, entity.id)
                    row_mentions += 1

            # 3. Refresh entity_count for this paper from the junction table.
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
