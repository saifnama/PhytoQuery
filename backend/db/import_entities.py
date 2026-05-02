import os
import sys
import asyncio
import pandas as pd
import logging
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import select

# Add project root to path so we can import 'backend'
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import httpx
from backend.db.database import DATABASE_URL, Base, engine, AsyncSessionLocal
from backend.db.models import Paper, PaperEntity

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
                    "journal": data.get("host_venue", {}).get("display_name") or data.get("primary_location", {}).get("source", {}).get("display_name"),
                    "year": data.get("publication_year"),
                    "is_oa": data.get("open_access", {}).get("is_oa", False)
                }
    except Exception as e:
        logger.warning(f"Metadata fetch failed for {doi}: {e}")
    return None


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXCEL_PATH = r"C:\Users\saif\saifnama_lab\csv data\testing\2_paper_data_for_NER.xlsx"

async def init_db():
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database initialized.")

def get_entity_mappings(row):
    """Map Excel columns to standard entity labels and bundle metadata."""
    entities = []
    
    def add_entity(label, text_col, meta_cols=None):
        if text_col in row.index and pd.notna(row[text_col]):
            text = str(row[text_col]).strip()
            if text and text.lower() != "nan":
                metadata = {}
                if meta_cols:
                    for meta_key, meta_col in meta_cols.items():
                        if meta_col in row.index and pd.notna(row[meta_col]):
                            val = str(row[meta_col]).strip()
                            if val and val.lower() != "nan":
                                metadata[meta_key] = val
                
                # Standardize capitalization based on label
                if label == "SPECIES":
                    text = text.title()
                else:
                    text = text.lower()
                    
                entities.append({
                    "label": label,
                    "canonical_text": text,
                    "metadata": metadata if metadata else None
                })

    # 1. Species (with Family and Common Name)
    add_entity("SPECIES", "Scientific_Name", {"family": "Family", "common_name": "Common name"})
    
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
    
    # 7. Bioactivity (Catch-all if column exists)
    add_entity("BIOACTIVITY", "Bioactivity")
    
    # 8. Disease
    add_entity("DISEASE", "Disease")
    
    # 9. Season
    add_entity("SEASON", "Season")
    
    # 10. Location (Location with City, State, Country as metadata)
    loc_text = str(row.get("Location", "")).strip() if pd.notna(row.get("Location")) else ""
    if not loc_text:
        # Fallback to Country if Location is blank
        if "Country" in row.index and pd.notna(row["Country"]) and str(row["Country"]).strip() and str(row["Country"]).strip().lower() != "nan":
            loc_text = str(row["Country"]).strip()
            
    if loc_text:
        metadata = {}
        if "Country" in row.index and pd.notna(row["Country"]) and str(row["Country"]).strip() and str(row["Country"]).strip().lower() != "nan":
            metadata["country"] = str(row["Country"]).strip()
        entities.append({
            "label": "LOCATION",
            "canonical_text": loc_text.title(),
            "metadata": metadata if metadata else None
        })

    return entities

async def import_data():
    await init_db()
    
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
                    is_oa = True # As per user request "and open access is true then?"
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
                    entity_count=0
                )
                session.add(paper)
                await session.flush()  # To get the paper.id
            
            # 2. Add Entities
            added_entities = 0
            for _, row in group.iterrows():
                mappings = get_entity_mappings(row)
                
                for em in mappings:
                    # Check if exact entity already exists for this paper to avoid duplicates
                    # (This is a simple check, could be optimized for large inserts)
                    existing = await session.execute(
                        select(PaperEntity).where(
                            PaperEntity.paper_id == paper.id,
                            PaperEntity.label == em["label"],
                            PaperEntity.canonical_text == em["canonical_text"]
                        )
                    )
                    
                    if not existing.scalars().first():
                        entity = PaperEntity(
                            paper_id=paper.id,
                            label=em["label"],
                            canonical_text=em["canonical_text"],
                            metadata_json=em["metadata"]
                        )
                        session.add(entity)
                        added_entities += 1
            
            # Update total entity count
            paper.entity_count += added_entities
            
            logger.info(f"Imported DOI: {doi_str} with {added_entities} new entities.")
            
        await session.commit()
        logger.info("Import complete.")

if __name__ == "__main__":
    asyncio.run(import_data())
