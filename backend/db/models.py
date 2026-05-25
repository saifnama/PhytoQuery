"""
SQLAlchemy models for the PhytoQuery per-type schema (v2).

Layout: 15 tables — one master papers table, one catalog table per entity type,
and one junction table per entity type linking papers to entities.

    papers                                              ← master paper list
    chemicals             + paper_chemicals             ← rich chemistry metadata
    species               + paper_species               ← rich taxonomy metadata
    locations             + paper_locations             ← country/state metadata
    plant_parts           + paper_plant_parts           ← bare label-only entities
    analytical_techniques + paper_analytical_techniques ← bare label-only entities
    extraction_methods    + paper_extraction_methods    ← bare label-only entities
    development_stages    + paper_development_stages    ← bare label-only entities

Casing rule (uniform across every entity table):
    ``canonical_text`` is stored LOWERCASED — the UNIQUE constraint on each
    table collapses casing-drift (e.g. "Lavandula stoechas" vs "lavandula
    stoechas" → one row). ``display_text`` keeps the preferred casing for UI.

Synonyms / aliases:
    Canonicalisation happens at extraction time via the gazetteer CSV files
    (chemical.csv, species.csv, …). By the time data reaches the DB the
    canonical form is fixed, so two papers with different surface forms
    ("GC-MS" vs "gas chromatography-mass spectrometry") both link to the same
    canonical row. Each entity table has an OPTIONAL ``aliases_json`` column
    storing known synonyms for display-side "also known as" rendering.

Junction tables:
    Composite PK (paper_id, entity_id), WITHOUT ROWID for tight storage.
    ON DELETE CASCADE on both FK columns — deleting a paper cleans up its
    junction rows; deleting a catalog row cleans up its junction rows.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    JSON,
    Text,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from backend.db.database import Base


def _utc_now_iso() -> str:
    """Current UTC time formatted to match SQLite's ``datetime('now')`` output.

    Used as a Python-side ``default=`` so columns auto-populate on INSERT
    even when the column was added via ALTER TABLE (where SQLite can't store
    a non-constant DEFAULT).
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────────
# Papers
# ─────────────────────────────────────────────────────────────────────────────

class Paper(Base):
    __tablename__ = "papers"

    id              = Column(Integer, primary_key=True)
    doi             = Column(String, unique=True, nullable=False, index=True)
    title           = Column(String)
    journal         = Column(String, index=True)
    year            = Column(Integer, index=True)
    is_open_access  = Column(Boolean, default=False)
    entity_count    = Column(Integer, default=0)

    created_at = Column(
        String,
        server_default=func.datetime("now"),
        default=_utc_now_iso,
    )
    updated_at = Column(
        String,
        server_default=func.datetime("now"),
        default=_utc_now_iso,
        onupdate=_utc_now_iso,
    )

    # Reverse relationships — one per junction table. Cascade-delete cleans up
    # all junction rows when a paper is deleted (matches the ON DELETE CASCADE
    # at the SQL level).
    paper_chemicals             = relationship(
        "PaperChemical",            back_populates="paper", cascade="all, delete-orphan", passive_deletes=True)
    paper_species               = relationship(
        "PaperSpecies",             back_populates="paper", cascade="all, delete-orphan", passive_deletes=True)
    paper_locations             = relationship(
        "PaperLocation",            back_populates="paper", cascade="all, delete-orphan", passive_deletes=True)
    paper_plant_parts           = relationship(
        "PaperPlantPart",           back_populates="paper", cascade="all, delete-orphan", passive_deletes=True)
    paper_analytical_techniques = relationship(
        "PaperAnalyticalTechnique", back_populates="paper", cascade="all, delete-orphan", passive_deletes=True)
    paper_extraction_methods    = relationship(
        "PaperExtractionMethod",    back_populates="paper", cascade="all, delete-orphan", passive_deletes=True)
    paper_development_stages    = relationship(
        "PaperDevelopmentStage",    back_populates="paper", cascade="all, delete-orphan", passive_deletes=True)


# ─────────────────────────────────────────────────────────────────────────────
# Entity catalog tables (one per type)
# ─────────────────────────────────────────────────────────────────────────────

class Chemical(Base):
    __tablename__ = "chemicals"

    id                = Column(Integer, primary_key=True)
    canonical_text    = Column(String, nullable=False)   # always lowercased
    display_text      = Column(String, nullable=False)   # preferred casing

    # Chemistry metadata (from PubChem-style gazetteers)
    inchikey          = Column(String, index=True)
    smiles            = Column(Text)
    molecular_formula = Column(String, index=True)
    preferred_name    = Column(String)
    source_db         = Column(String)
    source_url        = Column(String)

    # Optional: JSON list of known aliases for UI "also known as" display
    aliases_json      = Column(JSON)

    created_at        = Column(String, server_default=func.datetime("now"), default=_utc_now_iso)

    paper_chemicals = relationship(
        "PaperChemical", back_populates="chemical", cascade="all, delete-orphan", passive_deletes=True)

    __table_args__ = (
        UniqueConstraint("canonical_text", name="uq_chemicals_canonical"),
    )


class Species(Base):
    __tablename__ = "species"

    id                       = Column(Integer, primary_key=True)
    canonical_text           = Column(String, nullable=False)
    display_text             = Column(String, nullable=False)

    # Taxonomy metadata (from GBIF-style gazetteers + curated SQLite)
    accepted_scientific_name = Column(String)
    common_name              = Column(String)
    family                   = Column(String, index=True)
    taxon_id                 = Column(String, index=True)
    name_type                = Column(String)            # "scientific" | "common"
    source_db                = Column(String)
    source_url               = Column(String)

    aliases_json             = Column(JSON)

    created_at               = Column(String, server_default=func.datetime("now"), default=_utc_now_iso)

    paper_species = relationship(
        "PaperSpecies", back_populates="species", cascade="all, delete-orphan", passive_deletes=True)

    __table_args__ = (
        UniqueConstraint("canonical_text", name="uq_species_canonical"),
    )


class Location(Base):
    __tablename__ = "locations"

    id             = Column(Integer, primary_key=True)
    canonical_text = Column(String, nullable=False)
    display_text   = Column(String, nullable=False)

    # Curated metadata (from your original Excel; not from any gazetteer)
    country        = Column(String, index=True)
    state          = Column(String, index=True)

    aliases_json   = Column(JSON)

    created_at     = Column(String, server_default=func.datetime("now"), default=_utc_now_iso)

    paper_locations = relationship(
        "PaperLocation", back_populates="location", cascade="all, delete-orphan", passive_deletes=True)

    __table_args__ = (
        UniqueConstraint("canonical_text", name="uq_locations_canonical"),
    )


# Simple entity types — no per-type metadata yet. Each has the same minimal shape:
# (id, canonical_text, display_text, aliases_json, created_at). Add ALTER TABLE
# columns as needed when a specific type gains structured metadata.

class PlantPart(Base):
    __tablename__ = "plant_parts"

    id             = Column(Integer, primary_key=True)
    canonical_text = Column(String, nullable=False)
    display_text   = Column(String, nullable=False)
    aliases_json   = Column(JSON)
    created_at     = Column(String, server_default=func.datetime("now"), default=_utc_now_iso)

    paper_plant_parts = relationship(
        "PaperPlantPart", back_populates="plant_part", cascade="all, delete-orphan", passive_deletes=True)

    __table_args__ = (
        UniqueConstraint("canonical_text", name="uq_plant_parts_canonical"),
    )


class AnalyticalTechnique(Base):
    __tablename__ = "analytical_techniques"

    id             = Column(Integer, primary_key=True)
    canonical_text = Column(String, nullable=False)
    display_text   = Column(String, nullable=False)
    aliases_json   = Column(JSON)
    created_at     = Column(String, server_default=func.datetime("now"), default=_utc_now_iso)

    paper_analytical_techniques = relationship(
        "PaperAnalyticalTechnique", back_populates="analytical_technique",
        cascade="all, delete-orphan", passive_deletes=True)

    __table_args__ = (
        UniqueConstraint("canonical_text", name="uq_analytical_techniques_canonical"),
    )


class ExtractionMethod(Base):
    __tablename__ = "extraction_methods"

    id             = Column(Integer, primary_key=True)
    canonical_text = Column(String, nullable=False)
    display_text   = Column(String, nullable=False)
    aliases_json   = Column(JSON)
    created_at     = Column(String, server_default=func.datetime("now"), default=_utc_now_iso)

    paper_extraction_methods = relationship(
        "PaperExtractionMethod", back_populates="extraction_method",
        cascade="all, delete-orphan", passive_deletes=True)

    __table_args__ = (
        UniqueConstraint("canonical_text", name="uq_extraction_methods_canonical"),
    )


class DevelopmentStage(Base):
    __tablename__ = "development_stages"

    id             = Column(Integer, primary_key=True)
    canonical_text = Column(String, nullable=False)
    display_text   = Column(String, nullable=False)
    aliases_json   = Column(JSON)
    created_at     = Column(String, server_default=func.datetime("now"), default=_utc_now_iso)

    paper_development_stages = relationship(
        "PaperDevelopmentStage", back_populates="development_stage",
        cascade="all, delete-orphan", passive_deletes=True)

    __table_args__ = (
        UniqueConstraint("canonical_text", name="uq_development_stages_canonical"),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Junction tables (one per entity type)
# ─────────────────────────────────────────────────────────────────────────────
#
# Composite PK (paper_id, entity_id) + WITHOUT ROWID saves ~5 bytes/row on
# what becomes the largest set of tables in the DB. Each junction also has
# an index on the entity-side ID for reverse lookups ("which papers mention
# this entity?").
#
# ON DELETE CASCADE on both FK columns means:
#   - Deleting a paper auto-removes all of its mention rows.
#   - Deleting a catalog entry (e.g. removing a chemical) auto-removes all
#     references to it from junctions.
#
# Junction rows should be created idempotently via
# INSERT ... ON CONFLICT(paper_id, entity_id) DO UPDATE — re-importing the
# same paper updates mention_count rather than duplicating rows.

class PaperChemical(Base):
    __tablename__ = "paper_chemicals"

    paper_id      = Column(Integer, ForeignKey("papers.id",    ondelete="CASCADE"), primary_key=True)
    chemical_id   = Column(Integer, ForeignKey("chemicals.id", ondelete="CASCADE"), primary_key=True)
    mention_count = Column(Integer, nullable=False, default=1)

    paper    = relationship("Paper",    back_populates="paper_chemicals")
    chemical = relationship("Chemical", back_populates="paper_chemicals")

    __table_args__ = (
        Index("idx_paper_chemicals_chemical_id", "chemical_id"),
        {"sqlite_with_rowid": False},
    )


class PaperSpecies(Base):
    __tablename__ = "paper_species"

    paper_id      = Column(Integer, ForeignKey("papers.id",  ondelete="CASCADE"), primary_key=True)
    species_id    = Column(Integer, ForeignKey("species.id", ondelete="CASCADE"), primary_key=True)
    mention_count = Column(Integer, nullable=False, default=1)

    paper   = relationship("Paper",   back_populates="paper_species")
    species = relationship("Species", back_populates="paper_species")

    __table_args__ = (
        Index("idx_paper_species_species_id", "species_id"),
        {"sqlite_with_rowid": False},
    )


class PaperLocation(Base):
    __tablename__ = "paper_locations"

    paper_id      = Column(Integer, ForeignKey("papers.id",    ondelete="CASCADE"), primary_key=True)
    location_id   = Column(Integer, ForeignKey("locations.id", ondelete="CASCADE"), primary_key=True)
    mention_count = Column(Integer, nullable=False, default=1)

    paper    = relationship("Paper",    back_populates="paper_locations")
    location = relationship("Location", back_populates="paper_locations")

    __table_args__ = (
        Index("idx_paper_locations_location_id", "location_id"),
        {"sqlite_with_rowid": False},
    )


class PaperPlantPart(Base):
    __tablename__ = "paper_plant_parts"

    paper_id      = Column(Integer, ForeignKey("papers.id",      ondelete="CASCADE"), primary_key=True)
    plant_part_id = Column(Integer, ForeignKey("plant_parts.id", ondelete="CASCADE"), primary_key=True)
    mention_count = Column(Integer, nullable=False, default=1)

    paper      = relationship("Paper",     back_populates="paper_plant_parts")
    plant_part = relationship("PlantPart", back_populates="paper_plant_parts")

    __table_args__ = (
        Index("idx_paper_plant_parts_plant_part_id", "plant_part_id"),
        {"sqlite_with_rowid": False},
    )


class PaperAnalyticalTechnique(Base):
    __tablename__ = "paper_analytical_techniques"

    paper_id                = Column(Integer, ForeignKey("papers.id",                ondelete="CASCADE"), primary_key=True)
    analytical_technique_id = Column(Integer, ForeignKey("analytical_techniques.id", ondelete="CASCADE"), primary_key=True)
    mention_count           = Column(Integer, nullable=False, default=1)

    paper                = relationship("Paper",               back_populates="paper_analytical_techniques")
    analytical_technique = relationship("AnalyticalTechnique", back_populates="paper_analytical_techniques")

    __table_args__ = (
        Index("idx_paper_analytical_techniques_at_id", "analytical_technique_id"),
        {"sqlite_with_rowid": False},
    )


class PaperExtractionMethod(Base):
    __tablename__ = "paper_extraction_methods"

    paper_id             = Column(Integer, ForeignKey("papers.id",             ondelete="CASCADE"), primary_key=True)
    extraction_method_id = Column(Integer, ForeignKey("extraction_methods.id", ondelete="CASCADE"), primary_key=True)
    mention_count        = Column(Integer, nullable=False, default=1)

    paper             = relationship("Paper",            back_populates="paper_extraction_methods")
    extraction_method = relationship("ExtractionMethod", back_populates="paper_extraction_methods")

    __table_args__ = (
        Index("idx_paper_extraction_methods_em_id", "extraction_method_id"),
        {"sqlite_with_rowid": False},
    )


class PaperDevelopmentStage(Base):
    __tablename__ = "paper_development_stages"

    paper_id             = Column(Integer, ForeignKey("papers.id",              ondelete="CASCADE"), primary_key=True)
    development_stage_id = Column(Integer, ForeignKey("development_stages.id",  ondelete="CASCADE"), primary_key=True)
    mention_count        = Column(Integer, nullable=False, default=1)

    paper             = relationship("Paper",            back_populates="paper_development_stages")
    development_stage = relationship("DevelopmentStage", back_populates="paper_development_stages")

    __table_args__ = (
        Index("idx_paper_development_stages_ds_id", "development_stage_id"),
        {"sqlite_with_rowid": False},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Legacy import-compatibility shims
# ─────────────────────────────────────────────────────────────────────────────
#
# Migration v2 replaced the old single-table ``Entity`` + ``PaperEntity``
# classes with per-type tables (Chemical/Species/Location/... and their
# matching paper_X junctions). Pre-migration consumer code still does
# ``from backend.db.models import Paper, Entity, PaperEntity`` — without
# these stubs the FastAPI app crashes at import time before it can even
# start serving traffic.
#
# These ``None`` placeholders let those legacy imports succeed. Any code
# that actually USES Entity / PaperEntity at runtime (e.g.,
# ``select(PaperEntity.label)``) will fail with AttributeError on
# ``NoneType`` — the right signal that the endpoint needs rewriting for
# per-type tables. Currently-broken endpoints (need migration to per-type):
#
#   * backend/api/dashboard.py        — /api/dashboard/{metrics,sunburst,graph3d}
#   * backend/api/paper.py            — _entity_row_to_dict, /paper/db/{doi}/entities
#   * backend/db/import_entities.py   — bulk Excel importer
#
# Delete these shims once all three consumers have been ported.
Entity = None        # type: ignore[assignment]
PaperEntity = None   # type: ignore[assignment]
