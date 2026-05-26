"""
SQLAlchemy models for the PhytoQuery two-table schema.

The live database has exactly two user tables:

    papers           — one row per paper (DOI + display metadata)
    paper_entities   — one row per (paper, label, canonical_text) fact

This file mirrors that schema directly. Earlier attempts at a per-type
catalog schema (chemicals + species + locations + ... + 7 junction tables)
were abandoned; that DDL is no longer in the database. The simpler shape
matches the design diagram: the DB is a thin annotation layer that the
frontend lists / drills into. Paper content is fetched live from
Europe PMC / OpenAlex / Semantic Scholar at click time; entity metadata
enrichment (chemical SMILES, species taxonomy) comes from the gazetteer
CSV files at render time. The ``metadata`` column on ``paper_entities``
stores any structured payload kept at ingest time (e.g.
``{"family": "Verbenaceae"}`` for species rows, ``{"country": "..."}``
for locations); the frontend may ignore it and prefer the CSV.

Naming quirk:
    The SQL column ``metadata`` is exposed under the Python attribute
    ``meta`` because ``metadata`` is reserved on the declarative ``Base``
    class (``Base.metadata`` is the MetaData object). The
    ``Column("metadata", ...)`` form keeps the on-disk column name
    intact; only the Python-side accessor differs.
"""

from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    JSON,
    ForeignKey,
    CheckConstraint,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from backend.db.database import Base


class Paper(Base):
    __tablename__ = "papers"

    id              = Column(Integer, primary_key=True)
    doi             = Column(String, nullable=False, unique=True, index=True)
    title           = Column(String)
    journal         = Column(String, index=True)
    year            = Column(Integer, index=True)
    is_open_access  = Column(Boolean)
    # Denormalized cache. Maintained explicitly by writers (see
    # backend/db/import_entities.py); no DB-level trigger.
    entity_count    = Column(Integer)

    paper_entities = relationship(
        "PaperEntity",
        back_populates="paper",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )


class PaperEntity(Base):
    __tablename__ = "paper_entities"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    paper_id       = Column(
        Integer,
        ForeignKey("papers.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    label          = Column(String, nullable=False, index=True)
    canonical_text = Column(String, nullable=False)
    frequency      = Column(Integer)
    # SQL column named "metadata" (post-rename). Exposed in Python as
    # "meta" so it doesn't shadow Base.metadata. See module docstring.
    meta           = Column("metadata", JSON)

    paper = relationship("Paper", back_populates="paper_entities")

    __table_args__ = (
        UniqueConstraint(
            "paper_id", "label", "canonical_text",
            name="idx_paper_entities_uniq",
        ),
        CheckConstraint(
            "metadata IS NULL OR json_valid(metadata)",
            name="paper_entities_metadata_check",
        ),
    )
