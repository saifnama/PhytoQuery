from sqlalchemy import Column, Integer, String, Boolean, JSON, DateTime, ForeignKey, Index
from sqlalchemy.orm import relationship
from datetime import datetime, timezone
from backend.db.database import Base

class Paper(Base):
    __tablename__ = "papers"
    
    id = Column(Integer, primary_key=True, index=True)
    doi = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=True)
    journal = Column(String, index=True, nullable=True)
    year = Column(Integer, index=True, nullable=True)
    is_open_access = Column(Boolean, default=False)
    entity_count = Column(Integer, default=0)

    entities = relationship("PaperEntity", back_populates="paper", cascade="all, delete-orphan")

class PaperEntity(Base):
    __tablename__ = "paper_entities"
    
    id = Column(Integer, primary_key=True, index=True)
    paper_id = Column(Integer, ForeignKey("papers.id"), nullable=False, index=True)
    label = Column(String, index=True, nullable=False)  # e.g., CHEMICAL, SPECIES, PLANT PART
    canonical_text = Column(String, index=True, nullable=False)
    mention_count = Column(Integer, default=1)
    
    # Optional metadata specific to the entity instance
    metadata_json = Column(JSON, nullable=True)
    
    paper = relationship("Paper", back_populates="entities")

# Compound index for super-fast dashboard aggregations (e.g., top chemicals)
Index('idx_entity_label_text', PaperEntity.label, PaperEntity.canonical_text)
