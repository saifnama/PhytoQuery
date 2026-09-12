"""Dashboard router — aggregates over the two-table schema.

Single endpoint ``/metrics`` returns KPIs + chart data for the homepage
dashboard. Every count is a UNIQUE count (no mention-row duplication):

  * ``total_papers``         — distinct papers (UNIQUE(doi) enforced)
  * ``total_entities``       — distinct ``(label, canonical_text)`` tuples
  * ``total_journals``       — DISTINCT journal names
  * ``papers_by_journal``    — distinct papers per journal
  * ``entity_distribution``  — distinct canonical_text values per label
                                 (powers the entity donut)
  * ``papers_by_year``       — distinct papers per year
  * ``geo_distribution``     — distinct papers per country (heatmap)

All SQL lives in ``backend.src.db.repository``; this module is HTTP only.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.db import repository
from backend.src.db.session import get_db

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/metrics")
async def get_dashboard_metrics(db: AsyncSession = Depends(get_db)):
    """Aggregated metrics for the homepage dashboard."""
    try:
        return await repository.dashboard_metrics(db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
