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

The ``/sunburst`` and ``/graph3d`` endpoints used to live here too; they
were removed because the frontend doesn't render them and their value
metric was ``SUM(frequency)`` which mixed mention-counts with the
"uniqueness" principle the rest of the dashboard uses.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from backend.db.database import get_db
from backend.db.models import Paper, PaperEntity

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/metrics")
async def get_dashboard_metrics(db: AsyncSession = Depends(get_db)):
    """Aggregated metrics for the homepage dashboard."""
    try:
        # ---- KPIs ---------------------------------------------------------
        total_papers = (await db.execute(
            select(func.count(Paper.id))
        )).scalar() or 0

        # Count UNIQUE entities across the corpus — one (label, canonical_text)
        # tuple = one entity, regardless of how many papers reference it.
        # The previous COUNT(*) FROM paper_entities counted mention-rows,
        # which is misleading (e.g. "alpha-pinene" appearing in 200 papers
        # would inflate the chemical total by ~200x).
        total_entities = (await db.execute(
            select(func.count()).select_from(
                select(PaperEntity.label, PaperEntity.canonical_text)
                .distinct()
                .subquery()
            )
        )).scalar() or 0

        total_journals = (await db.execute(
            select(func.count(func.distinct(Paper.journal)))
            .where(Paper.journal.is_not(None))
        )).scalar() or 0

        top_3_journals_rows = (await db.execute(
            select(Paper.journal)
            .where(Paper.journal.is_not(None))
            .group_by(Paper.journal)
            .order_by(desc(func.count(Paper.id)))
            .limit(3)
        )).all()
        top_3_journals = [row[0] for row in top_3_journals_rows]

        # ---- Charts -------------------------------------------------------
        papers_by_journal_rows = (await db.execute(
            select(Paper.journal, func.count(Paper.id).label("count"))
            .where(Paper.journal.is_not(None))
            .group_by(Paper.journal)
            .order_by(desc("count"))
            .limit(10)
        )).all()
        papers_by_journal = [
            {"name": row[0], "value": row[1]} for row in papers_by_journal_rows
        ]

        # Entity distribution by label — count DISTINCT canonical_text per
        # label so the donut shows "how many unique chemicals/species/..."
        # rather than total mention-rows (which would over-report by the
        # average number of papers each entity appears in).
        entity_distribution_rows = (await db.execute(
            select(
                PaperEntity.label,
                func.count(func.distinct(PaperEntity.canonical_text)).label("count"),
            )
            .group_by(PaperEntity.label)
            .order_by(desc("count"))
        )).all()
        entity_distribution = [
            {"name": (row[0] or "").title(), "value": row[1]}
            for row in entity_distribution_rows
        ]

        papers_by_year_rows = (await db.execute(
            select(Paper.year, func.count(Paper.id).label("count"))
            .where(Paper.year.is_not(None))
            .group_by(Paper.year)
            .order_by(Paper.year)
        )).all()
        papers_by_year = [
            {"name": str(row[0]), "value": row[1]} for row in papers_by_year_rows
        ]

        # Geographic distribution — country lives inside the metadata JSON
        # on LOCATION rows now (no typed column). SQLite's json_extract is
        # exposed by SQLAlchemy via func.json_extract. The count is distinct
        # papers per country (not duplicated LOCATION rows).
        country_expr = func.json_extract(PaperEntity.meta, "$.country").label("country")
        geo_rows = (await db.execute(
            select(
                country_expr,
                func.count(func.distinct(PaperEntity.paper_id)).label("count"),
            )
            .where(PaperEntity.label == "LOCATION")
            .where(country_expr.is_not(None))
            .group_by("country")
            .order_by(desc("count"))
        )).all()
        geo_distribution = [
            {"name": row[0], "value": row[1]} for row in geo_rows if row[0]
        ]

        return {
            "kpis": {
                "total_papers": total_papers,
                "total_entities": total_entities,
                "total_journals": total_journals,
                "top_journals": ", ".join(top_3_journals),
            },
            "charts": {
                "papers_by_journal": papers_by_journal,
                "entity_distribution": entity_distribution,
                "papers_by_year": papers_by_year,
                "geo_distribution": geo_distribution,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
