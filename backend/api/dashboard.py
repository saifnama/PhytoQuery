from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from backend.db.database import get_db
from backend.db.models import Paper, PaperEntity

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])

@router.get("/metrics")
async def get_dashboard_metrics(db: AsyncSession = Depends(get_db)):
    """Fetch all aggregated metrics for the homepage dashboard."""
    try:
        # 1. KPI: Total Papers
        total_papers_result = await db.execute(select(func.count(Paper.id)))
        total_papers = total_papers_result.scalar() or 0

        # 2. KPI: Total Entities
        total_entities_result = await db.execute(select(func.count(PaperEntity.id)))
        total_entities = total_entities_result.scalar() or 0

        # 3. KPI: Journals Indexed
        total_journals_result = await db.execute(select(func.count(func.distinct(Paper.journal))).where(Paper.journal != None))
        total_journals = total_journals_result.scalar() or 0

        # Top 3 Journals for KPI subtitle
        top_3_journals_result = await db.execute(
            select(Paper.journal)
            .where(Paper.journal != None)
            .group_by(Paper.journal)
            .order_by(desc(func.count(Paper.id)))
            .limit(3)
        )
        top_3_journals = [row[0] for row in top_3_journals_result.all()]

        # 4. Chart: Papers by Journal
        papers_by_journal_result = await db.execute(
            select(Paper.journal, func.count(Paper.id).label("count"))
            .where(Paper.journal != None)
            .group_by(Paper.journal)
            .order_by(desc("count"))
            .limit(8)
        )
        papers_by_journal = [{"name": row[0], "value": row[1]} for row in papers_by_journal_result.all()]

        # 5. Chart: Entity Type Distribution
        entity_distribution_result = await db.execute(
            select(PaperEntity.label, func.count(PaperEntity.id).label("count"))
            .group_by(PaperEntity.label)
            .order_by(desc("count"))
        )
        entity_distribution = [{"name": row[0].title(), "value": row[1]} for row in entity_distribution_result.all()]

        # 6. Chart: Papers by Year
        papers_by_year_result = await db.execute(
            select(Paper.year, func.count(Paper.id).label("count"))
            .where(Paper.year != None)
            .group_by(Paper.year)
            .order_by(Paper.year)
        )
        papers_by_year = [{"name": str(row[0]), "value": row[1]} for row in papers_by_year_result.all()]

        # 7. Chart: Open Access Distribution
        oa_result = await db.execute(
            select(Paper.is_open_access, func.count(Paper.id).label("count"))
            .group_by(Paper.is_open_access)
        )
        oa_distribution = [{"name": "Open Access" if row[0] else "Restricted", "value": row[1]} for row in oa_result.all()]

        # 8. Chart: Geographic Distribution (Papers by Country)
        # We extract 'country' from the metadata_json of LOCATION entities
        geo_result = await db.execute(
            select(
                func.json_extract(PaperEntity.metadata_json, '$.country').label("country"),
                func.count(func.distinct(PaperEntity.paper_id)).label("count")
            )
            .where(PaperEntity.label == 'LOCATION')
            .group_by("country")
            .order_by(desc("count"))
        )
        geo_distribution = [{"name": row[0], "value": row[1]} for row in geo_result.all() if row[0]]

        return {
            "kpis": {
                "total_papers": total_papers,
                "total_entities": total_entities,
                "total_journals": total_journals,
                "top_journals": ", ".join(top_3_journals)
            },
            "charts": {
                "papers_by_journal": papers_by_journal,
                "entity_distribution": entity_distribution,
                "papers_by_year": papers_by_year,
                "oa_distribution": oa_distribution,
                "geo_distribution": geo_distribution
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
