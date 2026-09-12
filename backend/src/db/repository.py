"""Database repository — ALL SQL lives here. Routers do no SQL.

Move-only extraction (Phase 2b-1): query strings/plans are byte-identical
to the router originals. Behaviour changes (LIKE-escaping, covering
indexes, journal-list bounds) are separate follow-ups.
"""
import json
from typing import Any, Optional

from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.src.db.models import Paper, PaperEntity


def _maybe_json_load(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return None
    return value


def entity_row_to_dict(pe: PaperEntity) -> dict:
    """PaperEntity row → API entity dict (frontend reads flat keys)."""
    meta_raw = _maybe_json_load(pe.meta)
    metadata: dict = meta_raw if isinstance(meta_raw, dict) else {}

    aliases = metadata.pop("aliases", None) if isinstance(metadata, dict) else None
    if not isinstance(aliases, list):
        aliases = []

    return {
        "label": pe.label,
        "text": pe.canonical_text,
        "canonical": pe.canonical_text,
        "count": pe.frequency,
        "aliases": aliases,
        **metadata,
    }


async def dashboard_metrics(db: AsyncSession) -> dict:
    """KPIs + chart data. Every count is UNIQUE (see router docstring)."""
    total_papers = (await db.execute(
        select(func.count(Paper.id))
    )).scalar() or 0

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

    papers_by_journal_rows = (await db.execute(
        select(Paper.journal, func.count(Paper.id).label("count"))
        .where(Paper.journal.is_not(None))
        .group_by(Paper.journal)
        .order_by(desc("count"))
    )).all()
    papers_by_journal = [
        {"name": row[0], "value": row[1]} for row in papers_by_journal_rows
    ]

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


async def list_papers(
    db: AsyncSession,
    *,
    limit: int,
    offset: int,
    country: Optional[str],
    query: Optional[str],
    year: Optional[int],
) -> tuple[int, list[dict[str, Any]]]:
    """Paginated paper list + total. ponytail: `%query%` unescaped (LIKE-escaping follow-up)."""
    select_stmt = select(Paper)

    if country:
        country_expr = func.json_extract(PaperEntity.meta, "$.country")
        select_stmt = select_stmt.join(PaperEntity).where(PaperEntity.label == "LOCATION").where(country_expr == country)

    if query:
        select_stmt = select_stmt.where(
            Paper.title.ilike(f"%{query}%") | Paper.journal.ilike(f"%{query}%")
        )

    if year is not None:
        select_stmt = select_stmt.where(Paper.year == year)

    select_stmt = select_stmt.group_by(Paper.id).order_by(desc(Paper.id))

    result = await db.execute(select_stmt.limit(limit).offset(offset))
    papers = result.scalars().all()

    if country or query or year is not None:
        subq = select_stmt.limit(None).offset(None).subquery()
        count_stmt = select(func.count()).select_from(subq)
    else:
        count_stmt = select(func.count(Paper.id))

    count_result = await db.execute(count_stmt)
    total_count = count_result.scalar() or 0

    paper_list = [
        {
            "id": p.id,
            "doi": p.doi,
            "title": p.title,
            "journal": p.journal,
            "year": p.year,
            "is_open_access": p.is_open_access,
            "entity_count": p.entity_count,
        }
        for p in papers
    ]
    return total_count, paper_list


async def paper_id_by_doi(db: AsyncSession, clean_id: str, raw_doi: str) -> Optional[int]:
    result = await db.execute(select(Paper.id).where(Paper.doi == clean_id))
    paper_id = result.scalar_one_or_none()
    if not paper_id and clean_id != raw_doi:
        result = await db.execute(select(Paper.id).where(Paper.doi == raw_doi))
        paper_id = result.scalar_one_or_none()
    return paper_id


async def entities_for_paper(db: AsyncSession, paper_id: int) -> list[dict]:
    result = await db.execute(
        select(PaperEntity).where(PaperEntity.paper_id == paper_id)
    )
    return [entity_row_to_dict(pe) for pe in result.scalars().all()]
