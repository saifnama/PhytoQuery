from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from collections import defaultdict
from typing import Any

from backend.db.database import get_db
from backend.db.models import Paper, PaperEntity

router = APIRouter(prefix="/api/dashboard", tags=["Dashboard"])


@router.get("/metrics")
async def get_dashboard_metrics(db: AsyncSession = Depends(get_db)):
    """Fetch all aggregated metrics for the homepage dashboard."""
    try:
        total_papers_result = await db.execute(select(func.count(Paper.id)))
        total_papers = total_papers_result.scalar() or 0

        total_entities_result = await db.execute(select(func.count(PaperEntity.id)))
        total_entities = total_entities_result.scalar() or 0

        total_journals_result = await db.execute(
            select(func.count(func.distinct(Paper.journal))).where(Paper.journal != None)
        )
        total_journals = total_journals_result.scalar() or 0

        top_3_journals_result = await db.execute(
            select(Paper.journal)
            .where(Paper.journal != None)
            .group_by(Paper.journal)
            .order_by(desc(func.count(Paper.id)))
            .limit(3)
        )
        top_3_journals = [row[0] for row in top_3_journals_result.all()]

        papers_by_journal_result = await db.execute(
            select(Paper.journal, func.count(Paper.id).label("count"))
            .where(Paper.journal != None)
            .group_by(Paper.journal)
            .order_by(desc("count"))
            .limit(8)
        )
        papers_by_journal = [{"name": row[0], "value": row[1]} for row in papers_by_journal_result.all()]

        entity_distribution_result = await db.execute(
            select(PaperEntity.label, func.count(PaperEntity.id).label("count"))
            .group_by(PaperEntity.label)
            .order_by(desc("count"))
        )
        entity_distribution = [{"name": row[0].title(), "value": row[1]} for row in entity_distribution_result.all()]

        papers_by_year_result = await db.execute(
            select(Paper.year, func.count(Paper.id).label("count"))
            .where(Paper.year != None)
            .group_by(Paper.year)
            .order_by(Paper.year)
        )
        papers_by_year = [{"name": str(row[0]), "value": row[1]} for row in papers_by_year_result.all()]

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
                "geo_distribution": geo_distribution
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sunburst")
async def get_sunburst_data(db: AsyncSession = Depends(get_db)):
    """
    Hierarchical entity data for the sunburst chart.
    Structure: Entity Type → Entity Name → Paper Count (mention count)
    """
    try:
        result = await db.execute(
            select(
                PaperEntity.label,
                PaperEntity.canonical_text,
                func.sum(PaperEntity.mention_count).label("value")
            )
            .group_by(PaperEntity.label, PaperEntity.canonical_text)
            .order_by(PaperEntity.label, desc("value"))
        )
        rows = result.all()

        # Group by label (entity type)
        type_map: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in rows:
            label = row[0]
            name = row[1]
            value = row[2] or 0
            type_map[label][name] = value

        children = []
        for label, entities in sorted(type_map.items()):
            children.append({
                "name": label.title(),
                "children": [
                    {"name": name, "value": val}
                    for name, val in sorted(entities.items(), key=lambda x: -x[1])
                ]
            })

        return {"name": "Entities", "children": children}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/graph3d")
async def get_graph3d_data(db: AsyncSession = Depends(get_db)):
    """
    3D Knowledge Graph: Chemical + Species + Location entities across all papers.
    Nodes: entities (Chemical/Species/Location)
    Edges: entity co-appears in same paper → linked. Paper DOIs are NOT nodes
    (they act as implicit hubs — multiple chemicals linked by same paper are connected).
    """
    try:
        INCLUDED_LABELS = {'CHEMICAL', 'SPECIES', 'LOCATION'}

        # Fetch entities we care about
        result = await db.execute(
            select(
                PaperEntity.paper_id,
                PaperEntity.label,
                PaperEntity.canonical_text,
                PaperEntity.mention_count,
            )
            .where(PaperEntity.label.in_(INCLUDED_LABELS))
        )
        rows = result.all()

        if not rows:
            return {"nodes": [], "links": []}

        # Build entity nodes
        node_map: dict[str, dict] = {}
        for paper_id, label, canonical, count in rows:
            key = f"{label}::{canonical}"
            if key not in node_map:
                node_map[key] = {
                    "id": key,
                    "name": canonical,
                    "label": label,  # CHEMICAL | SPECIES | LOCATION
                    "count": 0,
                    "paper_ids": set(),
                }
            node_map[key]["count"] += (count or 0)
            node_map[key]["paper_ids"].add(paper_id)

        # Build co-occurrence: entities in same paper
        # Group entities by paper_id first
        paper_entity_keys: dict[int, list[str]] = defaultdict(list)
        for paper_id, label, canonical, _ in rows:
            key = f"{label}::{canonical}"
            if key not in paper_entity_keys[paper_id]:
                paper_entity_keys[paper_id].append(key)

        # Link every pair of entities in the same paper
        link_map: dict[tuple, int] = defaultdict(int)
        for paper_id, keys in paper_entity_keys.items():
            keys_sorted = sorted(keys)
            for i in range(len(keys_sorted)):
                for j in range(i + 1, len(keys_sorted)):
                    link_map[(keys_sorted[i], keys_sorted[j])] += 1

        nodes = [
            {
                "id": k,
                "name": v["name"],
                "label": v["label"],
                "count": v["count"],
                "paper_count": len(v["paper_ids"]),
            }
            for k, v in sorted(node_map.items(), key=lambda x: -x[1]["count"])
        ]

        # Filter: only keep links with weight >= 2 (entity appears in >= 2 papers together)
        links = [
            {"source": k[0], "target": k[1], "weight": w}
            for k, w in sorted(link_map.items(), key=lambda x: -x[1])
            if w >= 2
        ]

        return {"nodes": nodes, "links": links}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))