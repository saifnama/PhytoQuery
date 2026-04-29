"""Search Router — merged scholarly literature search."""

from fastapi import APIRouter, Form, HTTPException
from backend.services.search_service import SearchService
import logging

router = APIRouter(prefix="/search", tags=["search"])
logger = logging.getLogger(__name__)


@router.post("/json")
async def search_papers_json(
    query: str = Form(""),
    open_access: bool = Form(False),
    has_full_text: bool = Form(False),
    article_type: str = Form(""),
    sort: str = Form(""),
    page: int = Form(1),
    cursor_mark: str = Form("*"),
    source: str = Form("europepmc"),  # "europepmc" or "openalex" (default: europepmc)
):
    """JSON endpoint for searching literature.

    source param:
    - "europepmc" = Europe PMC only (default)
    - "openalex" = OpenAlex only
    """
    # Normalize source, default to europepmc
    source = source.lower() if source else "europepmc"
    if source not in ("europepmc", "openalex"):
        source = "europepmc"

    filters = {
        "open_access": open_access,
        "has_full_text": has_full_text,
        "article_type": article_type,
    }
    try:
        return await SearchService.search_literature(
            query=query,
            filters=filters,
            page_size=25,  # Fixed at 25
            page=max(1, page),
            sort=sort,
            source=source,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))


@router.get("/types")
async def get_article_types(source: str = "openalex"):
    """Get available article types for a search source.

    - source="openalex": Returns OpenAlex work types with live counts
    - source="europepmc": Returns Europe PMC publication types
    """
    source = source.lower() if source else "openalex"

    if source == "openalex":
        try:
            types = await SearchService.get_openalex_type_counts()
            return {"types": types}
        except Exception as e:
            logger.error(f"Failed to fetch OpenAlex types: {e}")
            raise HTTPException(status_code=502, detail=str(e))

    # Europe PMC types (static list)
    return {
        "types": [
            {"key": "", "display_name": "Any Type", "count": None},
            {"key": "Research-article", "display_name": "Research Article", "count": None},
            {"key": "Review", "display_name": "Review", "count": None},
        ]
    }
