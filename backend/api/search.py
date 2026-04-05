"""Search Router — Europe PMC literature search."""

from fastapi import APIRouter, Form
from backend.services.europe_pmc import EuropePMCService
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
):
    """JSON endpoint for searching literature."""
    filters = {
        "open_access": open_access,
        "has_full_text": has_full_text,
        "article_type": article_type,
    }
    try:
        return await EuropePMCService.search_literature(
            query=query,
            filters=filters,
            max_results=25,
            sort=sort,
            cursor_mark=cursor_mark,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return {"error": str(e), "results": [], "pagination": {}}
