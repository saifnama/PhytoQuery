"""Search Router — Europe PMC literature search."""

from fastapi import APIRouter, Form, HTTPException
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
    page_size: int = Form(25),
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
            max_results=max(10, min(page_size, 100)),
            sort=sort,
            cursor_mark=cursor_mark,
        )
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=502, detail=str(e))
