"""OpenAlex Service - Facade for OpenAlex operations."""

from typing import Dict, Any

from backend.services.openalex.client import OpenAlexClient


class OpenAlexService:
    """Facade for OpenAlex operations."""

    @classmethod
    async def fetch_paper(cls, doi: str) -> Dict[str, Any]:
        """Fetch paper metadata from OpenAlex by DOI."""
        return await OpenAlexClient.fetch_paper(doi)