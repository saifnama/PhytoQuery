"""OpenAlex client for fetching paper metadata by DOI."""

import logging
from typing import Dict, Any, Optional, List

from backend.core.http_client import HttpClientManager

logger = logging.getLogger(__name__)

BASE_URL = "https://api.openalex.org/works"


def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    # Build position -> word map (first occurrence wins)
    position_to_word: Dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            if pos not in position_to_word:
                position_to_word[pos] = word
    if not position_to_word:
        return ""
    # Sort by position and join (keep empty positions as spaces)
    max_pos = max(position_to_word.keys())
    words = [position_to_word.get(i, "") for i in range(max_pos + 1)]
    return " ".join(words).strip()


class OpenAlexClient:
    """Client for OpenAlex API."""

    BASE_URL = BASE_URL

    @classmethod
    async def fetch_paper(cls, doi: str) -> Dict[str, Any]:
        """Fetch paper metadata from OpenAlex API by DOI."""
        doi = doi.strip().lower()
        if doi.startswith("https://doi.org/"):
            doi = doi.replace("https://doi.org/", "")
        elif doi.startswith("http://doi.org/"):
            doi = doi.replace("http://doi.org/", "")
        elif doi.startswith("doi:"):
            doi = doi[4:].strip()

        if not doi:
            return {}

        url = f"{cls.BASE_URL}/doi:{doi}"
        try:
            client = await HttpClientManager.get_client()
            response = await client.get(url, timeout=15.0)
            if response.status_code != 200:
                logger.warning(f"OpenAlex returned {response.status_code} for DOI: {doi}")
                return {}
            data = response.json()
            
            title = data.get("title", "")
            if not title:
                return {}
            
            # Reconstruct abstract from inverted index
            abstract_index = data.get("abstract_inverted_index")
            abstract = _reconstruct_abstract(abstract_index)
            
            # Get PDF URL from best_oa_location
            best_oa = data.get("best_oa_location") or {}
            pdf_url = best_oa.get("pdf_url") or ""
            
            return {
                "doi": doi,
                "title": title,
                "abstract": abstract,
                "authors": [
                    a.get("author", {}).get("display_name", "")
                    for a in data.get("authorships", [])
                    if a.get("author", {}).get("display_name")
                ][:10],
                "year": data.get("publication_year"),
                "journal": data.get("primary_location", {})
                    .get("source", {})
                    .get("display_name", ""),
                "pmcid": data.get("ids", {}).get("pmcid", ""),
                "pmid": "",
                "source": "OpenAlex",
                "url": data.get("doi"),
                "pdfUrl": pdf_url,
            }
        except Exception as e:
            logger.error(f"OpenAlex fetch failed for {doi}: {e}")
            return {}