"""Europe PMC Service - Facade combining client and parser.

This is the main entry point for Europe PMC operations. It delegates to:
- EuropePMCClient for HTTP API calls
- XMLParser for JATS XML parsing
- JATSConverter for JATS-to-HTML conversion
"""

import hashlib
import logging
import os
from typing import Dict, Any, List, Optional, Tuple

from backend.services.europe_pmc.client import EuropePMCClient
from backend.services.europe_pmc.parser import XMLParser, JATSConverter
from backend.core.caching import pmc_cache
from backend.core.sanitizer import sanitize

logger = logging.getLogger(__name__)

# PARSER_VERSION: a hash of this file to help invalidate cached rendered HTML
# whenever the parser logic changes.
try:
    _this_file = os.path.abspath(__file__)
    with open(_this_file, "rb") as _f:
        PARSER_VERSION = hashlib.md5(_f.read()).hexdigest()[:8]
except Exception:
    PARSER_VERSION = "dev"


def _render_cache_key_for(identifier_value: str) -> str:
    """Return a cache key for rendered HTML, namespaced by parser version."""
    return f"rendered|{identifier_value}|ver={PARSER_VERSION}"


class EuropePMCService:
    """Facade for Europe PMC operations. Delegates to client and parser modules."""

    BASE_URL = EuropePMCClient.BASE_URL

    # --- Delegate to client ---

    @staticmethod
    def parse_identifier(raw_input: str) -> Tuple[str, str]:
        return EuropePMCClient.parse_identifier(raw_input)

    @classmethod
    async def fetch_full_text(cls, pmcid: str) -> Optional[str]:
        return await EuropePMCClient.fetch_full_text(pmcid)

    @classmethod
    async def fetch_paper_data(cls, doi: str) -> Tuple[Optional[str], str]:
        return await EuropePMCClient.fetch_paper_data(doi)

    @classmethod
    async def resolve_pdf_url(cls, identifier: str) -> Optional[Dict[str, str]]:
        return await EuropePMCClient.resolve_pdf_url(identifier)

    @classmethod
    async def search_literature(
        cls,
        query: str,
        filters: dict,
        max_results: int = 25,
        sort: str = "",
        cursor_mark: str = "*",
    ) -> Dict[str, Any]:
        return await EuropePMCClient.search_literature(
            query, filters, max_results, sort, cursor_mark
        )

    # --- Delegate to parser ---

    @staticmethod
    def extract_title_from_xml(xml_content: str) -> str:
        return XMLParser.extract_title_from_xml(xml_content)

    @staticmethod
    def extract_authors_from_xml(xml_content: str) -> list:
        return XMLParser.extract_authors_from_xml(xml_content)

    @staticmethod
    def extract_journal_from_xml(xml_content: str) -> str:
        return XMLParser.extract_journal_from_xml(xml_content)

    @staticmethod
    def extract_date_from_xml(xml_content: str) -> str:
        return XMLParser.extract_date_from_xml(xml_content)

    @staticmethod
    def extract_toc_from_html(html_content: str) -> List[Dict[str, Any]]:
        return XMLParser.extract_toc_from_html(html_content)

    @staticmethod
    def parse_sections_from_xml(
        xml_content: str, pmcid: str = ""
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        return XMLParser.parse_sections_from_xml(xml_content, pmcid)

    # --- Delegate to JATS converter ---

    @staticmethod
    def _jats_inline_to_html(text: str) -> str:
        return JATSConverter.inline_to_html(text)

    @staticmethod
    def _ensure_xmlns(xml_content: str) -> str:
        return JATSConverter.ensure_xmlns(xml_content)

    @staticmethod
    def clean_xml(xml_content: str) -> str:
        return JATSConverter.clean_xml(xml_content)

    # --- Service orchestration ---

    @classmethod
    async def fetch_structured_data(cls, doi: str) -> Dict[str, Any]:
        """
        Fetch paper and return structured sections and reference metadata.
        This path also caches the rendered HTML and TOC for faster future hits.
        """
        # Quick path: if we already have a rendered HTML cache for this identifier,
        # return it immediately to avoid re-parsing XML on subsequent requests.
        id_type, id_value = cls.parse_identifier(doi)
        render_key = _render_cache_key_for(id_value)
        cached_render = pmc_cache.get(render_key)
        if cached_render:
            return {
                "sections": cached_render.get("sections", []),
                "html": cached_render.get("html", ""),
                "mode": cached_render.get("mode", ""),
                "title": cached_render.get("title", ""),
                "references": cached_render.get("references", {}),
                "pmcid": cached_render.get("pmcid", ""),
                "toc": cached_render.get("toc", []),
                "journal": cached_render.get("journal", ""),
                "authors": cached_render.get("authors", []),
                "doi": cached_render.get("doi", ""),
                "date": cached_render.get("date", ""),
                "fallback_source": "Europe PMC",  # Always set when cached
                "fallback_url": f"https://europepmc.org/article/{id_value}",
            }

        # Proceed with normal processing if no cached rendered HTML is available
        id_type, id_value = cls.parse_identifier(doi)
        text, mode = await cls.fetch_paper_data(doi)

        # Read cached metadata (may have been populated by fetch_paper_data)
        cached_paper = pmc_cache.get(id_value) or {}
        paper_title_from_cache = cached_paper.get("title", "")
        paper_journal_from_cache = cached_paper.get("journal", "")
        paper_authors_from_cache = cached_paper.get("authors", [])
        paper_doi_from_cache = cached_paper.get("doi", "")
        paper_date_from_cache = cached_paper.get("date", "")

        if mode == "error" or not text:
            return {
                "sections": [],
                "html": "",
                "mode": mode,
                "title": paper_title_from_cache,
                "references": {},
                "pmcid": id_value if id_type == "pmcid" else "",
                "toc": [],
                "journal": paper_journal_from_cache,
                "authors": paper_authors_from_cache,
                "doi": paper_doi_from_cache,
                "date": paper_date_from_cache,
                "fallback_source": "",  # No source available
            }

        # If we have a PMCID, we need it for figure image URLs
        pmcid = id_value if id_type == "pmcid" else ""

        if mode == "abstract":
            # Convert JATS inline tags in abstract to HTML before sanitizing
            abstract_html = cls._jats_inline_to_html(text)
            abstract_html = sanitize(abstract_html)
            # Extract TOC from the abstract HTML
            toc = cls.extract_toc_from_html(abstract_html)
            # Extract paper title from metadata (single parse pass)
            paper_title = ""
            if text.startswith("<?xml"):
                metadata = XMLParser.extract_metadata_from_xml(text)
                paper_title = metadata["title"]
                if not paper_journal_from_cache:
                    paper_journal_from_cache = metadata["journal"]
                if not paper_authors_from_cache:
                    paper_authors_from_cache = metadata["authors"]
                if not paper_date_from_cache:
                    paper_date_from_cache = metadata["date"]
            return {
                "sections": [
                    {"title": "Abstract", "content": abstract_html, "headings": []}
                ],
                "html": abstract_html,
                "mode": mode,
                "title": paper_title or paper_title_from_cache,
                "references": {},
                "pmcid": pmcid,
                "toc": toc,
                "journal": paper_journal_from_cache,
                "authors": paper_authors_from_cache,
                "doi": paper_doi_from_cache,
                "date": paper_date_from_cache,
                "fallback_source": "Europe PMC",  # Has abstract
            }

        # Parse XML for sections, title, and references
        # If we didn't start with PMCID, we might find it in the XML or search metadata
        if not pmcid:
            try:
                from lxml import etree as ET

                root = ET.fromstring(cls._ensure_xmlns(text).encode("utf-8"))
                pmc_node = root.find(".//article-id[@pub-id-type='pmc']")
                if pmc_node is not None:
                    pmcid = pmc_node.text
                    if not pmcid.startswith("PMC"):
                        pmcid = f"PMC{pmcid}"
            except Exception:
                pass

        sections, references = cls.parse_sections_from_xml(text, pmcid=pmcid)

        # Extract ALL metadata (title, authors, journal, date) in a SINGLE parse pass
        paper_title = ""
        if text.startswith("<?xml"):
            metadata = XMLParser.extract_metadata_from_xml(text)
            paper_title = metadata["title"]
            if not paper_journal_from_cache:
                paper_journal_from_cache = metadata["journal"]
            if not paper_authors_from_cache:
                paper_authors_from_cache = metadata["authors"]
            if not paper_date_from_cache:
                paper_date_from_cache = metadata["date"]
        if not paper_title:
            paper_title = paper_title_from_cache

        if not sections:
            return {
                # DEPRECATED: sections array is kept for backward compatibility.
                # New code should use 'html' (unified HTML blob) and 'toc' (table of contents) instead.
                "sections": [{"title": "Full Text", "content": text, "headings": []}],
                "html": text,
                "mode": mode,
                "title": paper_title,
                "references": references,
                "pmcid": pmcid,
                "toc": [],
                "journal": paper_journal_from_cache,
                "authors": paper_authors_from_cache,
                "doi": paper_doi_from_cache,
                "date": paper_date_from_cache,
                "fallback_source": "Europe PMC",  # Has full text
            }

        # Sanitize all section contents before returning
        for s in sections:
            if isinstance(s, dict) and "content" in s:
                s["content"] = sanitize(s["content"])
        # Build full HTML blob by concatenating all section contents in order
        full_html = "".join(s.get("content", "") for s in sections)
        toc = cls.extract_toc_from_html(full_html)
        # Cache the rendered HTML + TOC for future fast access.
        try:
            cache_payload = {
                "sections": sections,
                "html": full_html,
                "mode": mode,
                "title": paper_title,
                "references": references,
                "pmcid": pmcid,
                "toc": toc,
                "journal": paper_journal_from_cache,
                "authors": paper_authors_from_cache,
                "doi": paper_doi_from_cache,
                "date": paper_date_from_cache,
            }
            pmc_cache.set(render_key, cache_payload)
        except Exception:
            pass

        return {
            # DEPRECATED: sections array is kept for backward compatibility.
            # New code should use 'html' (unified HTML blob) and 'toc' (table of contents) instead.
            "sections": sections,
            "html": full_html,
            "mode": mode,
            "title": paper_title,
            "references": references,
            "pmcid": pmcid,
            "toc": toc,
            "journal": paper_journal_from_cache,
            "authors": paper_authors_from_cache,
            "doi": paper_doi_from_cache,
            "date": paper_date_from_cache,
            "fallback_source": "Europe PMC",  # Has full text
        }
