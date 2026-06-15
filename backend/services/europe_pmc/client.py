"""Europe PMC HTTP API client.

Handles all network communication with Europe PMC:
- Identifier parsing (DOI, PMCID, PMID)
- Paper data fetching
- Literature search
- Full text XML retrieval
"""

import asyncio
import re
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, List
from xml.etree import ElementTree

import httpx

from backend.core.http_client import HttpClientManager
from backend.core.caching import pmc_cache

logger = logging.getLogger(__name__)


class EuropePMCClient:
    """HTTP client for Europe PMC API."""

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

    @staticmethod
    def _build_search_query(id_type: str, id_value: str) -> str:
        if id_type == "pmcid":
            return f"PMCID:{id_value}"
        if id_type == "pmid":
            return f"EXT_ID:{id_value}"
        return f"DOI:{id_value}"

    @staticmethod
    def _normalize_remote_url(url: str) -> str:
        if url.startswith("ftp://"):
            return "https://" + url[len("ftp://") :]
        return url

    @staticmethod
    def _build_pdf_filename(
        title: str, doi: str, pmcid: str, fallback_id: str
    ) -> str:
        base = title.strip() or doi.strip() or pmcid.strip() or fallback_id.strip() or "paper"
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._")
        if not safe:
            safe = "paper"
        return f"{safe[:120]}.pdf"

    @staticmethod
    def parse_identifier(raw_input: str) -> Tuple[str, str]:
        """
        Smart identifier parser. Accepts many formats, returns (type, value).

        Supported inputs:
          - https://doi.org/10.1038/nature12373         -> ("doi", "10.1038/nature12373")
          - http://dx.doi.org/10.1038/nature12373        -> ("doi", "10.1038/nature12373")
          - doi:10.1038/nature12373                      -> ("doi", "10.1038/nature12373")
          - 10.1038/nature12373                          -> ("doi", "10.1038/nature12373")
          - https://europepmc.org/article/MED/23903748   -> ("pmid", "23903748")
          - https://europepmc.org/article/PMC/PMC4221854 -> ("pmcid", "PMC4221854")
          - https://pubmed.ncbi.nlm.nih.gov/23903748     -> ("pmid", "23903748")
          - https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4221854 -> ("pmcid", "PMC4221854")
          - PMC4221854                                   -> ("pmcid", "PMC4221854")
          - 23903748                                     -> ("pmid", "23903748")
        """
        val = raw_input.strip()

        # 1. Europe PMC article URL
        m = re.match(
            r"https?://(?:www\.)?europepmc\.org/article/PMC/(PMC\d+)",
            val,
            re.IGNORECASE,
        )
        if m:
            return "pmcid", m.group(1)
        m = re.match(
            r"https?://(?:www\.)?europepmc\.org/article/MED/(\d+)", val, re.IGNORECASE
        )
        if m:
            return "pmid", m.group(1)

        # 2. PubMed URL
        m = re.match(
            r"https?://(?:www\.)?pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", val, re.IGNORECASE
        )
        if m:
            return "pmid", m.group(1)

        # 3. NCBI PMC URL (both www.ncbi.nlm.nih.gov and pmc.ncbi.nlm.nih.gov)
        m = re.match(
            r"https?://(?:www\.)?ncbi\.nlm\.nih\.gov/pmc/articles/(PMC\d+)",
            val,
            re.IGNORECASE,
        )
        if m:
            return "pmcid", m.group(1).upper()
        m = re.match(
            r"https?://pmc\.ncbi\.nlm\.nih\.gov/articles/(PMC\d+)",
            val,
            re.IGNORECASE,
        )
        if m:
            return "pmcid", m.group(1).upper()

        # 4. DOI URL
        m = re.match(r"https?://(dx\.)?doi\.org/(.+)", val)
        if m:
            return "doi", m.group(2).strip()

        # 5. doi: prefix
        m = re.match(r"^doi:(.+)", val, re.IGNORECASE)
        if m:
            return "doi", m.group(1).strip()

        # 6. Bare PMCID (starts with PMC)
        if re.match(r"^PMC\d+$", val, re.IGNORECASE):
            return "pmcid", val.upper()

        # 7. Bare DOI (starts with 10.)
        if val.startswith("10."):
            return "doi", val

        # 8. Pure number -> assume PMID
        if val.isdigit():
            return "pmid", val

        # 9. Bare DOI (must be actual DOI format: 10.xxxx/xxxx)
        if val.startswith("10.") and "/" in val:
            return "doi", val

        # 10. Not an identifier - return empty
        return None, val

    @classmethod
    async def fetch_full_text(cls, pmcid: str) -> Optional[str]:
        """Fetch full text XML from PMC. Tries Europe PMC first, then NCBI PMC."""
        # Try Europe PMC first
        url = f"{cls.BASE_URL}/{pmcid}/fullTextXML"
        try:
            client = await HttpClientManager.get_client()
            response = await client.get(url, timeout=30.0)
            if response.status_code == 200 and response.text.strip():
                return response.text
        except Exception as e:
            logger.debug(f"Europe PMC full text failed for {pmcid}: {e}")

        # Fallback to NCBI PMC
        pmc_id = pmcid.replace("PMC", "")
        url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {"db": "pmc", "id": pmc_id, "rettype": "xml", "retmode": "xml"}
        try:
            client = await HttpClientManager.get_client()
            response = await client.get(url, params=params, timeout=30.0)
            if response.status_code == 200 and response.text.strip():
                # NCBI returns a <pmc-articleset> wrapper — extract the article
                if "<pmc-articleset>" in response.text:
                    # Strip the wrapper to get clean JATS XML
                    text = response.text
                    text = text.replace("<pmc-articleset>", "").replace(
                        "</pmc-articleset>", ""
                    )
                    # Add XML declaration and root element if missing
                    if not text.startswith("<?xml"):
                        text = (
                            '<?xml version="1.0"?>\n<article>\n' + text + "\n</article>"
                        )
                    return text
                return response.text
        except Exception as e:
            logger.debug(f"NCBI PMC full text failed for {pmcid}: {e}")

        return None

    @classmethod
    async def resolve_pdf_url(cls, identifier: str) -> Optional[Dict[str, str]]:
        """Resolve a downloadable PDF URL for a paper identifier."""
        id_type, id_value = cls.parse_identifier(identifier)
        search_url = f"{cls.BASE_URL}/search"
        params = {
            "query": cls._build_search_query(id_type, id_value),
            "format": "json",
            "resultType": "core",
        }

        client = await HttpClientManager.get_client()
        response = await client.get(search_url, params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        results = data.get("resultList", {}).get("result", [])
        if not results:
            return None

        result = results[0]
        pmcid = (result.get("pmcid") or "").strip()
        doi = (result.get("doi") or "").strip()
        title = (result.get("title") or "").strip()
        filename = cls._build_pdf_filename(title, doi, pmcid, id_value)

        full_text_urls = result.get("fullTextUrlList", {}).get("fullTextUrl", [])
        if isinstance(full_text_urls, dict):
            full_text_urls = [full_text_urls]

        for item in full_text_urls:
            document_style = str(item.get("documentStyle", "")).lower()
            url = str(item.get("url", "")).strip()
            if document_style == "pdf" and url:
                return {
                    "url": cls._normalize_remote_url(url),
                    "filename": filename,
                    "source": str(item.get("site", "Europe PMC")),
                }

        if not pmcid:
            return None

        oa_url = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
        oa_response = await client.get(oa_url, params={"id": pmcid}, timeout=30.0)
        if oa_response.status_code != 200 or not oa_response.text.strip():
            return None

        try:
            root = ElementTree.fromstring(oa_response.text)
        except ElementTree.ParseError:
            return None

        for link in root.findall(".//record/link"):
            if str(link.get("format", "")).lower() != "pdf":
                continue
            href = (link.get("href") or "").strip()
            if href:
                return {
                    "url": cls._normalize_remote_url(href),
                    "filename": filename,
                    "source": "PMC OA",
                }

        return None

    @classmethod
    async def fetch_paper_data(cls, doi: str) -> Tuple[Optional[str], str]:
        """
        Fetch paper data by identifier. Accepts DOI, PMCID, PMID, or URLs.
        Returns: (text, mode)
        """
        id_type, id_value = cls.parse_identifier(doi)

        # Check cache using original normalized value
        cached = pmc_cache.get(id_value)
        if cached:
            return cached["text"], cached["mode"]

        try:
            client = await HttpClientManager.get_client()

            # --- Fast path: PMCID -> search for metadata first, then try full text ---
            if id_type == "pmcid":
                query = f"PMCID:{id_value}"
            elif id_type == "pmid":
                query = f"EXT_ID:{id_value}"
            else:
                query = f"DOI:{id_value}"

            # Always search first to get metadata (DOI, journal, title)
            search_url = f"{cls.BASE_URL}/search"
            params = {"query": query, "format": "json", "resultType": "core"}

            response = await client.get(search_url, params=params, timeout=30.0)
            response.raise_for_status()
            data = response.json()

            results = data.get("resultList", {}).get("result", [])
            if not results:
                # Not an error - paper may not be in Europe PMC (fallback to OpenAlex will handle it)
                return None, "error"

            result = results[0]
            pmcid = result.get("pmcid")
            abstract = result.get("abstractText", "")
            title = result.get("title", "")
            doi = result.get("doi", "")
            # Journal can be in journalTitle or journalInfo.journal.title
            journal = result.get("journalTitle", "")
            if not journal:
                journal_info = result.get("journalInfo", {})
                if journal_info:
                    journal = journal_info.get("journal", {}).get("title", "")
            # Authors from authorString
            authors_str = result.get("authorString", "")
            authors = (
                [a.strip() for a in authors_str.split(",") if a.strip()]
                if authors_str
                else []
            )
            # Publication date — use Europe PMC's firstPublicationDate when available
            raw_date = result.get("firstPublicationDate") or result.get("publicationDate", "")
            if raw_date:
                try:
                    # Europe PMC may return YYYY-MM-DD, YYYY-MM, or YYYY
                    parts = raw_date.split("-")
                    if len(parts) == 3:
                        dt = datetime.strptime(raw_date, "%Y-%m-%d")
                        pub_date = dt.strftime("%d %B %Y")
                    elif len(parts) == 2:
                        dt = datetime.strptime(raw_date, "%Y-%m")
                        pub_date = dt.strftime("%d %B %Y")
                    else:
                        pub_date = raw_date  # Already just year or unformatted
                except Exception:
                    pub_date = raw_date
            if not pub_date:
                # Fallback: construct from journalInfo (defaults to day=01)
                journal_info = result.get("journalInfo", {})
                month = journal_info.get("monthOfPublication")
                year = journal_info.get("yearOfPublication")
                if month and year:
                    try:
                        if isinstance(month, int):
                            month_num = month
                        else:
                            month_num = datetime.strptime(month[:3], "%b").month
                        pub_date = datetime.strptime(f"{year}-{month_num:02d}", "%Y-%m").strftime("%d %B %Y")
                    except Exception:
                        pub_date = str(year)
                elif year:
                    pub_date = str(year)
                else:
                    pub_year = result.get("pubYear")
                    if pub_year:
                        pub_date = str(pub_year)

            # If PMCID exists, try to get full text XML (regardless of OA flag)
            if pmcid:
                full_text = await cls.fetch_full_text(pmcid)
                if full_text:
                    pmc_cache.set(
                        id_value,
                        {
                            "text": full_text,
                            "mode": "full_text",
                            "title": title,
                            "journal": journal,
                            "doi": doi,
                            "authors": authors,
                            "date": pub_date,
                        },
                    )
                    return full_text, "full_text"

            # Fallback to abstract
            if abstract:
                logger.info(
                    f"Falling back to abstract for {id_type.upper()}: {id_value}"
                )
                pmc_cache.set(
                    id_value,
                    {
                        "text": abstract,
                        "mode": "abstract",
                        "title": title,
                        "journal": journal,
                        "doi": doi,
                        "authors": authors,
                        "date": pub_date,
                    },
                )
                return abstract, "abstract"

            logger.debug(f"Empty content for {id_type.upper()}: {id_value}")
            # Cache metadata even with no abstract/full text, including text=None and mode="empty"
            # so future fetch_paper_data calls can return gracefully with metadata
            try:
                pmc_cache.set(
                    id_value,
                    {
                        "text": None,
                        "mode": "empty",
                        "title": title,
                        "journal": journal,
                        "authors": authors,
                        "doi": doi,
                        "date": pub_date,
                    },
                )
            except Exception:
                pass  # Don't fail if cache set fails
            return None, "empty"

        except Exception as e:
            logger.error(f"Error fetching from Europe PMC: {e}")
            return None, "error"

    @classmethod
    async def search_literature(
        cls,
        query: str,
        filters: dict,
        max_results: int = 25,
        sort: str = "",
        cursor_mark: str = "*",
    ) -> Dict[str, Any]:
        """
        Search Europe PMC literature using advanced filters and sorting.
        Uses cursor-based pagination (Europe PMC API uses cursorMark, not page numbers).
        Returns results + pagination metadata.
        """
        search_parts = []
        if query and query.strip():
            search_parts.append(f"({query.strip()})")

        if filters.get("open_access"):
            search_parts.append("OPEN_ACCESS:y")
        if filters.get("has_full_text"):
            search_parts.append("HAS_FT:y")
        if filters.get("article_type"):
            search_parts.append(f'PUB_TYPE:"{filters["article_type"]}"')

        # Advanced Sorting logic in Europe PMC is often handled by appending keywords to query
        if sort == "cited":
            search_parts.append("sort_cited:y")
        elif sort == "date":
            search_parts.append("sort_date:y")

        final_query = " AND ".join(search_parts)
        if not final_query:
            return {
                "results": [],
                "pagination": {
                    "total": 0,
                    "cursorMark": "*",
                    "nextCursorMark": "",
                    "hasMore": False,
                    "pageSize": max_results,
                },
            }
        search_url = f"{cls.BASE_URL}/search"
        params = {
            "query": final_query,
            "format": "json",
            "resultType": "core",
            "pageSize": max_results,
            "cursorMark": cursor_mark,
        }

        max_retries = 3
        base_delay = 1.5
        max_retry_delay = 10.0

        try:
            client = await HttpClientManager.get_client()

            response = None
            for attempt in range(max_retries):
                try:
                    response = await client.get(search_url, params=params, timeout=30.0)
                    response.raise_for_status()
                    break
                except httpx.HTTPStatusError as e:
                    status_code = e.response.status_code if e.response is not None else None
                    is_transient = status_code is not None and 500 <= status_code < 600
                    if not is_transient or attempt == max_retries - 1:
                        raise

                    retry_after_header = e.response.headers.get("retry-after")
                    if retry_after_header:
                        try:
                            delay = min(float(retry_after_header), max_retry_delay)
                        except ValueError:
                            delay = base_delay * (2**attempt)
                    else:
                        delay = base_delay * (2**attempt)

                    logger.warning(
                        "Europe PMC search failed with %s. Retrying in %.1fs (attempt %s/%s)",
                        status_code,
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)
                except (httpx.TimeoutException, httpx.RequestError) as e:
                    if attempt == max_retries - 1:
                        raise

                    delay = base_delay * (2**attempt)
                    logger.warning(
                        "Europe PMC search transient error (%s). Retrying in %.1fs (attempt %s/%s)",
                        type(e).__name__,
                        delay,
                        attempt + 1,
                        max_retries,
                    )
                    await asyncio.sleep(delay)

            if response is None:
                raise RuntimeError("Europe PMC search returned no response after retries")

            data = response.json()

            # Pagination metadata from Europe PMC (cursor-based)
            hit_count = data.get("hitCount", 0)
            next_cursor = data.get("nextCursorMark", "")
            has_more = next_cursor != cursor_mark and next_cursor != ""

            results = data.get("resultList", {}).get("result", [])
            formatted_results = []
            for r in results:
                # Journal can be in journalTitle or journalInfo.journal.title
                journal = r.get("journalTitle", "")
                if not journal:
                    journal_info = r.get("journalInfo", {})
                    if journal_info:
                        journal = journal_info.get("journal", {}).get("title", "")
                if not journal:
                    journal = "Unknown journal"
                
                # Check for full text availability
                has_full_text = r.get("hasFullText") == "Y" or r.get("isOpenAccess") == "Y"
                
                formatted_results.append(
                    {
                        "id": r.get("id"),
                        "pmcid": r.get("pmcid"),
                        "doi": r.get("doi"),
                        "pmid": r.get("pmid"),
                        "title": r.get("title", "No title available"),
                        "authors": r.get("authorString", "Unknown authors"),
                        "journal": journal,
                        "year": r.get("pubYear", "Unknown year"),
                        "citationCount": r.get("citedByCount", 0),
                        "isOpenAccess": r.get("isOpenAccess") == "Y",
                        "hasTextMinedTerms": r.get("hasTextMinedTerms") == "Y",
                        "hasFullText": has_full_text,
                        "hasPdfUrl": has_full_text,  # Europe PMC OA means likely PDF
                        "pdfUrl": None,  # PDF URL resolved on paper fetch
                        "abstract": r.get("abstractText", ""),
                        "source": "Europe PMC",
                    }
                )
            return {
                "results": formatted_results,
                "pagination": {
                    "total": hit_count,
                    "cursorMark": cursor_mark,
                    "nextCursorMark": next_cursor,
                    "hasMore": has_more,
                    "pageSize": max_results,
                },
            }
        except Exception as e:
            logger.error(f"Error in search_literature: {e}")
            raise
