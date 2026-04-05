"""
DOI Abstract Fetcher
---------------------
Fetches abstract and metadata for DOIs when Europe PMC doesn't have them.
Multi-source fallback: OpenAlex → PubMed.
"""

import re
import logging
from typing import Optional, Dict, Any
from backend.core.http_client import HttpClientManager

logger = logging.getLogger(__name__)


# --- Source Adapters ---


async def fetch_from_openalex(doi: str) -> Optional[Dict[str, Any]]:
    """Fetch metadata from OpenAlex API. Returns title/authors/journal even without abstract."""
    url = f"https://api.openalex.org/works/doi:{doi}"
    try:
        client = await HttpClientManager.get_client()
        response = await client.get(url, timeout=15.0)
        if response.status_code != 200:
            return None
        data = response.json()
        title = data.get("title", "")
        if not title:
            return None
        abstract = data.get("abstract_inverted_index")
        abstract_text = _reconstruct_abstract(abstract)
        return {
            "doi": doi,
            "title": title,
            "abstract": abstract_text,
            "authors": [
                a.get("author", {}).get("display_name", "")
                for a in data.get("authorships", [])
                if a.get("author", {}).get("display_name")
            ][:10],
            "year": data.get("publication_year"),
            "journal": data.get("primary_location", {})
            .get("source", {})
            .get("display_name", ""),
            "source": "OpenAlex",
            "url": data.get("doi"),
        }
    except Exception as e:
        logger.error(f"OpenAlex fetch failed for {doi}: {e}")
        return None


def _reconstruct_abstract(inverted_index: Optional[Dict[str, list]]) -> str:
    """Reconstruct abstract from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    # Build position -> word map
    position_map = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            position_map[pos] = word
    if not position_map:
        return ""
    # Sort by position and join
    max_pos = max(position_map.keys())
    words = [position_map.get(i, "") for i in range(max_pos + 1)]
    return " ".join(w for w in words if w)


async def fetch_from_pubmed(doi: str) -> Optional[Dict[str, Any]]:
    """Fetch abstract/metadata from PubMed E-utilities API.

    Strategy:
    1. Search for PMID by DOI
    2. Fetch abstract + metadata from PubMed XML
    """
    # Step 1: Search for PMID by DOI
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": f"{doi}[doi]",
        "retmode": "json",
    }
    try:
        client = await HttpClientManager.get_client()
        search_resp = await client.get(search_url, params=search_params, timeout=15.0)
        if search_resp.status_code != 200:
            return None
        search_data = search_resp.json()
        pmid_list = search_data.get("esearchresult", {}).get("idlist", [])
        if not pmid_list:
            return None
        pmid = pmid_list[0]

        # Step 2: Fetch abstract/metadata from PubMed XML
        fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        fetch_params = {
            "db": "pubmed",
            "id": pmid,
            "rettype": "xml",
            "retmode": "xml",
        }
        fetch_resp = await client.get(fetch_url, params=fetch_params, timeout=15.0)
        if fetch_resp.status_code != 200:
            return None

        from lxml import etree as ET

        root = ET.fromstring(fetch_resp.content)

        article = root.find(".//Article")
        if article is None:
            return _parse_pubmed_text(fetch_resp.text, pmid, doi)

        # Extract real DOI from PubMed XML
        real_doi = doi  # fallback
        article_ids = root.findall(".//ArticleId")
        for aid in article_ids:
            if aid.get("IdType") == "doi" and aid.text:
                real_doi = aid.text.strip()
                break

        # Extract title
        title_elem = article.find(".//ArticleTitle")
        title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

        # Extract abstract
        abstract_parts = article.findall(".//Abstract/AbstractText")
        abstract = ""
        if len(abstract_parts) > 1:
            abstract = "\n\n".join(
                "".join(p.itertext()).strip() for p in abstract_parts
            )
        elif abstract_parts:
            abstract = "".join(abstract_parts[0].itertext()).strip()

        if not abstract:
            return None

        # Extract authors
        authors = []
        author_list = article.findall(".//AuthorList/Author")
        for author in author_list:
            last_name = author.findtext("LastName", "")
            fore_name = author.findtext("ForeName", "")
            initials = author.findtext("Initials", "")
            name = (
                f"{last_name} {fore_name}".strip() or f"{last_name} {initials}".strip()
            )
            if name:
                authors.append(name)
        authors = authors[:10]

        # Extract journal name
        journal = article.findtext(".//Journal/Title", "")
        if not journal:
            journal = article.findtext(".//Journal/ISOAbbreviation", "")

        # Extract year
        pub_date = article.find(".//Journal/JournalIssue/PubDate")
        year = None
        if pub_date is not None:
            year_elem = pub_date.find("Year")
            if year_elem is not None and year_elem.text:
                try:
                    year = int(year_elem.text)
                except ValueError:
                    pass
            if year is None:
                medline_date = pub_date.findtext("MedlineDate", "")
                if medline_date:
                    year_match = re.search(r"(\d{4})", medline_date)
                    if year_match:
                        year = int(year_match.group(1))

        return {
            "doi": real_doi,
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "year": year,
            "journal": journal,
            "source": "PubMed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
        }
    except Exception as e:
        logger.error(f"PubMed fetch failed for {doi}: {e}")
        return None


async def _fetch_pmc_fulltext(pmcid: str, doi: str) -> Optional[Dict[str, Any]]:
    """Fetch full text XML from NCBI PMC using PMCID."""
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        "db": "pmc",
        "id": pmcid.replace("PMC", ""),
        "rettype": "xml",
        "retmode": "xml",
    }
    try:
        client = await HttpClientManager.get_client()
        resp = await client.get(url, params=params, timeout=15.0)
        if resp.status_code != 200:
            return None

        from lxml import etree as ET

        root = ET.fromstring(resp.content)

        # Extract title
        title = ""
        title_node = root.find(".//article-title")
        if title_node is not None:
            title = "".join(title_node.itertext()).strip()

        # Extract abstract
        abstract = ""
        abstract_node = root.find(".//abstract")
        if abstract_node is not None:
            abstract = "".join(abstract_node.itertext()).strip()

        # Extract authors
        authors = []
        contribs = root.findall(".//contrib-group/contrib[@contrib-type='author']")
        for contrib in contribs:
            surname = contrib.findtext("name/surname", "")
            given = contrib.findtext("name/given-names", "")
            name = f"{surname} {given}".strip()
            if name:
                authors.append(name)
        authors = authors[:10]

        # Extract journal
        journal = root.findtext(".//journal-title", "")
        if not journal:
            journal = root.findtext(".//abbrev-journal-title", "")

        # Extract year
        year = None
        pub_date = root.find(".//pub-date/year")
        if pub_date is not None and pub_date.text:
            try:
                year = int(pub_date.text)
            except ValueError:
                pass

        return {
            "doi": doi,
            "title": title,
            "abstract": abstract or "Full text available — see sections.",
            "authors": authors,
            "year": year,
            "journal": journal,
            "source": "PubMed",
            "url": f"https://pubmed.ncbi.nlm.nih.gov/",
            "full_text_xml": resp.text,
        }
    except Exception as e:
        logger.error(f"PMC full text fetch failed for {pmcid}: {e}")
        return None


async def _parse_pubmed_xml(root, pmid: str, doi: str) -> Optional[Dict[str, Any]]:
    """Extract abstract/metadata from PubMed XML."""
    article = root.find(".//Article")
    if article is None:
        return _parse_pubmed_text(root.text or "", pmid, doi)

    # Extract title
    title_elem = article.find(".//ArticleTitle")
    title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""

    # Extract abstract
    abstract_parts = article.findall(".//Abstract/AbstractText")
    abstract = ""
    if len(abstract_parts) > 1:
        abstract = "\n\n".join("".join(p.itertext()).strip() for p in abstract_parts)
    elif abstract_parts:
        abstract = "".join(abstract_parts[0].itertext()).strip()

    # Extract authors
    authors = []
    author_list = article.findall(".//AuthorList/Author")
    for author in author_list:
        last_name = author.findtext("LastName", "")
        fore_name = author.findtext("ForeName", "")
        initials = author.findtext("Initials", "")
        name = f"{last_name} {fore_name}".strip() or f"{last_name} {initials}".strip()
        if name:
            authors.append(name)
    authors = authors[:10]

    # Extract journal name
    journal = article.findtext(".//Journal/Title", "")
    if not journal:
        journal = article.findtext(".//Journal/ISOAbbreviation", "")

    # Extract year
    pub_date = article.find(".//Journal/JournalIssue/PubDate")
    year = None
    if pub_date is not None:
        year_elem = pub_date.find("Year")
        if year_elem is not None and year_elem.text:
            try:
                year = int(year_elem.text)
            except ValueError:
                pass
        if year is None:
            medline_date = pub_date.findtext("MedlineDate", "")
            if medline_date:
                year_match = re.search(r"(\d{4})", medline_date)
                if year_match:
                    year = int(year_match.group(1))

    return {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "year": year,
        "journal": journal,
        "source": "PubMed",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


def _parse_pubmed_text(text: str, pmid: str, doi: str) -> Optional[Dict[str, Any]]:
    """Fallback parser for plain text PubMed response."""
    lines = text.strip().split("\n")
    title = lines[0].strip() if lines else ""
    if not title:
        return None
    abstract = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""
    return {
        "doi": doi,
        "title": title,
        "abstract": abstract,
        "authors": [],
        "year": None,
        "journal": "",
        "source": "PubMed",
        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
    }


async def fetch_from_semantic_scholar(doi: str) -> Optional[Dict[str, Any]]:
    """Fetch metadata from Semantic Scholar API. Returns title/authors/journal even without abstract."""
    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "title,abstract,authors,year,venue,externalIds,openAccessPdf"}
    try:
        client = await HttpClientManager.get_client()
        response = await client.get(url, params=params, timeout=15.0)
        if response.status_code != 200:
            return None
        data = response.json()
        title = data.get("title", "")
        if not title:
            return None
        return {
            "doi": doi,
            "title": title,
            "abstract": data.get("abstract", ""),
            "authors": [
                a.get("name", "") for a in data.get("authors", []) if a.get("name")
            ][:10],
            "year": data.get("year"),
            "journal": data.get("venue", ""),
            "source": "Semantic Scholar",
            "url": f"https://www.semanticscholar.org/paper/{doi}",
        }
    except Exception as e:
        logger.error(f"Semantic Scholar fetch failed for {doi}: {e}")
        return None


# --- Main Fetcher with Fallback ---


async def fetch_doi_abstract(doi: str) -> Optional[Dict[str, Any]]:
    """
    Fetch metadata for a DOI with multi-source fallback.
    Order: OpenAlex → Semantic Scholar → PubMed
    Returns metadata (title, authors, journal, year) even if no abstract.
    """
    # Normalize DOI
    doi = doi.strip().lower()
    if doi.startswith("https://doi.org/"):
        doi = doi.replace("https://doi.org/", "")
    elif doi.startswith("http://doi.org/"):
        doi = doi.replace("http://doi.org/", "")
    elif doi.startswith("doi:"):
        doi = doi[4:].strip()

    if not doi:
        return None

    # Try each source in order — prefer one with abstract, but accept metadata-only
    sources = [
        fetch_from_openalex,
        fetch_from_semantic_scholar,
        fetch_from_pubmed,
    ]

    best_result = None
    for fetch_fn in sources:
        result = await fetch_fn(doi)
        if result:
            logger.info(f"Found metadata for {doi} via {result['source']}")
            if result.get("abstract"):
                return result  # Prefer abstract
            if best_result is None:
                best_result = result  # Keep as fallback

    if best_result:
        return best_result

    logger.warning(f"No metadata found for DOI: {doi}")
    return None
