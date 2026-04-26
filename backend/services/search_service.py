import asyncio
import hashlib
import logging
import math
import random
import re
import time
from datetime import date
from typing import Any, Dict, List, Optional

from backend.core.http_client import HttpClientManager
from backend.services.europe_pmc import EuropePMCService

logger = logging.getLogger(__name__)

# Simple in-memory cache for search results
_SEARCH_CACHE: Dict[str, Dict[str, Any]] = {}
_CACHE_TTL = 300  # 5 minutes


class SearchService:
    OPENALEX_BASE_URL = "https://api.openalex.org"

# Bucketed shuffle config
    BUCKET_THRESHOLDS = [0.85, 0.6]  # R >= 0.85, 0.6-0.85, <0.6
    W_CITATION = 0.08  # Citation boost weight
    W_RECENCY = 0.05   # Recency boost weight
    ALPHA = 0.15       # Shuffle noise (inside buckets)

    @classmethod
    def _compute_tf_idf_score(cls, item: Dict[str, Any], query: str) -> float:
        """
        Compute TF-IDF style relevance score (0-1).
        Checks query terms in title/abstract.
        """
        if not query:
            return 0.5
            
        query_terms = query.lower().split()
        title = (item.get("title") or "").lower()
        abstract = (item.get("abstract") or "").lower()
        
        # Count matching terms
        matches = sum(1 for term in query_terms if term in title or term in abstract)
        
        # Normalize by query length + title length factor
        base_score = matches / max(len(query_terms), 1)
        # Bonus for title matches (higher weight)
        title_bonus = sum(0.1 for term in query_terms if term in title)
        
        return min(1.0, base_score + title_bonus)

    @classmethod
    def _normalize_vector(cls, values: List[float]) -> List[float]:
        """Min-max normalize to 0-1."""
        if not values:
            return []
        min_v = min(values)
        max_v = max(values)
        if max_v - min_v < 0.0001:
            return [0.5] * len(values)
        return [(v - min_v) / (max_v - min_v) for v in values]

    @classmethod
    def _compute_recency_score(cls, year_str: str) -> float:
        """Compute recency score based on publication year."""
        try:
            year = int(re.sub(r"\D", "", str(year_str)))
            current_year = date.today().year
            age = current_year - year
            # Sigmoid-like: recent = 1, older = 0
            if age <= 0:
                return 1.0
            if age >= 20:
                return 0.0
            return 1.0 / (1.0 + age / 5)  # decay over ~5 years
        except:
            return 0.5

    @classmethod
    def _compute_hybrid_score(cls, item: Dict[str, Any], query: str, query_citations: List[float], query_recency: List[float], index: int) -> float:
        """
        Compute hybrid relevance score: TF-IDF + citation boost + recency boost.
        Base score: S0 = (1-w_c-w_r)*R + w_c*C + w_r*T
        """
        # Base relevance (TF-IDF style)
        r = cls._compute_tf_idf_score(item, query)
        
        # Citation boost (normalized)
        citations = item.get("citationCount") or 0
        c = query_citations[index] if index < len(query_citations) else 0.5
        
        # Recency boost (normalized)
        recency = query_recency[index] if index < len(query_recency) else 0.5
        
        # Combine
        w_c = cls.W_CITATION
        w_r = cls.W_RECENCY
        s0 = (1 - w_c - w_r) * r + w_c * c + w_r * recency
        
        return min(1.0, max(0.0, s0))

    @classmethod
    def _get_bucket(cls, relevance: float) -> int:
        """Assign bucket index based on relevance thresholds."""
        if relevance >= cls.BUCKET_THRESHOLDS[0]:
            return 0  # Top bucket
        if relevance >= cls.BUCKET_THRESHOLDS[1]:
            return 1
        return 2  # Lower bucket

    @classmethod
    def _seeded_random(cls, query: str, index: int, seed_suffix: str = "") -> float:
        """Generate seeded random number for reproducibility."""
        today = date.today().isoformat()
        seed_str = f"{today}:{seed_suffix}:{query}:{index}"
        seed_bytes = hashlib.sha256(seed_str.encode()).digest()
        seed_int = int.from_bytes(seed_bytes[:4], 'big')
        random.seed(seed_int)
        return random.random()

    @classmethod
    def _bucketed_shuffle(
        cls,
        results: List[Dict[str, Any]],
        query: str,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid scoring + bucketed shuffle with seeded noise.
        
        Pipeline:
        1. Compute TF-IDF relevance for query
        2. Normalize citations + recency
        3. Compute hybrid score S0
        4. Bucket by S0, add noise within bucket
        5. Return sorted by score
        """
        if not results:
            return results
        
        n = len(results)
        
        # Compute citation scores and normalize
        citation_scores = [math.log(1 + (r.get("citationCount") or 0)) for r in results]
        normalized_citations = cls._normalize_vector(citation_scores)
        
        # Compute recency scores and normalize
        recency_scores = [cls._compute_recency_score(r.get("year", "")) for r in results]
        normalized_recency = cls._normalize_vector(recency_scores)
        
        # Compute hybrid scores
        hybrid_scores = []
        for i, item in enumerate(results):
            score = cls._compute_hybrid_score(item, query, normalized_citations, normalized_recency, i)
            bucket = cls._get_bucket(score)
            
            # Add seeded noise within bucket
            rand_val = cls._seeded_random(query, i, f"bucket_{bucket}")
            noise = cls.ALPHA * (1 - score) * (rand_val - 0.5)  # -0.5 to 0.5
            final_score = min(1.0, max(0.0, score + noise))
            
            hybrid_scores.append({
                "item": item,
                "score": final_score,
                "bucket": bucket,
            })
        
        # Sort by final score descending
        hybrid_scores.sort(key=lambda x: x["score"], reverse=True)
        
        return [x["item"] for x in hybrid_scores]

    @staticmethod
    def _map_openalex_type(article_type: str) -> str:
        normalized = (article_type or "").strip().lower()
        if normalized == "research-article":
            return "article"
        if normalized == "review":
            return "review"
        return normalized

    @staticmethod
    def _normalize_doi(value: Optional[str]) -> str:
        if not value:
            return ""
        normalized = value.strip().lower()
        normalized = re.sub(r"^https?://(dx\.)?doi\.org/", "", normalized)
        normalized = re.sub(r"^doi:", "", normalized)
        return normalized.strip()

    @staticmethod
    def _normalize_pmid(value: Optional[str]) -> str:
        if not value:
            return ""
        digits = re.findall(r"\d+", value)
        return digits[0] if digits else ""

    @staticmethod
    def _normalize_pmcid(value: Optional[str]) -> str:
        if not value:
            return ""
        match = re.search(r"PMC\d+", value.upper())
        return match.group(0) if match else ""

    @classmethod
    def _build_dedupe_key(cls, item: Dict[str, Any]) -> str:
        doi = cls._normalize_doi(item.get("doi"))
        if doi:
            return f"doi:{doi}"

        pmid = cls._normalize_pmid(item.get("pmid"))
        if pmid:
            return f"pmid:{pmid}"

        pmcid = cls._normalize_pmcid(item.get("pmcid"))
        if pmcid:
            return f"pmcid:{pmcid}"

        title = re.sub(r"\s+", " ", (item.get("title") or "").strip().lower())
        year = str(item.get("year") or "")
        return f"title:{title}|year:{year}"

    @staticmethod
    def _coerce_year(value: Any) -> str:
        if value is None:
            return "Unknown year"
        text = str(value).strip()
        return text or "Unknown year"

    @staticmethod
    def _join_authors(authorships: List[Dict[str, Any]]) -> str:
        names = [
            authorship.get("author", {}).get("display_name", "").strip()
            for authorship in authorships or []
            if authorship.get("author", {}).get("display_name", "").strip()
        ]
        return ", ".join(names[:10]) if names else "Unknown authors"

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
        if not inverted_index:
            return ""
        
        # Each position maps to exactly one word (first occurrence)
        position_to_word: Dict[int, str] = {}
        for word, positions in inverted_index.items():
            for pos in positions:
                if pos not in position_to_word:
                    position_to_word[pos] = word
        
        if not position_to_word:
            return ""
        
        max_position = max(position_to_word.keys())
        words = [position_to_word.get(i, "") for i in range(max_position + 1)]
        return " ".join(words).strip()

    @classmethod
    async def search_openalex(
        cls,
        query: str,
        filters: Dict[str, Any],
        max_results: int,
        page: int,
        sort: str = "",
    ) -> Dict[str, Any]:
        if not query.strip():
            return {"results": [], "pagination": {"total": 0, "page": page, "pageSize": max_results, "hasMore": False}}

        params: Dict[str, Any] = {
            "search": query.strip(),
            "per_page": max_results,
            "page": max(1, page),
        }

        filter_parts: List[str] = []
        
        # Always apply these filters for quality papers
        filter_parts.append("has_doi:true")  # Must have DOI
        filter_parts.append("has_content.pdf:true")  # Must have PDF
        filter_parts.append("is_oa:true")  # Must be Open Access
        filter_parts.append("primary_topic.domain.id:1")  # Biology/Life Sciences
        
        if filters.get("article_type"):
            filter_parts.append(f"type:{cls._map_openalex_type(filters['article_type'])}")
        
        if filter_parts:
            params["filter"] = ",".join(filter_parts)

        if sort == "cited":
            params["sort"] = "cited_by_count:desc"
        elif sort == "date":
            params["sort"] = "publication_date:desc"

        client = await HttpClientManager.get_client()
        response = await client.get(f"{cls.OPENALEX_BASE_URL}/works", params=params, timeout=30.0)
        response.raise_for_status()
        data = response.json()

        results = []
        for work in data.get("results", []):
            ids = work.get("ids", {}) or {}
            doi = cls._normalize_doi(ids.get("doi") or work.get("doi"))
            pmid = cls._normalize_pmid(ids.get("pmid") or work.get("pmid"))
            pmcid = cls._normalize_pmcid(ids.get("pmcid") or work.get("pmcid"))
            primary_location = work.get("primary_location", {}) or {}
            source = primary_location.get("source", {}) or {}
            
            # Find PDF URL from locations
            pdf_url = None
            locations = work.get("locations", []) or []
            for loc in locations:
                if loc.get("pdf_url"):
                    pdf_url = loc.get("pdf_url")
                    break
                # Also check best_oa_location
                best_oa = work.get("best_oa_location") or {}
                if best_oa.get("pdf_url"):
                    pdf_url = best_oa.get("pdf_url")
                    break
            
            results.append(
                {
                    "id": work.get("id") or f"openalex:{doi or pmid or pmcid or work.get('display_name') or work.get('title', '')}",
                    "pmcid": pmcid or None,
                    "doi": doi or None,
                    "pmid": pmid or None,
                    "title": work.get("display_name") or work.get("title") or "No title available",
                    "authors": cls._join_authors(work.get("authorships", [])),
                    "journal": source.get("display_name") or "Unknown journal",
                    "year": cls._coerce_year(work.get("publication_year")),
                    "citationCount": work.get("cited_by_count", 0),
                    "isOpenAccess": bool((work.get("open_access") or {}).get("is_oa")),
                    "hasTextMinedTerms": False,
                    "hasFullText": work.get("has_fulltext", False),
                    "hasPdfUrl": bool(work.get("has_pdf_url")) or bool(pdf_url),
                    "pdfUrl": pdf_url,
                    "abstract": cls._reconstruct_abstract(work.get("abstract_inverted_index")),
                    "source": "OpenAlex",
                }
            )

        meta = data.get("meta", {}) or {}
        total = int(meta.get("count", 0) or 0)
        has_more = page * max_results < total
        return {
            "results": results,
            "pagination": {
                "total": total,
                "page": page,
                "pageSize": max_results,
                "hasMore": has_more,
            },
        }

    @classmethod
    def _merge_results(
        cls,
        europe_results: List[Dict[str, Any]],
        openalex_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Simple merge: Europe PMC first, then OpenAlex (no dedupe, just combine).
        Keep it simple - user selects which source to search in UI.
        """
        merged: List[Dict[str, Any]] = []
        
        # Add Europe PMC results
        for item in europe_results:
            merged.append({**item, "source": "Europe PMC"})
        
        # Add OpenAlex results (no dedupe check - simple append)
        for item in openalex_results:
            merged.append({**item, "source": "OpenAlex"})
        
        return merged

    @classmethod
    def _sort_results(cls, results: List[Dict[str, Any]], sort: str) -> List[Dict[str, Any]]:
        if sort == "cited":
            return sorted(results, key=lambda item: item.get("citationCount") or 0, reverse=True)
        if sort == "date":
            return sorted(results, key=lambda item: int(re.sub(r"\D", "", str(item.get("year") or "0")) or 0), reverse=True)
        return results

    @classmethod
    async def search_literature(
        cls,
        query: str,
        filters: Dict[str, Any],
        page_size: int = 25,
        page: int = 1,
        sort: str = "",
        source: str = "all",  # "all", "europepmc", or "openalex"
    ) -> Dict[str, Any]:
        """
        Unified search across Europe PMC and OpenAlex.
        
        Source modes (for keyword searches only):
        - "all": Call BOTH APIs in parallel, merge/dedupe/rank results
        - "europepmc": Only Europe PMC
        - "openalex": Only OpenAlex
        
        Identifier searches (DOI/PMCID/PMID) always ignore the source filter and
        search ALL sources (Europe PMC → OpenAlex → Semantic Scholar/PMC) to find
        the paper wherever it exists. Returns the first available result.
        """
        
        if not query.strip():
            return {
                "results": [],
                "pagination": {"total": 0, "page": 1, "pageSize": page_size, "hasMore": False},
            }

        fetch_size = min(100, max(page_size, page * page_size * 2))
        
        # Check if query is an identifier (DOI/PMCID/PMID)
        from backend.services.europe_pmc import EuropePMCService
        id_type, id_value = EuropePMCService.parse_identifier(query)
        
        # Only treat as identifier if it has a valid type (doi/pmcid/pmid)
        is_identifier = id_type in ("doi", "pmcid", "pmid")
        
        if is_identifier:
            # Identifier search: always use full multi-source fallback chain
            # (ignore source filter — we want the paper wherever it exists)
            return await cls.search_by_identifier(id_type, id_value, page_size, page)
        
        # Check cache first
        cache_key = f"{source}:{query}:{page}:{page_size}"
        if cache_key in _SEARCH_CACHE:
            cached = _SEARCH_CACHE[cache_key]
            if cached.get("timestamp", 0) + _CACHE_TTL > time.time():
                return cached.get("data")
        
        # Keyword search based on source mode
        results: List[Dict[str, Any]] = []
        total = 0
        has_more = False
        
        if source == "openalex":
            # OpenAlex only
            openalex_data = await cls.search_openalex(
                query=query,
                filters=filters,
                max_results=fetch_size,
                page=1,
                sort=sort,
            )
            results = openalex_data.get("results", []) if openalex_data else []
            total = openalex_data.get("pagination", {}).get("total", 0) if openalex_data else 0
            has_more = openalex_data.get("pagination", {}).get("hasMore", False) if openalex_data else False
        else:
            # Europe PMC only (default)
            europe_data = await EuropePMCService.search_literature(
                query=query,
                filters=filters,
                max_results=fetch_size,
                sort=sort,
                cursor_mark="*",
            )
            results = europe_data.get("results", []) if europe_data else []
            total = europe_data.get("pagination", {}).get("total", 0) if europe_data else 0
            has_more = europe_data.get("pagination", {}).get("hasMore", False) if europe_data else False
        
        # Paginate
        start = max(0, (page - 1) * page_size)
        end = start + page_size
        
        # Apply full-text filter if requested
        if filters.get("has_full_text"):
            results = cls._filter_full_text(results)
        
        page_results = results[start:end]

        # Spawn background enrichment task (non-blocking)
        # Only for "all" mode - enrich top results with full-text info
        if source == "all" and page == 1:
            asyncio.create_task(cls._enrich_results_async(page_results[:15]))

        return {
            "results": page_results,
            "pagination": {
                "total": total,
                "page": page,
                "pageSize": page_size,
                "hasMore": has_more and len(results) > end,
            },
        }

    @classmethod
    def _filter_full_text(cls, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter results that have full-text available."""
        return [
            r for r in results
            if r.get("hasFullText") or r.get("isOpenAccess") or r.get("pmcid")
        ]

    @classmethod
    async def _enrich_results_async(cls, results: List[Dict[str, Any]]):
        """
        Background task to enrich OpenAlex results with full-text availability.
        Non-blocking - does not affect response time.
        """
        if not results:
            return
        
        from backend.services.europe_pmc import EuropePMCService
        
        for result in results:
            doi = cls._normalize_doi(result.get("doi"))
            if not doi:
                continue
            
            try:
                # Quick check if Europe PMC has full-text
                paper_data = await EuropePMCService.fetch_structured_data(doi)
                if paper_data and paper_data.get("sections"):
                    result["hasFullText"] = True
                    result["fullTextUrl"] = f"https://europepmc.org/article/med/{doi}"
            except Exception:
                pass  # Non-blocking - ignore errors

    @classmethod
    def _deduplicate_results(cls, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Deduplicate by DOI - keep LAST occurrence.
        
        Since Europe PMC is added AFTER OpenAlex in merge step,
        Europe PMC version wins for same DOI.
        """
        seen: Dict[str, Dict[str, Any]] = {}
        
        for item in results:
            doi = cls._normalize_doi(item.get("doi"))
            if doi:
                key = f"doi:{doi}"
                # Simply keep last occurrence
                seen[key] = item
            else:
                # Fall back to title + year
                title = (item.get("title") or "").strip().lower()
                year = str(item.get("year") or "")
                if title:
                    key = f"title:{title}|year:{year}"
                    seen[key] = item
        
        return list(seen.values())

    @classmethod
    async def search_by_identifier(
        cls,
        id_type: str,
        id_value: str,
        page_size: int = 25,
        page: int = 1,
    ) -> Dict[str, Any]:
        """
        Search by identifier (DOI/PMCID/PMID) — always searches ALL sources.
        
        Unlike keyword search, identifier searches ignore the source filter.
        They attempt to find the paper across all available sources in order:
        
        For DOI:
          1. Europe PMC (if abstract available → return)
          2. OpenAlex (if abstract available → return)
          3. Semantic Scholar (if abstract available → return)
          4. Otherwise return first metadata result found (Europe PMC > OpenAlex > Semantic Scholar)
        
        For PMCID:
          1. Europe PMC (return if found)
          2. PMC direct (return if found)
        
        For PMID:
          1. Europe PMC only (return if found)
        
        Returns first result with content; if none have abstracts, returns best metadata.
        """
        from backend.services.europe_pmc import EuropePMCService
        from backend.services.doi_resolver import (
            fetch_from_semantic_scholar,
            fetch_pmc_by_pmcid,
        )
        
        # Helper: compute content quality score (2=full-text, 1=abstract, 0=metadata only)
        def _content_score(item: Dict[str, Any]) -> int:
            # Full-text indicators
            if item.get("hasFullText") or item.get("pdfUrl") or item.get("openAccessPdf") or item.get("full_text_xml"):
                return 2
            abstract = (item.get("abstract") or "").strip()
            if abstract:
                return 1
            return 0
        
        best_candidate = None
        best_score = -1
        
        # Try Europe PMC first
        europe_data = await EuropePMCService.search_literature(
            query=id_value,
            filters={},
            max_results=10,
            sort="",
            cursor_mark="*",
        )
        if europe_data and europe_data.get("results"):
            first = europe_data["results"][0]
            first["source"] = "Europe PMC"
            score = _content_score(first)
            if score > best_score:
                best_candidate = first
                best_score = score
            if score == 2:  # Full text — return immediately
                return {
                    "results": [first],
                    "pagination": {"total": 1, "page": 1, "pageSize": 1, "hasMore": False},
                }
            # For non-DOI identifiers (pmid/pmcid), Europe PMC is the only source — return what we have
            if id_type != "doi":
                return {
                    "results": [first],
                    "pagination": {"total": 1, "page": 1, "pageSize": 1, "hasMore": False},
                }
        
        # If not found in Europe PMC (or only metadata) and it's a DOI, try OpenAlex
        if id_type == "doi":
            openalex_data = await cls.search_openalex(
                query=id_value,
                filters={},
                max_results=10,
                page=1,
                sort="",
            )
            if openalex_data and openalex_data.get("results"):
                first = openalex_data["results"][0]
                first["source"] = "OpenAlex"
                score = _content_score(first)
                if score > best_score:
                    best_candidate = first
                    best_score = score
                if score == 2:  # Shouldn't happen (OpenAlex has no full text), but safe
                    return {
                        "results": [first],
                        "pagination": {"total": 1, "page": 1, "pageSize": 1, "hasMore": False},
                    }
                # Continue to Semantic Scholar even if we have abstract candidate
            
            # Try Semantic Scholar fallback (direct DOI fetch)
            ss_data = await fetch_from_semantic_scholar(id_value)
            if ss_data:
                # Build result dict matching expected fields
                ss_result = {
                    "doi": ss_data.get("doi", id_value),
                    "title": ss_data.get("title", ""),
                    "authors": ss_data.get("authors", []),
                    "year": ss_data.get("year"),
                    "journal": ss_data.get("journal", ""),
                    "pmcid": ss_data.get("pmcid", ""),
                    "pmid": ss_data.get("pmid", ""),
                    "abstract": ss_data.get("abstract", ""),
                    "source": ss_data.get("source", "Semantic Scholar"),
                    "openAccessPdf": ss_data.get("openAccessPdf"),
                }
                score = _content_score(ss_result)
                if score > best_score:
                    best_candidate = ss_result
                    best_score = score
                if score == 2:
                    return {
                        "results": [ss_result],
                        "pagination": {"total": 1, "page": 1, "pageSize": 1, "hasMore": False},
                    }
        
        # PMCID fallback (PMC direct)
        if id_type == "pmcid":
            pmc_data = await fetch_pmc_by_pmcid(id_value)
            if pmc_data:
                pmc_result = {
                    "pmcid": pmc_data.get("pmcid", id_value),
                    "title": pmc_data.get("title", ""),
                    "authors": pmc_data.get("authors", []),
                    "year": pmc_data.get("year"),
                    "journal": pmc_data.get("journal", ""),
                    "abstract": pmc_data.get("abstract", ""),
                    "source": "PMC",
                    "full_text_xml": pmc_data.get("full_text_xml"),
                }
                score = _content_score(pmc_result)
                if score > best_score:
                    best_candidate = pmc_result
                    best_score = score
                if score == 2:
                    return {
                        "results": [pmc_result],
                        "pagination": {"total": 1, "page": 1, "pageSize": 1, "hasMore": False},
                    }
        
        # Return best candidate found (may be metadata-only)
        if best_candidate:
            return {
                "results": [best_candidate],
                "pagination": {"total": 1, "page": 1, "pageSize": 1, "hasMore": False},
            }
        
        # Nothing found anywhere
        return {
            "results": [],
            "pagination": {"total": 0, "page": 1, "pageSize": page_size, "hasMore": False},
        }
