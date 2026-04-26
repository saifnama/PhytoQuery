import pytest

from backend.services.search_service import SearchService


@pytest.mark.asyncio
async def test_search_service_merges_and_dedupes_by_doi(monkeypatch):
    async def fake_europe_search(query, filters, max_results, sort, cursor_mark):
        return {
            "results": [
                {
                    "id": "epmc-1",
                    "doi": "10.1000/test",
                    "pmid": "12345",
                    "pmcid": None,
                    "title": "Shared Paper",
                    "authors": "Europe Author",
                    "journal": "Europe Journal",
                    "year": "2024",
                    "citationCount": 5,
                    "isOpenAccess": True,
                    "hasTextMinedTerms": False,
                    "abstract": "Europe abstract",
                    "source": "Europe PMC",
                },
                {
                    "id": "epmc-2",
                    "doi": "10.1000/epmc-only",
                    "pmid": None,
                    "pmcid": None,
                    "title": "Europe Only",
                    "authors": "E Author",
                    "journal": "Europe Journal",
                    "year": "2023",
                    "citationCount": 3,
                    "isOpenAccess": False,
                    "hasTextMinedTerms": False,
                    "abstract": "",
                    "source": "Europe PMC",
                },
            ],
            "pagination": {"total": 2, "cursorMark": "*", "nextCursorMark": "x", "hasMore": False, "pageSize": max_results},
        }

    async def fake_openalex_search(query, filters, max_results, page, sort):
        return {
            "results": [
                {
                    "id": "oa-1",
                    "doi": "https://doi.org/10.1000/test",
                    "pmid": None,
                    "pmcid": None,
                    "title": "Shared Paper",
                    "authors": "OpenAlex Author",
                    "journal": "OpenAlex Journal",
                    "year": "2024",
                    "citationCount": 99,
                    "isOpenAccess": True,
                    "hasTextMinedTerms": False,
                    "abstract": "OpenAlex abstract",
                    "source": "OpenAlex",
                },
                {
                    "id": "oa-2",
                    "doi": "10.1000/oa-only",
                    "pmid": None,
                    "pmcid": None,
                    "title": "OpenAlex Only",
                    "authors": "OA Author",
                    "journal": "OA Journal",
                    "year": "2022",
                    "citationCount": 10,
                    "isOpenAccess": True,
                    "hasTextMinedTerms": False,
                    "abstract": "",
                    "source": "OpenAlex",
                },
            ],
            "pagination": {"total": 2, "page": 1, "pageSize": max_results, "hasMore": False},
        }

    monkeypatch.setattr("backend.services.search_service.EuropePMCService.search_literature", fake_europe_search)
    monkeypatch.setattr(SearchService, "search_openalex", fake_openalex_search)

    result = await SearchService.search_literature("test", {}, page_size=10, page=1)

    assert result["pagination"]["total"] == 3
    assert len(result["results"]) == 3
    shared = next(item for item in result["results"] if item["title"] == "Shared Paper")
    assert shared["source"] == "Europe PMC"
    assert shared["citationCount"] == 5
    assert shared["abstract"] == "Europe abstract"


@pytest.mark.asyncio
async def test_search_service_sorts_by_citations(monkeypatch):
    async def fake_europe_search(query, filters, max_results, sort, cursor_mark):
        return {
            "results": [
                {
                    "id": "epmc-1",
                    "doi": "10.1000/a",
                    "title": "Europe Result",
                    "authors": "A",
                    "journal": "J",
                    "year": "2020",
                    "citationCount": 2,
                    "isOpenAccess": False,
                    "abstract": "",
                    "source": "Europe PMC",
                }
            ],
            "pagination": {"total": 1, "cursorMark": "*", "nextCursorMark": "", "hasMore": False, "pageSize": max_results},
        }

    async def fake_openalex_search(query, filters, max_results, page, sort):
        return {
            "results": [
                {
                    "id": "oa-1",
                    "doi": "10.1000/b",
                    "title": "OpenAlex Result",
                    "authors": "B",
                    "journal": "J",
                    "year": "2021",
                    "citationCount": 50,
                    "isOpenAccess": True,
                    "abstract": "",
                    "source": "OpenAlex",
                }
            ],
            "pagination": {"total": 1, "page": 1, "pageSize": max_results, "hasMore": False},
        }

    monkeypatch.setattr("backend.services.search_service.EuropePMCService.search_literature", fake_europe_search)
    monkeypatch.setattr(SearchService, "search_openalex", fake_openalex_search)

    result = await SearchService.search_literature("test", {}, page_size=10, page=1, sort="cited")

    assert result["results"][0]["title"] == "OpenAlex Result"
