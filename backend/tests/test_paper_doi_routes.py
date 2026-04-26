from fastapi.testclient import TestClient
import pytest

from backend.app import app
from backend.api import doi as doi_api
from backend.api import paper as paper_api
from backend.api.paper import get_ner_service


class StubPaperNerService:
    def __init__(self):
        self.result_cache = {}

    async def process_text(self, text: str):
        return {"count": 1}, [{"text": "Eugenol", "label": "CHEMICAL", "score": 0.99}]

    async def process_sections(self, sections):
        return {"count": len(sections)}, [{"text": "Eugenol", "label": "CHEMICAL", "score": 0.99}]


def test_doi_abstract_cache_normalizes_key_and_hides_metadata(monkeypatch):
    calls = {"count": 0}

    async def fake_fetch_doi_abstract(doi: str):
        calls["count"] += 1
        return {
            "doi": doi,
            "title": "Test Title",
            "abstract": "Test Abstract",
            "authors": ["Author A"],
            "year": 2024,
            "journal": "Test Journal",
            "source": "OpenAlex",
            "url": f"https://example.org/{doi}",
        }

    monkeypatch.setattr(doi_api, "fetch_doi_abstract", fake_fetch_doi_abstract)
    client = TestClient(app)

    first = client.get("/doi/abstract", params={"doi": "https://doi.org/10.1234/ABC"})
    second = client.get("/doi/abstract", params={"doi": "doi:10.1234/abc"})

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls["count"] == 1
    assert first.json() == second.json()
    assert "_version" not in second.json()
    assert "_cached_at" not in second.json()
    assert second.json()["doi"] == "10.1234/abc"


@pytest.mark.asyncio
async def test_paper_cache_summary_and_section_lookup_use_normalized_identifier(client, monkeypatch):
    stub_service = StubPaperNerService()
    app.dependency_overrides[get_ner_service] = lambda: stub_service

    monkeypatch.setattr(
        paper_api.EuropePMCService,
        "parse_identifier",
        staticmethod(lambda identifier: ("doi", identifier.replace("https://doi.org/", "").replace("doi:", "").strip())),
    )

    async def fake_fetch_structured_data(doi: str):
        return {
            "doi": doi,
            "mode": "full_text",
            "title": "Paper Title",
            "sections": [{"title": "Abstract", "content": "Eugenol appears here."}],
            "references": {},
            "pmcid": "",
            "journal": "Journal",
            "authors": ["Author A"],
            "date": "2024",
        }

    monkeypatch.setattr(
        paper_api.EuropePMCService,
        "fetch_structured_data",
        staticmethod(fake_fetch_structured_data),
    )
    monkeypatch.setattr(
        paper_api.Highlighter,
        "highlight",
        staticmethod(lambda content, entities: f"H::{len(entities)}::{content}"),
    )

    extracted = await client.post(
        "/paper/json",
        data={"doi": "https://doi.org/10.1234/abc", "run_ner": "true"},
    )
    assert extracted.status_code == 200
    extracted_json = extracted.json()
    assert extracted_json["is_extracted"] is True
    assert extracted_json["summary"] == {"count": 2}
    assert stub_service.result_cache["10.1234/abc"] == {
        "entities": [{"text": "Eugenol", "label": "CHEMICAL", "score": 0.99}],
        "summary": {"count": 2},
    }

    cached = await client.post(
        "/paper/json",
        data={"doi": "doi:10.1234/ABC", "run_ner": "false"},
    )
    assert cached.status_code == 200
    assert cached.json()["summary"] == {"count": 2}
    assert cached.json()["doi"] == "10.1234/abc"

    section = await client.post(
        "/paper/section/json",
        data={"doi": "https://doi.org/10.1234/ABC", "section_idx": "0"},
    )
    assert section.status_code == 200
    assert section.json()["highlighted"].startswith("H::1::")


@pytest.mark.asyncio
async def test_doi_fallback_abstract_runs_shared_ner_flow(client, monkeypatch):
    stub_service = StubPaperNerService()
    app.dependency_overrides[get_ner_service] = lambda: stub_service

    monkeypatch.setattr(
        paper_api.EuropePMCService,
        "parse_identifier",
        staticmethod(lambda identifier: ("doi", identifier.strip().lower())),
    )
    async def fake_fetch_structured_data(doi: str):
        return {
            "doi": doi,
            "mode": "abstract",
            "title": "",
            "sections": [],
            "references": {},
            "pmcid": "",
            "journal": "",
            "authors": [],
            "date": "",
        }

    monkeypatch.setattr(
        paper_api.EuropePMCService,
        "fetch_structured_data",
        staticmethod(fake_fetch_structured_data),
    )

    async def fake_fallback(id_type: str, clean_id: str):
        assert id_type == "doi"
        assert clean_id == "10.1234/abc"
        return {
            "doi": clean_id,
            "title": "Fallback Title",
            "abstract": "Eugenol appears in abstract.",
            "authors": ["Author A"],
            "year": 2024,
            "journal": "Fallback Journal",
            "source": "OpenAlex",
            "url": f"https://openalex.org/{clean_id}",
        }

    monkeypatch.setattr(paper_api, "_fetch_identifier_fallback", fake_fallback)
    monkeypatch.setattr(
        paper_api.Highlighter,
        "highlight",
        staticmethod(lambda content, entities: f"H::{len(entities)}::{content}"),
    )

    response = await client.post(
        "/paper/json",
        data={"doi": "10.1234/abc", "run_ner": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["is_extracted"] is True
    assert payload["summary"] == {"count": 2}
    assert payload["fallback_source"] == "OpenAlex"
    assert payload["fallback_url"] == "https://openalex.org/10.1234/abc"
    assert payload["entities"] == [{"text": "Eugenol", "label": "CHEMICAL", "score": 0.99}]
    assert payload["title"].startswith("H::1::")
    assert payload["sections"][0]["content"].startswith("H::1::")
    assert stub_service.result_cache["10.1234/abc"] == {
        "entities": [{"text": "Eugenol", "label": "CHEMICAL", "score": 0.99}],
        "summary": {"count": 2},
    }


@pytest.mark.asyncio
async def test_pmcid_xml_fallback_runs_shared_ner_flow(client, monkeypatch):
    stub_service = StubPaperNerService()
    app.dependency_overrides[get_ner_service] = lambda: stub_service

    monkeypatch.setattr(
        paper_api.EuropePMCService,
        "parse_identifier",
        staticmethod(lambda identifier: ("pmcid", identifier.strip().upper())),
    )
    async def fake_fetch_structured_data(pmcid: str):
        return {
            "doi": "",
            "mode": "full_text",
            "title": "",
            "sections": [],
            "references": {},
            "pmcid": pmcid,
            "journal": "",
            "authors": [],
            "date": "",
        }

    monkeypatch.setattr(
        paper_api.EuropePMCService,
        "fetch_structured_data",
        staticmethod(fake_fetch_structured_data),
    )
    monkeypatch.setattr(
        paper_api.EuropePMCService,
        "parse_sections_from_xml",
        staticmethod(
            lambda xml, pmcid=None: (
                [{"title": "Body", "content": "Eugenol appears in body."}],
                {"R1": "Reference 1"},
            )
        ),
    )

    async def fake_fallback(id_type: str, clean_id: str):
        assert id_type == "pmcid"
        assert clean_id == "PMC123456"
        return {
            "doi": "10.5555/test",
            "pmcid": clean_id,
            "title": "PMC Fallback Title",
            "full_text_xml": "<article></article>",
            "authors": ["Author P"],
            "year": 2023,
            "journal": "PMC Journal",
            "source": "PMC",
            "url": f"https://pmc.ncbi.nlm.nih.gov/articles/{clean_id}/",
        }

    monkeypatch.setattr(paper_api, "_fetch_identifier_fallback", fake_fallback)
    monkeypatch.setattr(
        paper_api.Highlighter,
        "highlight",
        staticmethod(lambda content, entities: f"H::{len(entities)}::{content}"),
    )

    response = await client.post(
        "/paper/json",
        data={"doi": "PMC123456", "run_ner": "true"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["is_extracted"] is True
    assert payload["summary"] == {"count": 2}
    assert payload["fallback_source"] == "PMC"
    assert payload["fallback_url"] == "https://pmc.ncbi.nlm.nih.gov/articles/PMC123456/"
    assert payload["references"] == {"R1": "Reference 1"}
    assert payload["title"].startswith("H::1::")
    assert payload["sections"][0]["content"].startswith("H::1::")
