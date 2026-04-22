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
    assert extracted_json["summary"] == {"count": 1}
    assert stub_service.result_cache["10.1234/abc"] == {
        "entities": [{"text": "Eugenol", "label": "CHEMICAL", "score": 0.99}],
        "summary": {"count": 1},
    }

    cached = await client.post(
        "/paper/json",
        data={"doi": "doi:10.1234/ABC", "run_ner": "false"},
    )
    assert cached.status_code == 200
    assert cached.json()["summary"] == {"count": 1}
    assert cached.json()["doi"] == "10.1234/abc"

    section = await client.post(
        "/paper/section/json",
        data={"doi": "https://doi.org/10.1234/ABC", "section_idx": "0"},
    )
    assert section.status_code == 200
    assert section.json()["highlighted"].startswith("H::1::")
