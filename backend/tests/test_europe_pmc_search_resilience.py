import httpx
import pytest

from backend.services.europe_pmc.client import EuropePMCClient
from backend.core.http_client import HttpClientManager


class _MockClient:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    async def get(self, *args, **kwargs):
        self.calls += 1
        next_item = self._responses.pop(0)
        if isinstance(next_item, Exception):
            raise next_item
        return next_item


class _SleepRecorder:
    def __init__(self):
        self.delays = []

    async def __call__(self, delay):
        self.delays.append(delay)
        return None


@pytest.mark.asyncio
async def test_search_literature_retries_on_504_then_succeeds(monkeypatch):
    request = httpx.Request("GET", "https://www.ebi.ac.uk/europepmc/webservices/rest/search")
    transient_response = httpx.Response(504, request=request)
    success_response = httpx.Response(
        200,
        request=request,
        json={
            "hitCount": 1,
            "nextCursorMark": "AoIIPzM",
            "resultList": {
                "result": [
                    {
                        "id": "1",
                        "title": "Test paper",
                        "authorString": "Author A",
                        "journalTitle": "Journal",
                        "pubYear": "2024",
                    }
                ]
            },
        },
    )
    mock_client = _MockClient([transient_response, success_response])

    async def fake_get_client():
        return mock_client

    fake_sleep = _SleepRecorder()

    monkeypatch.setattr(HttpClientManager, "get_client", fake_get_client)
    monkeypatch.setattr("backend.services.europe_pmc.client.asyncio.sleep", fake_sleep)

    result = await EuropePMCClient.search_literature(
        query="phytochemical",
        filters={},
        max_results=25,
        cursor_mark="*",
    )

    assert mock_client.calls == 2
    assert result["results"][0]["title"] == "Test paper"
    assert result["pagination"]["nextCursorMark"] == "AoIIPzM"


@pytest.mark.asyncio
async def test_search_literature_returns_empty_after_retry_exhaustion(monkeypatch):
    timeout_error = httpx.ReadTimeout(
        "timed out",
        request=httpx.Request("GET", "https://www.ebi.ac.uk/europepmc/webservices/rest/search"),
    )
    mock_client = _MockClient([timeout_error, timeout_error, timeout_error])

    async def fake_get_client():
        return mock_client

    fake_sleep = _SleepRecorder()

    monkeypatch.setattr(HttpClientManager, "get_client", fake_get_client)
    monkeypatch.setattr("backend.services.europe_pmc.client.asyncio.sleep", fake_sleep)

    result = await EuropePMCClient.search_literature(
        query="phytochemical",
        filters={},
        max_results=25,
        cursor_mark="*",
    )

    assert mock_client.calls == 3
    assert result == {"results": [], "pagination": {}}


@pytest.mark.asyncio
async def test_search_literature_caps_retry_after_delay(monkeypatch):
    request = httpx.Request("GET", "https://www.ebi.ac.uk/europepmc/webservices/rest/search")
    transient_response = httpx.Response(504, request=request, headers={"retry-after": "999"})
    success_response = httpx.Response(
        200,
        request=request,
        json={
            "hitCount": 1,
            "nextCursorMark": "next",
            "resultList": {
                "result": [
                    {
                        "id": "1",
                        "title": "Recovered paper",
                        "authorString": "Author A",
                        "journalTitle": "Journal",
                        "pubYear": "2024",
                    }
                ]
            },
        },
    )
    mock_client = _MockClient([transient_response, success_response])
    fake_sleep = _SleepRecorder()

    async def fake_get_client():
        return mock_client

    monkeypatch.setattr(HttpClientManager, "get_client", fake_get_client)
    monkeypatch.setattr("backend.services.europe_pmc.client.asyncio.sleep", fake_sleep)

    result = await EuropePMCClient.search_literature(
        query="phytochemical",
        filters={},
        max_results=25,
        cursor_mark="*",
    )

    assert mock_client.calls == 2
    assert fake_sleep.delays == [10.0]
    assert result["results"][0]["title"] == "Recovered paper"
