from fastapi.testclient import TestClient

from backend.app import app
from backend.api.ner import ner_service
from backend.core.caching import ner_cache


def test_core_ner_routes_still_exist():
    route_paths = {route.path for route in app.routes}

    assert "/ner/process" in route_paths
    assert "/ner/doi/json" in route_paths


def test_removed_rdkit_routes_absent():
    route_paths = {route.path for route in app.routes}

    assert "/ner/molecule/image" not in route_paths
    assert "/ner/molecule/info" not in route_paths


def test_clear_ner_cache_route_exists():
    route_paths = {route.path for route in app.routes}

    assert "/ner/cache/{doi}" in route_paths


def test_clear_ner_cache_removes_disk_and_memory_entries():
    client = TestClient(app)
    doi = "PMC123"
    cache_key = f"doi_json::{doi}"

    ner_cache.set(cache_key, {"doi": doi, "mode": "full_text", "text": "x", "entities": []})
    ner_service.result_cache[doi] = [{"text": "Mock", "label": "CHEMICAL", "score": 0.99}]

    response = client.delete(f"/ner/cache/{doi}")

    assert response.status_code == 200
    assert response.json() == {"status": "cleared", "doi": doi}
    assert ner_cache.get(cache_key) is None
    assert doi not in ner_service.result_cache
