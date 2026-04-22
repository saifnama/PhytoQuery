"""
Test fixtures and configuration for PhytoQuery backend tests.
"""

import pytest
import os
import shutil
import tempfile
import logging
from httpx import AsyncClient, ASGITransport
from typing import List, Dict, Any, Optional, Tuple
from backend.app import app
from backend.api.paper import get_ner_service, get_pmc_service
from backend.api.rag import get_rag_service
from backend.core.caching import doi_cache, ner_cache, pmc_cache

import asyncio


# =============================================================================
# Mock Service Classes
# =============================================================================


class MockNerService:
    """Mock NER service for testing."""

    def __init__(self):
        self.should_fail = False
        self.delay = 0
        self.return_value = [
            {"text": "Mock Molecule", "label": "CHEMICAL", "score": 0.99}
        ]
        self.logger = logging.getLogger("MockNerService")
        self.result_cache = {}

    async def process_text(self, text: str) -> List[Dict[str, Any]]:
        """Mock entity extraction."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            self.logger.error("NER Mock Failure Triggered")
            raise Exception("NER Mock Failure")
        return self.return_value

    def deduplicate(
        self, entities: List[Dict], text: str
    ) -> Tuple[List[Dict], List[Dict]]:
        """Mock deduplication."""
        return entities, []


class MockRagService:
    """Mock RAG service for testing."""

    def __init__(self):
        self.should_fail = False
        self.delay = 0
        self.indexed_files = []
        self.logger = logging.getLogger("MockRagService")
        self.vectorstore = type(
            "obj",
            (object,),
            {
                "_collection": type(
                    "obj", (object,), {"count": lambda: len(self.indexed_files)}
                )
            },
        )

    def process_and_index_pdfs(self, pdf_paths: List[str]) -> List[str]:
        if self.should_fail:
            self.logger.error("RAG Indexing Mock Failure")
            raise Exception("RAG Indexing Failure")
        basenames = [os.path.basename(p) for p in pdf_paths]
        self.indexed_files.extend(basenames)
        return basenames

    async def query(self, question: str) -> Dict[str, Any]:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            self.logger.error("RAG Query Mock Failure")
            raise Exception("RAG Query Failure")
        if not self.indexed_files:
            return {"answer": "No documents indexed.", "sources": []}
        return {
            "answer": f"Mock answer for: {question}",
            "sources": [
                {"source": f, "section": "Mock Section"} for f in self.indexed_files
            ],
        }


class MockEuropePMCService:
    """Mock Europe PMC service for testing."""

    def __init__(self):
        self.return_text = "Phytochemicals like Eugenol are found in cloves."
        self.return_mode = "full_text"
        self.return_sections = [
            {
                "title": "Introduction",
                "content": "Eugenol is a phenylpropanoid.",
                "headings": [],
            },
            {
                "title": "Methods",
                "content": "Extraction was performed.",
                "headings": [],
            },
        ]
        self.return_references = {}
        self.return_title = "Test Paper Title"
        self.should_fail = False
        self.delay = 0
        self.logger = logging.getLogger("MockEuropePMCService")

    async def fetch_paper_data(self, doi: str) -> Tuple[Optional[str], str]:
        """Mock paper data fetch (legacy)."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            self.logger.error(f"PMC Mock Failure for {doi}")
            return None, "error"
        if doi == "not_found":
            self.logger.warning(f"No results found in Europe PMC for DOI: {doi}")
            return None, "empty"
        return self.return_text, self.return_mode

    async def fetch_structured_data(self, doi: str) -> Dict[str, Any]:
        """Mock structured data fetch."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.should_fail:
            return {
                "sections": [],
                "references": {},
                "title": "",
                "mode": "error",
                "pmcid": "",
            }
        if doi == "not_found":
            return {
                "sections": [],
                "references": {},
                "title": "",
                "mode": "empty",
                "pmcid": "",
            }
        return {
            "sections": self.return_sections,
            "references": self.return_references,
            "title": self.return_title,
            "mode": self.return_mode,
            "pmcid": doi,
        }

    @staticmethod
    def parse_identifier(identifier: str) -> Tuple[str, str]:
        """Mock identifier parsing."""
        if identifier.startswith("PMC"):
            return "pmcid", identifier
        return "doi", identifier

    @staticmethod
    async def search_literature(
        query: str, filters: Dict, max_results: int = 10, sort: str = ""
    ) -> List[Dict]:
        """Mock literature search."""
        return [
            {
                "title": f"Result for {query}",
                "doi": "10.1038/mock",
                "authors": ["Author A"],
            },
        ]


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def clear_caches_before_test():
    """Ensure a clean state by clearing global caches."""
    ner_cache.clear()
    pmc_cache.clear()
    doi_cache.clear()
    yield


@pytest.fixture(scope="function")
def mock_ner():
    return MockNerService()


@pytest.fixture(scope="function")
def mock_rag():
    return MockRagService()


@pytest.fixture(scope="function")
def mock_pmc():
    return MockEuropePMCService()


@pytest.fixture(autouse=True)
def setup_overrides(mock_ner, mock_rag, mock_pmc):
    """Override dependencies for all tests."""
    app.dependency_overrides[get_ner_service] = lambda: mock_ner
    app.dependency_overrides[get_rag_service] = lambda: mock_rag
    app.dependency_overrides[get_pmc_service] = lambda: mock_pmc
    yield
    app.dependency_overrides.clear()


@pytest.fixture
async def client():
    """Async client for testing."""
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as ac:
        yield ac


@pytest.fixture(scope="session")
def test_data_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_chroma_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture(autouse=True)
def configure_logging():
    logging.basicConfig(level=logging.INFO)
