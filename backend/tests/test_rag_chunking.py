import pytest
import os
from backend.services.rag_engine import RAGService, config


@pytest.fixture
def service():
    return RAGService()


def test_chunking_extremely_small_text(service):
    """Test chunking with text smaller than min_chunk_size."""
    text = "Short."
    sections = service._detect_sections(text)
    chunks = service._chunk_by_sections(sections, [])

    # config.min_chunk_size is 100 by default. "Short." is too small.
    assert len(chunks) == 0


def test_chunking_boundary_size(service):
    """Test chunking with text exactly equal to chunk_size."""
    # Create text of exact length
    text = "A" * config.chunk_size
    sections = service._detect_sections(text)
    chunks = service._chunk_by_sections(sections, [])

    assert len(chunks) == 1
    assert len(chunks[0]["text"]) == config.chunk_size


def test_chunking_off_by_one_boundary(service):
    """Test chunking with text exactly chunk_size + 1."""
    text = "A" * (config.chunk_size + 1)
    sections = service._detect_sections(text)
    chunks = service._chunk_by_sections(sections, [])

    # Should result in 2 chunks if overlap is handled correctly
    # RecursiveCharacterTextSplitter will split it.
    assert len(chunks) >= 1
    total_len = sum(len(c["text"]) for c in chunks)
    assert total_len >= config.chunk_size + 1


def test_chunking_extremely_large_text(service):
    """Test chunking with very large text."""
    text = "Word " * 10000  # ~50,000 chars
    sections = service._detect_sections(text)
    chunks = service._chunk_by_sections(sections, [])

    assert len(chunks) > 50  # chunk_size=500, so ~100 chunks expected
    for c in chunks:
        assert (
            len(c["text"]) <= config.chunk_size + 100
        )  # Allow some leeway for separators


def test_section_detection_boundaries(service):
    """Test section detection with various headers."""
    text = "# Header 1\nContent 1\n## Header 2\nContent 2\n1. List Header\nContent 3"
    sections = service._detect_sections(text)

    assert len(sections) == 3
    assert sections[0]["title"] == "Header 1"
    assert sections[1]["title"] == "Header 2"
    assert "List Header" in sections[2]["title"]
