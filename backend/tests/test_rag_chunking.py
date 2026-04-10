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


def test_process_pdf_uses_docling_only(service, monkeypatch, tmp_path):
    """PDF processing should rely on Docling output only."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    monkeypatch.setattr(
        service,
        "_extract_with_docling",
        lambda path: (
            "# Methods\nDocling extracted content.",
            [{"content": "| A |\n| --- |\n| 1 |", "page": 1}],
        ),
    )

    docs = service._process_pdf(str(pdf_path))

    assert docs
    assert any(doc.metadata.get("content_type") == "table" for doc in docs)
    assert any("Docling extracted content." in doc.page_content for doc in docs)


def test_process_pdf_returns_empty_when_docling_extracts_no_text(
    service, monkeypatch, tmp_path
):
    """Without Docling text, the docling-only pipeline should return no chunks."""
    pdf_path = tmp_path / "empty.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    monkeypatch.setattr(service, "_extract_with_docling", lambda path: (None, []))

    docs = service._process_pdf(str(pdf_path))

    assert docs == []


def test_process_and_index_pdfs_accepts_parser_type(service, monkeypatch, tmp_path):
    """process_and_index_pdfs should accept parser_type parameter."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    # Track which extractor was called
    extractors_used = []

    def mock_docling(path):
        extractors_used.append("docling")
        return "# Test\nContent from docling.", []

    def mock_pymupdf(path):
        extractors_used.append("pymupdf")
        return "Content from pymupdf.", []

    monkeypatch.setattr(service, "_extract_with_docling", mock_docling)
    monkeypatch.setattr(service, "_extract_with_pymupdf", mock_pymupdf)
    monkeypatch.setattr(service.vectorstore, "add_documents", lambda docs: None)

    # Test with docling (default)
    service.process_and_index_pdfs([str(pdf_path)], parser_type="docling")
    assert "docling" in extractors_used

    # Reset and test with pymupdf
    extractors_used.clear()
    service.process_and_index_pdfs([str(pdf_path)], parser_type="pymupdf")
    assert "pymupdf" in extractors_used


def test_process_pdf_uses_pymupdf_when_selected(service, monkeypatch, tmp_path):
    """When parser_type is pymupdf, _extract_with_pymupdf should be used."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    called = {"pymupdf": False, "docling": False}

    def mock_pymupdf(path):
        called["pymupdf"] = True
        return "PyMuPDF text.", []

    def mock_docling(path):
        called["docling"] = True
        return "Docling text.", []

    monkeypatch.setattr(service, "_extract_with_pymupdf", mock_pymupdf)
    monkeypatch.setattr(service, "_extract_with_docling", mock_docling)

    # Set parser to pymupdf
    service._current_parser = "pymupdf"
    docs = service._process_pdf(str(pdf_path))

    assert called["pymupdf"] is True
    assert called["docling"] is False
    assert len(docs) > 0
    assert docs[0].metadata.get("parser_type") == "pymupdf"


def test_process_pdf_metadata_includes_parser_type(service, monkeypatch, tmp_path):
    """Document metadata should include parser_type."""
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 mock")

    monkeypatch.setattr(
        service, "_extract_with_docling", lambda path: ("Test content.", [])
    )
    monkeypatch.setattr(
        service, "_extract_with_pymupdf", lambda path: ("Test content.", [])
    )

    # Test docling metadata
    service._current_parser = "docling"
    docs = service._process_pdf(str(pdf_path))
    assert docs[0].metadata.get("parser_type") == "docling"

    # Test pymupdf metadata
    service._current_parser = "pymupdf"
    docs = service._process_pdf(str(pdf_path))
    assert docs[0].metadata.get("parser_type") == "pymupdf"
