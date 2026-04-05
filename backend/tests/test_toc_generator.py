import pytest
from backend.services.europe_pmc import EuropePMCService


@pytest.fixture
def sample_xml_with_h3():
    """Minimal JATS XML with H3 headings."""
    return """<?xml version="1.0"?>
    <article>
        <body>
            <sec>
                <title>Introduction</title>
                <p>Intro text.</p>
                <p>[H3]Background[/H3]</p>
                <p>Background details.</p>
                <p>[H3]Objectives[/H3]</p>
                <p>Study objectives.</p>
            </sec>
        </body>
    </article>"""


@pytest.fixture
def sample_xml_no_h3():
    """JATS XML without H3 headings."""
    return """<?xml version="1.0"?>
    <article>
        <body>
            <sec>
                <title>Methods</title>
                <p>Method text without headings.</p>
            </sec>
        </body>
    </article>"""


@pytest.fixture
def sample_xml_multiple_sections():
    """JATS XML with multiple sections."""
    return """<?xml version="1.0"?>
    <article>
        <body>
            <sec>
                <title>Results</title>
                <p>[H3]Primary Findings[/H3]</p>
                <p>Main results.</p>
                <p>[H3]Secondary Findings[/H3]</p>
                <p>Additional results.</p>
            </sec>
            <sec>
                <title>Discussion</title>
                <p>[H3]Implications[/H3]</p>
                <p>Discussion text.</p>
            </sec>
        </body>
    </article>"""


def test_extracts_h3_headings_from_section(sample_xml_with_h3, monkeypatch):
    """H3 markers should be extracted as heading objects."""
    # Ensure sanitization is a no-op for testing the TOC logic
    import backend.services.europe_pmc as eps

    monkeypatch.setattr(eps, "sanitize", lambda x: x)
    sections, references = EuropePMCService.parse_sections_from_xml(
        sample_xml_with_h3, "10.1234/test"
    )
    assert len(sections) == 1
    headings = sections[0]["headings"]
    assert len(headings) == 2
    assert headings[0]["text"] == "Background"
    assert headings[1]["text"] == "Objectives"


def test_headings_have_unique_ids(sample_xml_with_h3, monkeypatch):
    """Each heading should have a slugified ID."""
    import backend.services.europe_pmc as eps

    monkeypatch.setattr(eps, "sanitize", lambda x: x)
    sections, references = EuropePMCService.parse_sections_from_xml(
        sample_xml_with_h3, "10.1234/test"
    )
    headings = sections[0]["headings"]
    assert headings[0]["id"] == "background"
    assert headings[1]["id"] == "objectives"
    assert headings[0]["id"] != headings[1]["id"]


def test_slug_id_special_characters():
    """Special characters should be converted to hyphens."""
    xml = """<?xml version="1.0"?>
    <article><body><sec><title>Test</title>
    <p>[H3]Chemical: Compound-A (2019) &amp; More![/H3]</p>
    </sec></body></article>"""
    sections, references = EuropePMCService.parse_sections_from_xml(xml, "10.1234/test")
    headings = sections[0]["headings"]
    assert headings[0]["id"] == "chemical-compound-a-2019-more-"


def test_id_truncated_to_50_chars():
    """Heading IDs should be truncated to 50 characters."""
    long_title = "A" * 100
    xml = f"""<?xml version="1.0"?>
    <article><body><sec><title>T</title>
    <p>[H3]{long_title}[/H3]</p>
    </sec></body></article>"""
    sections, references = EuropePMCService.parse_sections_from_xml(xml, "10.1234/test")
    headings = sections[0]["headings"]
    assert len(headings[0]["id"]) <= 50


def test_no_headings_returns_empty_list(sample_xml_no_h3, monkeypatch):
    """Sections without H3 markers should have empty headings list."""
    import backend.services.europe_pmc as eps

    monkeypatch.setattr(eps, "sanitize", lambda x: x)
    sections, references = EuropePMCService.parse_sections_from_xml(
        sample_xml_no_h3, "10.1234/test"
    )
    assert len(sections) == 1
    assert sections[0]["headings"] == []


def test_multiple_sections_with_headings(sample_xml_multiple_sections, monkeypatch):
    """Each section should have its own headings list."""
    import backend.services.europe_pmc as eps

    monkeypatch.setattr(eps, "sanitize", lambda x: x)
    sections, references = EuropePMCService.parse_sections_from_xml(
        sample_xml_multiple_sections, "10.1234/test"
    )
    assert len(sections) == 2
    assert sections[0]["title"] == "Results"
    assert len(sections[0]["headings"]) == 2
    assert sections[0]["headings"][0]["text"] == "Primary Findings"
    assert sections[1]["title"] == "Discussion"
    assert len(sections[1]["headings"]) == 1
    assert sections[1]["headings"][0]["text"] == "Implications"


def test_h3_marker_preserved_for_template(sample_xml_with_h3, monkeypatch):
    """[H3] markers are preserved for template to generate <h3 id="...">."""
    import backend.services.europe_pmc as eps

    monkeypatch.setattr(eps, "sanitize", lambda x: x)
    sections, references = EuropePMCService.parse_sections_from_xml(
        sample_xml_with_h3, "10.1234/test"
    )
    content = sections[0]["content"]
    # Template generates <h3 id="..."> from [H3] marker — no span injection needed
    assert "[H3]Background[/H3]" in content
    assert "[H3]Objectives[/H3]" in content


def test_section_content_preserved(sample_xml_with_h3, monkeypatch):
    """The actual content should be preserved (not removed)."""
    import backend.services.europe_pmc as eps

    monkeypatch.setattr(eps, "sanitize", lambda x: x)
    sections, references = EuropePMCService.parse_sections_from_xml(
        sample_xml_with_h3, "10.1234/test"
    )
    content = sections[0]["content"]
    assert "Intro text" in content
    assert "Background details" in content
    assert "Study objectives" in content


def test_empty_xml_returns_empty_sections():
    """Empty or minimal XML should not crash."""
    xml = '<?xml version="1.0"?><article><body></body></article>'
    sections, references = EuropePMCService.parse_sections_from_xml(xml, "10.1234/test")
    assert sections == []


def test_title_extraction():
    """Section titles should be extracted correctly."""
    xml = """<?xml version="1.0"?>
    <article><body>
    <sec><title>My Custom Title</title><p>Content</p></sec>
    </body></article>"""
    sections, references = EuropePMCService.parse_sections_from_xml(xml, "10.1234/test")
    assert sections[0]["title"] == "My Custom Title"
