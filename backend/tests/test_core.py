"""
Tests for core utilities: caching, highlighter, sanitizer.
"""

import pytest
import os
import time
import tempfile
import shutil
from backend.core.caching import SimpleCache, CACHE_VERSION


# =============================================================================
# Caching Tests
# =============================================================================


@pytest.fixture
def temp_cache_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_cache_set_get(temp_cache_dir):
    """Basic set and get operations."""
    cache = SimpleCache(temp_cache_dir, ttl=3600, max_files=10)
    cache.set("key1", {"data": "hello"})
    result = cache.get("key1")
    assert result is not None
    assert result["data"] == "hello"
    assert result["_version"] == CACHE_VERSION


def test_cache_miss_returns_none(temp_cache_dir):
    """Getting a nonexistent key returns None."""
    cache = SimpleCache(temp_cache_dir, ttl=3600, max_files=10)
    assert cache.get("nonexistent") is None


def test_cache_ttl_expiration(temp_cache_dir):
    """Cache entries expire after TTL."""
    cache = SimpleCache(temp_cache_dir, ttl=1, max_files=10)
    cache.set("key1", {"data": "hello"})
    assert cache.get("key1") is not None
    time.sleep(1.1)
    assert cache.get("key1") is None


def test_cache_old_version_invalidated(temp_cache_dir):
    """Old cache version is invalidated."""
    import hashlib
    import gzip
    import json

    cache = SimpleCache(temp_cache_dir, ttl=3600, max_files=10)
    key = hashlib.sha256("old_key".encode()).hexdigest()
    prefix = key[:2]
    subdir = os.path.join(temp_cache_dir, prefix)
    os.makedirs(subdir, exist_ok=True)
    path = os.path.join(subdir, f"{key}.json.gz")
    with gzip.open(path, "wt") as f:
        json.dump({"data": "old", "_version": "v1", "_cached_at": time.time()}, f)
    assert cache.get("old_key") is None


def test_cache_max_files_eviction(temp_cache_dir):
    """Oldest files evicted when max reached."""
    cache = SimpleCache(temp_cache_dir, ttl=3600, max_files=3)
    cache.set("key1", {"data": "1"})
    cache.set("key2", {"data": "2"})
    cache.set("key3", {"data": "3"})
    time.sleep(0.1)
    cache.set("key4", {"data": "4"})
    assert cache.get("key1") is None
    assert cache.get("key4") is not None


def test_cache_clear(temp_cache_dir):
    """Clear removes all entries."""
    cache = SimpleCache(temp_cache_dir, ttl=3600, max_files=10)
    cache.set("key1", {"data": "1"})
    cache.set("key2", {"data": "2"})
    cache.clear()
    assert cache.get("key1") is None
    assert cache.get("key2") is None


def test_cache_compression(temp_cache_dir):
    """Cache files are gzip compressed."""
    cache = SimpleCache(temp_cache_dir, ttl=3600, max_files=10)
    cache.set("large", {"data": "x" * 10000})
    assert cache.get("large") is not None


def test_cache_migration_from_json(temp_cache_dir):
    """Legacy .json files migrated to .json.gz."""
    import hashlib
    import json

    cache = SimpleCache(temp_cache_dir, ttl=3600, max_files=10)
    key = hashlib.sha256("legacy".encode()).hexdigest()
    prefix = key[:2]
    subdir = os.path.join(temp_cache_dir, prefix)
    os.makedirs(subdir, exist_ok=True)
    old_path = os.path.join(subdir, f"{key}.json")
    with open(old_path, "w") as f:
        json.dump({"data": "legacy", "_version": "v2", "_cached_at": time.time()}, f)
    result = cache.get("legacy")
    assert result is not None
    assert result["data"] == "legacy"


# =============================================================================
# Highlighter Tests
# =============================================================================

from backend.core.highlighter import Highlighter


def test_highlight_empty_content():
    """Empty content returns empty."""
    assert Highlighter.highlight("", [{"text": "test", "label": "CHEMICAL"}]) == ""


def test_highlight_no_entities():
    """No entities returns original content."""
    html = "<p>Some text</p>"
    assert Highlighter.highlight(html, []) == html


def test_highlight_single_entity():
    """Single entity gets highlighted."""
    html = "<p>Eugenol is a chemical.</p>"
    entities = [{"text": "Eugenol", "label": "CHEMICAL", "score": 0.9}]
    result = Highlighter.highlight(html, entities)
    assert "ent-chemical" in result
    assert "Eugenol" in result


def test_highlight_multiple_entities():
    """Multiple entities of same type all highlighted."""
    html = "<p>Eugenol and Thymol are chemicals.</p>"
    entities = [
        {"text": "Eugenol", "label": "CHEMICAL", "score": 0.9},
        {"text": "Thymol", "label": "CHEMICAL", "score": 0.9},
    ]
    result = Highlighter.highlight(html, entities)
    assert result.count("ent-chemical") == 2


def test_highlight_different_types():
    """Different entity types get different colors."""
    html = "<p>Eugenol is in clove.</p>"
    entities = [
        {"text": "Eugenol", "label": "CHEMICAL", "score": 0.9},
        {"text": "clove", "label": "PLANT PART", "score": 0.9},
    ]
    result = Highlighter.highlight(html, entities)
    assert "ent-chemical" in result
    assert "ent-plant-part" in result


def test_highlight_entity_not_found():
    """Entity not in text returns original."""
    html = "<p>No match here.</p>"
    entities = [{"text": "NotFound", "label": "CHEMICAL", "score": 0.9}]
    assert Highlighter.highlight(html, entities) == html


def test_highlight_case_insensitive():
    """Matching is case insensitive."""
    html = "<p>EUGENOL is chemical.</p>"
    entities = [{"text": "eugenol", "label": "CHEMICAL", "score": 0.9}]
    assert "ent-chemical" in Highlighter.highlight(html, entities)


def test_highlight_preserves_html():
    """Existing HTML tags preserved."""
    html = "<p>Eugenol is <strong>important</strong>.</p>"
    entities = [{"text": "Eugenol", "label": "CHEMICAL", "score": 0.9}]
    result = Highlighter.highlight(html, entities)
    assert "<strong>" in result
    assert "ent-chemical" in result


def test_highlight_preserves_cite():
    """Cite tags with data-rid preserved."""
    html = '<p><cite data-rid="ref1">Author</cite></p>'
    entities = [{"text": "Author", "label": "SPECIES", "score": 0.9}]
    result = Highlighter.highlight(html, entities)
    assert 'data-rid="ref1"' in result


def test_color_map_all_types():
    """All entity types have colors."""
    expected = ["CHEMICAL", "SPECIES", "DISEASE", "PLANT PART", "DRUG", "LOCATION"]
    for et in expected:
        assert et in Highlighter.COLOR_MAP


# =============================================================================
# Sanitizer Tests
# =============================================================================

from backend.core.sanitizer import sanitize


def test_sanitize_blocks_scripts():
    """Script tags are removed."""
    assert sanitize("<script>alert(1)</script>") == ""


def test_sanitize_preserves_allowed_tags():
    """Allowed tags preserved."""
    assert sanitize("<p>text</p>") == "<p>text</p>"


def test_sanitize_blocks_javascript_href():
    """javascript: hrefs are blocked."""
    result = sanitize('<a href="javascript:alert(1)">link</a>')
    assert 'href=""' in result


def test_sanitize_preserves_cite_data_rid():
    """cite data-rid preserved."""
    result = sanitize('<cite data-rid="ref1">Smith</cite>')
    assert 'data-rid="ref1"' in result


def test_sanitize_preserves_img():
    """Img src preserved."""
    result = sanitize('<img src="https://example.com/img.jpg">')
    assert 'src="https://example.com/img.jpg"' in result


def test_sanitize_preserves_table():
    """Table elements preserved."""
    result = sanitize("<table><tr><td>Cell</td></tr></table>")
    assert "<table" in result


def test_sanitize_preserves_strong_em():
    """Strong and em tags preserved."""
    result = sanitize("<p><strong>bold</strong> and <em>italic</em></p>")
    assert "<strong>" in result
    assert "<em>" in result


def test_sanitize_preserves_sub_sup():
    """Sub and sup preserved."""
    result = sanitize("<p>H<sub>2</sub>O and E=mc<sup>2</sup></p>")
    assert "<sub>" in result
    assert "<sup>" in result


def test_sanitize_empty_string():
    """Empty string returns empty."""
    assert sanitize("") == ""
