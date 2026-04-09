import importlib
from typing import Any, Dict, List, Tuple, cast

import pytest
import spacy


def _build_matcher_with_fresh_cache(
    module_name: str, class_name: str, tmp_path, monkeypatch
):
    module = importlib.import_module(module_name)
    monkeypatch.setattr(module, "CACHE_FILE", tmp_path / f"{class_name}.pkl")
    if hasattr(module, "_matcher"):
        monkeypatch.setattr(module, "_matcher", None)
    matcher_class = getattr(module, class_name)
    return matcher_class(nlp=spacy.blank("en"))


@pytest.mark.parametrize(
    (
        "module_name",
        "class_name",
        "text",
        "expected_span",
        "expected_canonical",
        "expected_label",
    ),
    [
        (
            "backend.gazetteer.analytical_technique_matcher",
            "AnalyticalTechniqueMatcher",
            "The extract was profiled by gc-ms.",
            "gc-ms",
            "gas chromatography-mass spectrometry",
            "ANALYTICAL TECHNIQUE",
        ),
        (
            "backend.gazetteer.extraction_method_matcher",
            "ExtractionMethodMatcher",
            "The extract was prepared by hot continuous extraction.",
            "hot continuous extraction",
            "soxhlet extraction",
            "EXTRACTION METHOD",
        ),
        (
            "backend.gazetteer.development_stage_matcher",
            "DevelopmentStageMatcher",
            "Samples were collected at full bloom.",
            "full bloom",
            "full flowering",
            "DEVELOPMENT STAGE",
        ),
        (
            "backend.gazetteer.season_matcher",
            "SeasonMatcher",
            "Leaves were harvested in fall season.",
            "fall season",
            "autumn",
            "SEASON",
        ),
    ],
)
def test_new_dictionary_matchers_support_alias_matching(
    module_name,
    class_name,
    text,
    expected_span,
    expected_canonical,
    expected_label,
    tmp_path,
    monkeypatch,
):
    matcher = _build_matcher_with_fresh_cache(
        module_name, class_name, tmp_path, monkeypatch
    )

    matches = matcher.match(text)

    assert matches, f"Expected at least one match for {module_name}"
    assert any(match["span"].lower() == expected_span for match in matches)

    target = next(match for match in matches if match["span"].lower() == expected_span)
    assert target["canonical"] == expected_canonical
    assert target["label"] == expected_label
    assert expected_span in target["aliases"]


@pytest.mark.asyncio
async def test_ner_service_process_text_extracts_new_dictionary_entities(
    tmp_path, monkeypatch
):
    import backend.services.ner_engine as ner_engine
    import backend.gazetteer.extraction_method_matcher as extraction_method_matcher
    import backend.gazetteer.development_stage_matcher as development_stage_matcher
    import backend.gazetteer.season_matcher as season_matcher

    blank_nlp = spacy.blank("en")
    monkeypatch.setattr(ner_engine.spacy, "load", lambda *args, **kwargs: blank_nlp)

    for module, cache_name in [
        (extraction_method_matcher, "extraction.pkl"),
        (development_stage_matcher, "development.pkl"),
        (season_matcher, "season.pkl"),
    ]:
        monkeypatch.setattr(module, "CACHE_FILE", tmp_path / cache_name)
        monkeypatch.setattr(module, "_matcher", None)

    async def fake_call_llm(self, text_chunk: str) -> str:
        return ""

    def fake_deduplicate(self, entities, text):
        return {"entities": entities}, entities

    monkeypatch.setattr(ner_engine.NERService, "call_llm", fake_call_llm)
    monkeypatch.setattr(ner_engine.NERService, "deduplicate", fake_deduplicate)

    service = ner_engine.NERService()
    _, entities = cast(
        Tuple[Dict[str, Any], List[Dict[str, Any]]],
        await service.process_text(
            "Leaves were collected in fall season at full bloom after hot continuous extraction and profiled by gc-ms."
        ),
    )

    canonical_by_label = {
        (entity["label"], entity.get("canonical")) for entity in entities
    }

    assert ("PLANT PART", "leaf") in canonical_by_label
    assert (
        "ANALYTICAL TECHNIQUE",
        "gas chromatography-mass spectrometry",
    ) in canonical_by_label
    assert ("SEASON", "autumn") in canonical_by_label
    assert ("DEVELOPMENT STAGE", "full flowering") in canonical_by_label
    assert ("EXTRACTION METHOD", "soxhlet extraction") in canonical_by_label


def test_highlighter_supports_new_dictionary_entity_classes():
    from backend.core.highlighter import Highlighter

    html = "<p>The leaves were collected in fall season at full bloom.</p>"
    entities = [
        {"text": "leaves", "label": "PLANT PART", "score": 1.0},
        {"text": "fall season", "label": "SEASON", "score": 1.0},
        {"text": "full bloom", "label": "DEVELOPMENT STAGE", "score": 1.0},
    ]

    highlighted = Highlighter.highlight(html, entities)

    assert "ent-plant-part" in highlighted
    assert "ent-season" in highlighted
    assert "ent-development-stage" in highlighted
