import importlib
import pickle
import time
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


def _write_species_csv(tmp_path) -> None:
    (tmp_path / "species.csv").write_text(
        "scientific_name_input,scientific_name_verified,accepted_scientific_name,common_name,source_db,source_url,taxon_id,match_status,review_required\n"
        "Ocimum sanctum,Ocimum tenuiflorum,Ocimum tenuiflorum,Tulsi,GBIF,https://example.org/species/1,1,exact,no\n",
        encoding="utf-8",
    )


def _write_chemical_csv(tmp_path) -> None:
    (tmp_path / "chemical.csv").write_text(
        "term,synonyms,inchikey,smiles,molecular_formula,source_db,source_url\n"
        "Eugenol,4-Allyl-2-methoxyphenol|Clove oil phenol,RNGBKPVMWVROLL-UHFFFAOYSA-N,COC1=C(C=CC=C1)CC=C,C10H12O2,PubChem,https://pubchem.ncbi.nlm.nih.gov/compound/3314\n",
        encoding="utf-8",
    )


def _write_variant_term_chemical_csv(tmp_path) -> None:
    (tmp_path / "chemical.csv").write_text(
        "term,synonyms,inchikey,smiles,molecular_formula,source_db,source_url\n"
        "4-Allyl-2-methoxyphenol,Eugenol|Clove oil phenol,RNGBKPVMWVROLL-UHFFFAOYSA-N,COC1=C(C=CC=C1)CC=C,C10H12O2,PubChem,https://pubchem.ncbi.nlm.nih.gov/compound/3314\n",
        encoding="utf-8",
    )


def _write_collision_chemical_csv(tmp_path) -> None:
    (tmp_path / "chemical.csv").write_text(
        "term,synonyms,inchikey,smiles,molecular_formula,source_db,source_url\n"
        "Eugenol,Shared Alias,RNGBKPVMWVROLL-UHFFFAOYSA-N,COC1=C(C=CC=C1)CC=C,C10H12O2,PubChem,https://pubchem.ncbi.nlm.nih.gov/compound/3314\n"
        "Isoeugenol,Shared Alias,YLQBMQCUIZJEEH-UHFFFAOYSA-N,COC1=CC=CC(C=C)=C1O,C10H12O2,PubChem,https://pubchem.ncbi.nlm.nih.gov/compound/853433\n",
        encoding="utf-8",
    )


def _write_primary_priority_chemical_csv(tmp_path) -> None:
    (tmp_path / "chemical.csv").write_text(
        "term,synonyms,inchikey,smiles,molecular_formula,source_db,source_url\n"
        "Linalool,Linalol,CDOSHBSSFJOMGT-UHFFFAOYSA-N,CC(=CCCC(C)(C=C)O)C,C10H18O,PubChem,https://pubchem.ncbi.nlm.nih.gov/compound/6549\n"
        "Another compound,Linalool,AAAAAAAAAAAAAAAAAA-UHFFFAOYSA-N,CC,C2H6,PubChem,https://pubchem.ncbi.nlm.nih.gov/compound/1\n",
        encoding="utf-8",
    )


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


def test_species_matcher_supports_verified_species_metadata(tmp_path, monkeypatch):
    import backend.gazetteer.species_matcher as species_matcher

    _write_species_csv(tmp_path)
    monkeypatch.setattr(species_matcher, "DATA_FILE", tmp_path / "species.csv")
    monkeypatch.setattr(species_matcher, "CACHE_FILE", tmp_path / "species.pkl")
    monkeypatch.setattr(species_matcher, "_matcher", None)

    matcher = species_matcher.SpeciesMatcher(nlp=spacy.blank("en"))
    matches = matcher.match("Ocimum sanctum leaves were dried before extraction.")

    assert matches
    target = matches[0]
    assert target["text"] == "Ocimum sanctum"
    assert target["canonical"] == "Ocimum tenuiflorum"
    assert target["accepted_scientific_name"] == "Ocimum tenuiflorum"
    assert target["common_name"] == "Tulsi"
    assert target["taxon_id"] == "1"
    assert target["name_type"] == "scientific"


def test_chemical_matcher_supports_term_matching_and_metadata(tmp_path, monkeypatch):
    import backend.gazetteer.chemical_matcher as chemical_matcher

    _write_chemical_csv(tmp_path)
    monkeypatch.setattr(chemical_matcher, "DATA_FILE", tmp_path / "chemical.csv")
    monkeypatch.setattr(chemical_matcher, "CACHE_FILE", tmp_path / "chemical.pkl")
    monkeypatch.setattr(chemical_matcher, "_matcher", None)

    matcher = chemical_matcher.ChemicalMatcher(nlp=spacy.blank("en"))
    matches = matcher.match("Eugenol was detected in the extract.")

    assert matches
    target = matches[0]
    assert target["text"] == "Eugenol"
    assert target["canonical"] == "Eugenol"
    assert target["preferred_name"] == "Eugenol"
    assert "Clove oil phenol" in target["aliases"]
    assert target["inchikey"] == "RNGBKPVMWVROLL-UHFFFAOYSA-N"
    assert target["smiles"] == "COC1=C(C=CC=C1)CC=C"
    assert target["molecular_formula"] == "C10H12O2"
    assert target["source_db"] == "PubChem"
    assert target["source_url"] == "https://pubchem.ncbi.nlm.nih.gov/compound/3314"


def test_chemical_matcher_supports_normalized_primary_term_variants(tmp_path, monkeypatch):
    import backend.gazetteer.chemical_matcher as chemical_matcher

    _write_variant_term_chemical_csv(tmp_path)
    monkeypatch.setattr(chemical_matcher, "DATA_FILE", tmp_path / "chemical.csv")
    monkeypatch.setattr(chemical_matcher, "CACHE_FILE", tmp_path / "chemical.pkl")
    monkeypatch.setattr(chemical_matcher, "_matcher", None)

    matcher = chemical_matcher.ChemicalMatcher(nlp=spacy.blank("en"))
    matches = matcher.match("4 Allyl 2 methoxyphenol was detected in the extract.")

    assert matches
    target = matches[0]
    assert target["canonical"] == "4-Allyl-2-methoxyphenol"
    assert target["preferred_name"] == "4-Allyl-2-methoxyphenol"


def test_chemical_matcher_does_not_match_synonyms(tmp_path, monkeypatch):
    import backend.gazetteer.chemical_matcher as chemical_matcher

    _write_chemical_csv(tmp_path)
    monkeypatch.setattr(chemical_matcher, "DATA_FILE", tmp_path / "chemical.csv")
    monkeypatch.setattr(chemical_matcher, "CACHE_FILE", tmp_path / "chemical.pkl")
    monkeypatch.setattr(chemical_matcher, "_matcher", None)

    matcher = chemical_matcher.ChemicalMatcher(nlp=spacy.blank("en"))
    matches = matcher.match("Clove oil phenol was detected in the extract.")
    metadata = matcher.lookup("Eugenol")

    assert matches == []
    assert metadata is not None
    assert "Clove oil phenol" in metadata["aliases"]


def test_chemical_matcher_preserves_primary_term_over_synonym_collision(
    tmp_path, monkeypatch
):
    import backend.gazetteer.chemical_matcher as chemical_matcher

    _write_primary_priority_chemical_csv(tmp_path)
    monkeypatch.setattr(chemical_matcher, "DATA_FILE", tmp_path / "chemical.csv")
    monkeypatch.setattr(chemical_matcher, "CACHE_FILE", tmp_path / "chemical.pkl")
    monkeypatch.setattr(chemical_matcher, "_matcher", None)

    matcher = chemical_matcher.ChemicalMatcher(nlp=spacy.blank("en"))

    lookup = matcher.lookup("Linalool")
    assert lookup is not None
    assert lookup["canonical"] == "Linalool"


def test_chemical_matcher_rebuilds_stale_cache(tmp_path, monkeypatch):
    import backend.gazetteer.chemical_matcher as chemical_matcher

    _write_chemical_csv(tmp_path)
    monkeypatch.setattr(chemical_matcher, "DATA_FILE", tmp_path / "chemical.csv")
    monkeypatch.setattr(chemical_matcher, "CACHE_FILE", tmp_path / "chemical.pkl")
    monkeypatch.setattr(chemical_matcher, "_matcher", None)

    matcher = chemical_matcher.ChemicalMatcher(nlp=spacy.blank("en"))
    initial = matcher.lookup("Eugenol")
    assert initial is not None
    assert initial["canonical"] == "Eugenol"

    time.sleep(0.02)
    (tmp_path / "chemical.csv").write_text(
        "term,synonyms,inchikey,smiles,molecular_formula,source_db,source_url\n"
        "Isoeugenol,Clove oil phenol,YLQBMQCUIZJEEH-UHFFFAOYSA-N,COC1=CC=CC(C=C)=C1O,C10H12O2,PubChem,https://pubchem.ncbi.nlm.nih.gov/compound/853433\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(chemical_matcher, "_matcher", None)
    rebuilt = chemical_matcher.ChemicalMatcher(nlp=spacy.blank("en"))
    rebuilt_lookup = rebuilt.lookup("Isoeugenol")
    assert rebuilt_lookup is not None
    assert rebuilt_lookup["canonical"] == "Isoeugenol"


def test_chemical_matcher_rebuilds_incomplete_legacy_cache(tmp_path, monkeypatch):
    import backend.gazetteer.chemical_matcher as chemical_matcher

    _write_chemical_csv(tmp_path)
    monkeypatch.setattr(chemical_matcher, "DATA_FILE", tmp_path / "chemical.csv")
    monkeypatch.setattr(chemical_matcher, "CACHE_FILE", tmp_path / "chemical.pkl")
    monkeypatch.setattr(chemical_matcher, "_matcher", None)

    legacy_cache = {
        "CHEMICAL": {
            "terms": ["eugenol", "4-allyl-2-methoxyphenol"],
            "count": 2,
        }
    }
    with open(tmp_path / "chemical.pkl", "wb") as f:
        pickle.dump(legacy_cache, f)

    matcher = chemical_matcher.ChemicalMatcher(nlp=spacy.blank("en"))
    lookup = matcher.lookup("Eugenol")

    assert lookup is not None
    assert lookup["canonical"] == "Eugenol"


def test_chemical_matcher_rebuilds_cache_on_version_mismatch(tmp_path, monkeypatch):
    import backend.gazetteer.chemical_matcher as chemical_matcher

    _write_chemical_csv(tmp_path)
    monkeypatch.setattr(chemical_matcher, "DATA_FILE", tmp_path / "chemical.csv")
    monkeypatch.setattr(chemical_matcher, "CACHE_FILE", tmp_path / "chemical.pkl")
    monkeypatch.setattr(chemical_matcher, "_matcher", None)

    stale_cache = {
        "CHEMICAL": {
            "terms": ["eugenol"],
            "canonical_map": {"eugenol": "Stale"},
            "metadata_map": {
                "eugenol": {
                    "canonical": "Stale",
                    "preferred_name": "Stale",
                    "aliases": ["eugenol"],
                }
            },
            "aliases_by_canonical": {"Stale": ["eugenol"]},
            "cache_version": "v1",
        }
    }
    with open(tmp_path / "chemical.pkl", "wb") as f:
        pickle.dump(stale_cache, f)

    matcher = chemical_matcher.ChemicalMatcher(nlp=spacy.blank("en"))

    lookup = matcher.lookup("Eugenol")
    assert lookup is not None
    assert lookup["canonical"] == "Eugenol"


def test_chemical_matcher_rebuilds_cache_on_source_mtime_mismatch(tmp_path, monkeypatch):
    import backend.gazetteer.chemical_matcher as chemical_matcher

    _write_chemical_csv(tmp_path)
    monkeypatch.setattr(chemical_matcher, "DATA_FILE", tmp_path / "chemical.csv")
    monkeypatch.setattr(chemical_matcher, "CACHE_FILE", tmp_path / "chemical.pkl")
    monkeypatch.setattr(chemical_matcher, "_matcher", None)

    stale_cache = {
        "CHEMICAL": {
            "terms": ["eugenol"],
            "canonical_map": {"eugenol": "Stale"},
            "metadata_map": {
                "eugenol": {
                    "canonical": "Stale",
                    "preferred_name": "Stale",
                    "aliases": ["eugenol"],
                }
            },
            "aliases_by_canonical": {"Stale": ["eugenol"]},
            "cache_version": chemical_matcher.CACHE_VERSION,
            "source_mtime": 0,
        }
    }
    with open(tmp_path / "chemical.pkl", "wb") as f:
        pickle.dump(stale_cache, f)

    matcher = chemical_matcher.ChemicalMatcher(nlp=spacy.blank("en"))

    lookup = matcher.lookup("Eugenol")
    assert lookup is not None
    assert lookup["canonical"] == "Eugenol"


def test_build_chemical_matcher_reads_term_schema(tmp_path):
    from backend.gazetteer.build_matcher import build_chemical_matcher
    from backend.gazetteer.chemical_matcher import CACHE_VERSION

    _write_chemical_csv(tmp_path)

    cache = build_chemical_matcher(tmp_path / "chemical.csv")

    assert cache["canonical_map"]["eugenol"] == "Eugenol"
    assert cache["metadata_map"]["eugenol"]["preferred_name"] == "Eugenol"
    assert "Clove oil phenol" in cache["metadata_map"]["eugenol"]["aliases"]
    assert cache["cache_version"] == CACHE_VERSION


@pytest.mark.asyncio
async def test_ner_service_process_text_extracts_new_dictionary_entities(
    tmp_path, monkeypatch
):
    import backend.services.ner_engine as ner_engine
    import backend.gazetteer.chemical_matcher as chemical_matcher
    import backend.gazetteer.extraction_method_matcher as extraction_method_matcher
    import backend.gazetteer.development_stage_matcher as development_stage_matcher
    import backend.gazetteer.season_matcher as season_matcher
    import backend.gazetteer.species_matcher as species_matcher

    for module, cache_name in [
        (extraction_method_matcher, "extraction.pkl"),
        (development_stage_matcher, "development.pkl"),
        (season_matcher, "season.pkl"),
    ]:
        monkeypatch.setattr(module, "CACHE_FILE", tmp_path / cache_name)
        monkeypatch.setattr(module, "_matcher", None)

    _write_species_csv(tmp_path)
    _write_chemical_csv(tmp_path)
    monkeypatch.setattr(species_matcher, "DATA_FILE", tmp_path / "species.csv")
    monkeypatch.setattr(species_matcher, "CACHE_FILE", tmp_path / "species.pkl")
    monkeypatch.setattr(species_matcher, "_matcher", None)
    monkeypatch.setattr(chemical_matcher, "DATA_FILE", tmp_path / "chemical.csv")
    monkeypatch.setattr(chemical_matcher, "CACHE_FILE", tmp_path / "chemical.pkl")
    monkeypatch.setattr(chemical_matcher, "_matcher", None)

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
            "Leaves of Ocimum sanctum were collected in fall season at full bloom after hot continuous extraction and profiled by gc-ms. Eugenol was the major compound."
        ),
    )

    canonical_by_label = {
        (entity["label"], entity.get("canonical")) for entity in entities
    }

    assert ("PLANT PART", "leaf") in canonical_by_label
    assert ("SEASON", "autumn") in canonical_by_label
    assert ("DEVELOPMENT STAGE", "full flowering") in canonical_by_label
    assert ("EXTRACTION METHOD", "soxhlet extraction") in canonical_by_label

    species_entity = next(entity for entity in entities if entity["label"] == "SPECIES")
    assert species_entity["canonical"] == "Ocimum tenuiflorum"
    assert species_entity["accepted_scientific_name"] == "Ocimum tenuiflorum"
    assert species_entity["common_name"] == "Tulsi"
    assert species_entity["taxon_id"] == "1"

    chemical_entity = next(
        entity for entity in entities if entity["label"] == "CHEMICAL"
    )
    assert chemical_entity["canonical"] == "Eugenol"
    assert chemical_entity["preferred_name"] == "Eugenol"
    assert chemical_entity["inchikey"] == "RNGBKPVMWVROLL-UHFFFAOYSA-N"
    assert chemical_entity["smiles"] == "COC1=C(C=CC=C1)CC=C"
    assert chemical_entity["molecular_formula"] == "C10H12O2"
    assert chemical_entity["source_db"] == "PubChem"
    assert (
        chemical_entity["source_url"]
        == "https://pubchem.ncbi.nlm.nih.gov/compound/3314"
    )


def test_enrich_chemical_like_entity_supports_chemical_metadata(tmp_path, monkeypatch):
    import backend.services.ner_engine as ner_engine
    import backend.gazetteer.chemical_matcher as chemical_matcher

    _write_chemical_csv(tmp_path)
    monkeypatch.setattr(chemical_matcher, "DATA_FILE", tmp_path / "chemical.csv")
    monkeypatch.setattr(chemical_matcher, "CACHE_FILE", tmp_path / "chemical.pkl")
    monkeypatch.setattr(chemical_matcher, "_matcher", None)

    matcher = chemical_matcher.ChemicalMatcher(nlp=spacy.blank("en"))
    chemical_entity = {
        "text": "Eugenol",
        "label": "CHEMICAL",
        "score": 1.0,
        "name_type": None,
        "linked_to": None,
    }

    ner_engine.enrich_chemical_like_entity(chemical_entity, matcher)

    assert chemical_entity["canonical"] == "Eugenol"
    assert chemical_entity["preferred_name"] == "Eugenol"
    assert chemical_entity["inchikey"] == "RNGBKPVMWVROLL-UHFFFAOYSA-N"
    assert chemical_entity["smiles"] == "COC1=C(C=CC=C1)CC=C"
    assert chemical_entity["molecular_formula"] == "C10H12O2"
    assert chemical_entity["source_db"] == "PubChem"
    assert (
        chemical_entity["source_url"]
        == "https://pubchem.ncbi.nlm.nih.gov/compound/3314"
    )


def test_parse_llm_response_normalizes_drug_to_chemical():
    import backend.services.ner_engine as ner_engine

    service = object.__new__(ner_engine.NERService)
    service.all_labels = list(ner_engine.LABEL_DEFINITIONS.keys())

    parsed = service.parse_llm_response(
        '[{"span":"streptozotocin","type":"DRUG","start":0,"end":15,"name_type":null,"linked_to":null}]'
    )

    assert parsed
    assert parsed[0]["label"] == "CHEMICAL"


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


def test_entity_schema_preserves_chemical_metadata_fields():
    from backend.schemas.schemas import Entity

    entity = Entity(
        text="Eugenol",
        label="CHEMICAL",
        score=1.0,
        canonical="Eugenol",
        preferred_name="Eugenol",
        inchikey="RNGBKPVMWVROLL-UHFFFAOYSA-N",
        smiles="COC1=C(C=CC=C1)CC=C",
        molecular_formula="C10H12O2",
        source_db="PubChem",
        source_url="https://pubchem.ncbi.nlm.nih.gov/compound/3314",
    )

    assert entity.preferred_name == "Eugenol"
    assert entity.inchikey == "RNGBKPVMWVROLL-UHFFFAOYSA-N"
    assert entity.smiles == "COC1=C(C=CC=C1)CC=C"
    assert entity.molecular_formula == "C10H12O2"


@pytest.mark.asyncio
async def test_process_sections_tracks_section_field(tmp_path, monkeypatch):
    """Test that process_sections adds section field to entities."""
    import backend.services.ner_engine as ner_engine
    import backend.gazetteer.chemical_matcher as chemical_matcher

    monkeypatch.setattr(chemical_matcher, "_matcher", None)

    async def fake_call_llm(self, text_chunk: str) -> str:
        return ""

    def fake_deduplicate(self, entities, text):
        return {}, entities

    monkeypatch.setattr(ner_engine.NERService, "call_llm", fake_call_llm)
    monkeypatch.setattr(ner_engine.NERService, "deduplicate", fake_deduplicate)

    service = ner_engine.NERService()
    sections = [
        {"title": "Abstract", "content": "Eugenol is a phenylpropene."},
        {"title": "Methods", "content": "Leaves were collected."},
    ]
    _, entities = await service.process_sections(sections)

    # All entities should have a section field
    for entity in entities:
        assert "section" in entity, f"Entity {entity['text']} missing section field"
        assert entity["section"] in ["Abstract", "Methods"]


@pytest.mark.asyncio
async def test_process_sections_handles_empty():
    """Test that process_sections handles empty sections gracefully."""
    import backend.services.ner_engine as ner_engine

    service = ner_engine.NERService()
    summary, entities = await service.process_sections([])
    assert summary == {}
    assert entities == []

    summary, entities = await service.process_sections([{"title": "Empty", "content": ""}])
    assert summary == {}
