"""Split of services/ner_engine.py (Phase 3). Verbatim moves — no logic changes."""
import asyncio
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from backend.src.settings import (
    NER_BUDGET_SECONDS,
    NER_CHUNK_WORDS,
    NER_CONFIDENCE_THRESHOLD,
    NER_HYBRID,
    NER_MAX_ATTEMPTS,
)
from backend.src.common.llm_client import (
    LLMAuthError,
    LLMRateLimitError,
    LLMResponse,
    get_llm_client,
)
from backend.src.ner.dictionaries.analytical_technique import match_analytical_techniques
from backend.src.ner.dictionaries.bioactivity import match_bioactivities
from backend.src.ner.dictionaries.chemical import match_chemicals
from backend.src.ner.dictionaries.plant_part import match_plant_parts
from backend.src.ner.dictionaries.species import match_species

class _DictionaryMixin:
    pass  # methods attached below (verbatim moves)

    def _match_dictionary_in_text(self, text: str) -> List[Dict[str, Any]]:
        """Run all dictionary matchers on text and return normalized entities."""
        entities = []

        # 1. Plant parts
        for e in match_plant_parts(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "PLANT PART")),
                "score": e.get("score", 1.0),
                "start": e.get("start"),
                "end": e.get("end"),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            })

        # 2. Analytical techniques
        for e in match_analytical_techniques(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "ANALYTICAL TECHNIQUE")),
                "score": e.get("score", 1.0),
                "start": e.get("start"),
                "end": e.get("end"),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            })

        # 3. Extraction methods
        from backend.src.ner.dictionaries.extraction_method import match_extraction_methods
        for e in match_extraction_methods(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "EXTRACTION METHOD")),
                "score": e.get("score", 1.0),
                "start": e.get("start"),
                "end": e.get("end"),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            })

        # 4. Development stages
        from backend.src.ner.dictionaries.development_stage import match_development_stages
        for e in match_development_stages(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "DEVELOPMENT STAGE")),
                "score": e.get("score", 1.0),
                "start": e.get("start"),
                "end": e.get("end"),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            })

        # 5. Seasons
        from backend.src.ner.dictionaries.season import match_seasons
        for e in match_seasons(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "SEASON")),
                "score": e.get("score", 1.0),
                "start": e.get("start"),
                "end": e.get("end"),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            })

        # 6. Species
        for e in match_species(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "SPECIES")),
                "score": e.get("score", 1.0),
                "start": e.get("start"),
                "end": e.get("end"),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
                "name_type": e.get("name_type"),
                "accepted_scientific_name": e.get("accepted_scientific_name"),
                "common_name": e.get("common_name"),
                "source_db": e.get("source_db"),
                "source_url": e.get("source_url"),
                "taxon_id": e.get("taxon_id"),
                "match_status": e.get("match_status"),
                "review_required": e.get("review_required"),
                "scientific_name_verified": e.get("scientific_name_verified"),
            })

        # 7. Chemicals
        for e in match_chemicals(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "CHEMICAL")),
                "score": e.get("score", 1.0),
                "start": e.get("start"),
                "end": e.get("end"),
                "canonical": e.get("canonical"),
                "preferred_name": e.get("preferred_name"),
                "aliases": e.get("aliases"),
                "inchikey": e.get("inchikey"),
                "smiles": e.get("smiles"),
                "molecular_formula": e.get("molecular_formula"),
                "source_db": e.get("source_db"),
                "source_url": e.get("source_url"),
            })

        # 8. Bioactivities
        from backend.src.ner.dictionaries.bioactivity import match_bioactivities
        for e in match_bioactivities(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "BIOACTIVITY")),
                "score": e.get("score", 1.0),
                "start": e.get("start"),
                "end": e.get("end"),
                "canonical": e.get("canonical"),
                "synonyms": e.get("synonyms"),
            })

        # Deduplicate overlapping spans - longest match wins
        kept = []
        for e in sorted(entities, key=lambda x: (x.get("start") or 0, -(x.get("end") or 0))):
            s, en = e.get("start"), e.get("end")
            if s is None or en is None:
                kept.append(e)
                continue
            overlaps = any(
                s < k["end"] and en > k["start"]
                for k in kept
                if "start" in k and "end" in k
            )
            if not overlaps:
                kept.append(e)
        entities = kept

        return entities


    def split_into_word_chunks(
        self, text: str, chunk_size: int = NER_CHUNK_WORDS
    ) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks


def enrich_chemical_like_entity(entity: Dict[str, Any], chemical_matcher: Any) -> None:
    chemical_metadata = chemical_matcher.lookup(entity.get("text", ""))
    if chemical_metadata:
        for key, value in chemical_metadata.items():
            if key in {"text", "span", "start", "end"}:
                continue
            if value not in (None, "", []):
                entity[key] = value
    if not entity.get("canonical"):
        text_lower = entity.get("text", "").lower()
        entity["canonical"] = chemical_matcher.canonical_map.get(
            text_lower, entity.get("text", "")
        )


def preload_gazetteers() -> int:
    """Compile all dictionary matchers now (spaCy + ~300K terms).

    Matchers are module singletons, so warming them here is exactly what
    first-request loading does — just moved to backend startup so the
    first user never pays the ~minute compile cost. Safe to skip on
    failure: requests fall back to lazy loading as before.
    """
    from backend.src.ner.dictionaries.analytical_technique import (
        get_matcher as get_analytical_matcher,
    )
    from backend.src.ner.dictionaries.bioactivity import (
        get_matcher as get_bioactivity_matcher,
    )
    from backend.src.ner.dictionaries.chemical import (
        get_matcher as get_chemical_matcher,
    )
    from backend.src.ner.dictionaries.development_stage import (
        get_matcher as get_development_matcher,
    )
    from backend.src.ner.dictionaries.extraction_method import (
        get_matcher as get_extraction_matcher,
    )
    from backend.src.ner.dictionaries.plant_part import (
        get_matcher as get_plant_matcher,
    )
    from backend.src.ner.dictionaries.season import (
        get_matcher as get_season_matcher,
    )
    from backend.src.ner.dictionaries.species import (
        get_matcher as get_species_matcher,
    )

    loaders = (
        get_analytical_matcher,
        get_bioactivity_matcher,
        get_chemical_matcher,
        get_development_matcher,
        get_extraction_matcher,
        get_plant_matcher,
        get_season_matcher,
        get_species_matcher,
    )
    for load in loaders:
        load()
    return len(loaders)


