import os
import json
import re
import asyncio
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
from backend.core.http_client import HttpClientManager

# Import dictionary-based matchers
from backend.gazetteer.plant_part_matcher import match_plant_parts
from backend.gazetteer.analytical_technique_matcher import match_analytical_techniques
from backend.gazetteer.species_matcher import match_species
from backend.gazetteer.chemical_matcher import match_chemicals
from backend.gazetteer.bioactivity_matcher import match_bioactivities

logger = logging.getLogger(__name__)

# --- Import from NER config ---
from backend.config import (
    NER_OLLAMA_URL,
    NER_OLLAMA_MODEL,
    NER_OPENROUTER_API_KEY,
    NER_OPENROUTER_MODEL,
    NER_GROQ_API_KEY,
    NER_GROQ_MODEL,
    NER_LLAMACPP_URL,
    NER_LLAMACPP_API_KEY,
    NER_LLAMACPP_MODEL,
    NER_CONFIDENCE_THRESHOLD,
    NER_CHUNK_SIZE_WORDS,
    GROQ_URL,
    OPENROUTER_URL,
    _normalize_openai_compat_url,
    get_ner_provider,
)


def get_active_provider():
    """Determine which LLM provider to use as the primary for NER.

    Priority: llama.cpp/OpenAI-compat (explicit opt-in) > Ollama (local,
    fast for bulk) > Groq (cloud-fast) > OpenRouter (cloud-diverse).
    Other configured providers remain available as fall-throughs in
    ``call_llm``.
    """
    if NER_LLAMACPP_URL:
        return "llamacpp"
    if NER_OLLAMA_URL:
        return "ollama"
    if NER_GROQ_API_KEY:
        return "groq"
    if NER_OPENROUTER_API_KEY:
        return "openrouter"
    raise ValueError(
        "No LLM provider configured. Set NER_LLAMACPP_URL, NER_OLLAMA_URL, "
        "NER_GROQ_API_KEY, or NER_OPENROUTER_API_KEY"
    )


# --- System Prompt (Llama 3.1 8B Ethnobotany NER) ---
SYSTEM_PROMPT = """You are a highly precise Named Entity Recognition (NER) specialist for phytochemical and ethnobotanical research.

Your task is to extract named entities from scientific text and return them as a structured JSON array.

═══════════════════════════════════════
ENTITY TYPES TO EXTRACT
═══════════════════════════════════════

1. CHEMICAL
   - Any chemical compound, constituent, or substance
   - Includes: essential oil constituents, flavonoids, alkaloids, terpenes, phenolics, solvents
   - Examples: eugenol, quercetin, linalool, β-caryophyllene, tannins, methanol, hexane

2. SPECIES
   - Any organism reference — whether by scientific binomial name OR common name
   - Includes: plants, fungi, bacteria, insects, animals used in studies or as test organisms
   - Add a "name_type" field: "scientific" if binomial/trinomial, "common" if vernacular
   - Examples (scientific): Ocimum sanctum, Cinnamomum verum, Azadirachta indica, Candida albicans, Staphylococcus aureus
   - Examples (common): tulsi, neem, basil, yeast, staph bacteria

3. LOCATION
   - Geographic region where organism was collected, studied, or reported
   - Includes: countries, states, districts, forest names, altitude descriptions
   - Examples: Western Ghats, Wayanad, Kerala, Himalayan foothills

4. BIOACTIVITY
   - Biological, pharmacological, or chemical activity/property of a compound.
   - Examples: antimicrobial, antioxidant, anti-inflammatory, cytotoxic, antifungal, larvicidal

5. DISEASE
   - Medical condition, disease, or pathological state
   - Examples: malaria, diabetes, tuberculosis, leishmaniasis, oxidative stress

════════════════════════════════════════
DISAMBIGUATION RULES
════════════════════════════════════════

- ANALYTICAL TECHNIQUE vs EXTRACTION METHOD: Always prefer ANALYTICAL TECHNIQUE when the technique is a specific separation process (e.g., steam distillation, HPLC). Use EXTRACTION METHOD for bulk crude extraction processes (e.g., Soxhlet, maceration).
- SPECIES (scientific) vs SPECIES (common): Same entity type — distinguish only via the "name_type" field. Both "tulsi" and "Ocimum sanctum" → SPECIES.
- Nested entities are allowed. Extract overlapping entities separately.
- If a span is ambiguous between two types, choose the most specific and contextually dominant type.

════════════════════════════════════════
REASONING STEP (MANDATORY)
════════════════════════════════════════

Before returning the JSON, reason briefly in this format:

<reasoning>
1. Candidate spans: [list all potential entities you see]
2. Ambiguities: [list any spans that could fit multiple types and your resolution]
3. Nested entities: [note any spans that overlap and need separate extraction]
</reasoning>

Then output the final JSON.

════════════════════════════════════════
OUTPUT FORMAT
════════════════════════════════════════

Return a JSON array where each entity is an object with these fields:

{
  "span": "<exact text from the input>",
  "type": "<ENTITY_TYPE>",
  "start": <character index where span starts>,
  "end": <character index where span ends>,
  "name_type": "<scientific | common | null>",
  "linked_to": "<related entity span if applicable, else null>"
}

Rules:
- "span" must be the EXACT substring from the input text (no paraphrasing)
- "start" and "end" are zero-indexed character offsets
- "name_type" is ONLY populated for SPECIES entities. Set null for all other types.
- "linked_to" is required for BIOACTIVITY (link to the chemical performing the activity). Set null otherwise.
- Return ONLY the <reasoning> block followed by the JSON array. No extra text, no markdown code fences.

════════════════════════════════════════
FEW-SHOT EXAMPLES
════════════════════════════════════════

── EXAMPLE 1 ──

Input:
"Essential oil of Cinnamomum verum bark collected from Wayanad, Kerala was obtained by steam distillation. Eugenol (72.4%) exhibited strong antimicrobial activity against Staphylococcus aureus."

<reasoning>
1. Candidate spans: Cinnamomum verum (species), Wayanad, Kerala (location), steam distillation (isolation method), Eugenol (chemical), antimicrobial (bioactivity), Staphylococcus aureus (species)
2. Ambiguities: None.
</reasoning>

[
  {"span": "Cinnamomum verum",    "type": "SPECIES",           "start": 17,  "end": 33,  "name_type": "scientific", "linked_to": null},
  {"span": "Wayanad, Kerala",     "type": "LOCATION",          "start": 54,  "end": 69,  "name_type": null,         "balanced": null},
  {"span": "steam distillation",  "type": "ANALYTICAL TECHNIQUE",  "start": 86,  "end": 103, "name_type": null,         "linked_to": null},
  {"span": "Eugenol",             "type": "CHEMICAL",          "start": 105, "end": 112, "name_type": null,         "linked_to": null},
  {"span": "antimicrobial",       "type": "BIOACTIVITY", "start": 140, "end": 153, "name_type": null,         "linked_to": "Eugenol"},
  {"span": "Staphylococcus aureus","type": "SPECIES",          "start": 162, "end": 183, "name_type": "scientific", "linked_to": null}
]

── EXAMPLE 2 ──

Input:
"Leaves and roots of neem collected from Tamil Nadu were subjected to Soxhlet extraction using methanol. The crude extract showed antifungal activity against Candida albicans and antidiabetic potential in streptozotocin-induced diabetes models. Nimbolide (3.8%) was isolated via column chromatography and found to inhibit COX-2 enzyme."

<reasoning>
1. Candidate spans: neem (species), Tamil Nadu (location), Soxhlet extraction (extraction method), methanol (chemical), antifungal (bioactivity), Candida albicans (species), antidiabetic (bioactivity), diabetes (disease), streptozotocin (chemical), Nimbolide (chemical), column chromatography (isolation method)
2. Ambiguities: streptozotocin — even in pharmacological model context, normalize compounds to CHEMICAL.
</reasoning>

[
  {"span": "neem",                  "type": "SPECIES",           "start": 20,  "end": 24,  "name_type": "common",     "linked_to": null},
  {"span": "Tamil Nadu",            "type": "LOCATION",          "start": 40,  "end": 50,  "name_type": null,         "linked_to": null},
  {"span": "Soxhlet extraction",    "type": "EXTRACTION METHOD", "start": 68,  "end": 85,  "name_type": null,         "linked_to": null},
  {"span": "methanol",              "type": "CHEMICAL",          "start": 92,  "end": 100, "name_type": null,         "linked_to": null},
  {"span": "antifungal",            "type": "BIOACTIVITY", "start": 120, "end": 130, "name_type": null,         "linked_to": null},
  {"span": "Candida albicans",      "type": "SPECIES",           "start": 139, "end": 155, "name_type": "scientific", "linked_to": null},
  {"span": "antidiabetic",          "type": "BIOACTIVITY", "start": 160, "end": 172, "name_type": null,         "linked_to": null},
  {"span": "streptozotocin",        "type": "CHEMICAL",          "start": 176, "end": 190, "name_type": null,         "linked_to": null},
  {"span": "diabetes",              "type": "DISEASE",           "start": 199, "end": 207, "name_type": null,         "linked_to": null},
  {"span": "Nimbolide",             "type": "CHEMICAL",          "start": 216, "end": 225, "name_type": null,         "linked_to": null},
  {"span": "column chromatography", "type": "ANALYTICAL TECHNIQUE",  "start": 247, "end": 267, "name_type": null,         "linked_to": null}
]
"""

# --- Label Definitions ---
LABEL_DEFINITIONS = {
    # Dictionary-only types (handled by gazetteer, no LLM needed)
    "PLANT PART": "Plant morphological structures (leaf, bark, root, flower, etc.).",
    "ANALYTICAL TECHNIQUE": "Specific separation or analytical technique.",
    "EXTRACTION METHOD": "Physical or mechanical extraction process.",
    "DEVELOPMENT STAGE": "Plant development stage (seedling, flowering, etc.).",
    "SEASON": "Seasonal reference (monsoon, winter, etc.).",
    # LLM-extracted types (no dictionary or requires context)
    "CHEMICAL": "Chemical compounds, natural molecules, phytochemicals.",
    "SPECIES": "Living organisms (plants, bacteria, fungi, animals).",
    "LOCATION": "Geographic locations, regions, institutions.",
    "BIOACTIVITY": "Biological or chemical activity of substances.",
    "DISEASE": "Medical conditions, diseases, disorders.",
}

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


class NERService:
    def __init__(self):
        self.all_labels = list(LABEL_DEFINITIONS.keys())
        self.result_cache = {}  # Cache: DOI -> list of entities

    async def process_text(
        self, text: str, max_chunks: int = 3
    ) -> List[Dict[str, Any]]:
        """Main entry point for NER processing with chunking.

        Args:
            text: Input text to process
            max_chunks: Maximum number of chunks to process (for performance)
        """
        # 1. Dictionary-based extraction (PLANT PARTS) - fast, from CSV
        dict_entities = match_plant_parts(text)
        # Normalize dict entities: span->text, type->label for deduplication
        dict_entities = [
            {
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),  # Include canonical for normalization
                "aliases": e.get("aliases"),
            }
            for e in dict_entities
        ]

        # 1b. Dictionary-based ANALYTICAL TECHNIQUE extraction
        analytical_entities = match_analytical_techniques(text)
        analytical_entities = [
            {
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),  # Include canonical for normalization
                "aliases": e.get("aliases"),
            }
            for e in analytical_entities
        ]

        # 1c. Dictionary-based EXTRACTION METHOD extraction
        from backend.gazetteer.extraction_method_matcher import (
            match_extraction_methods,
        )

        extraction_entities = match_extraction_methods(text)
        extraction_entities = [
            {
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            }
            for e in extraction_entities
        ]

        # 1d. Dictionary-based DEVELOPMENT STAGE extraction
        from backend.gazetteer.development_stage_matcher import (
            match_development_stages,
        )

        development_entities = match_development_stages(text)
        development_entities = [
            {
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            }
            for e in development_entities
        ]

        # 1e. Dictionary-based SEASON extraction
        from backend.gazetteer.season_matcher import match_seasons

        season_entities = match_seasons(text)
        season_entities = [
            {
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            }
            for e in season_entities
        ]

        # 1f. Dictionary-based SPECIES extraction (scientific names only)
        species_entities = match_species(text)

        # 1g. Dictionary-based CHEMICAL extraction
        chemical_entities = match_chemicals(text)
        chemical_entities = [
            {
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "preferred_name": e.get("preferred_name"),
                "aliases": e.get("aliases"),
                "inchikey": e.get("inchikey"),
                "smiles": e.get("smiles"),
                "molecular_formula": e.get("molecular_formula"),
                "source_db": e.get("source_db"),
                "source_url": e.get("source_url"),
            }
            for e in chemical_entities
        ]

        # 1h. Dictionary-based BIOACTIVITY extraction
        bioactivity_entities = match_bioactivities(text)
        bioactivity_entities = [
            {
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "synonyms": e.get("synonyms"),
            }
            for e in bioactivity_entities
        ]

        # 2. Chunking - limit chunks for performance
        chunks = self.split_into_word_chunks(text)
        if len(chunks) > max_chunks:
            # Take first N chunks and join them
            chunks = chunks[:max_chunks]
            text = " ".join(chunks)
        else:
            text = text  # Keep original text reference

        # 4. LLM extraction (sequential to avoid rate limiting)
        # Each chunk goes through ``_extract_entities_with_retry`` —
        # on schema/validation failure we re-prompt the model with
        # the specific error so a 7B local model can correct itself
        # instead of silently returning []. Dictionary entities are
        # still the safety net if the LLM path errors out entirely.
        llm_entities = []
        try:
            for chunk in chunks:
                parsed = await self._extract_entities_with_retry(chunk)
                llm_entities.extend(parsed)
        except Exception as e:
            logger.warning(
                f"LLM extraction failed: {e}. Using dictionary entities only."
            )

        # 5. Combine and deduplicate (only dictionary + LLM)
        all_entities = (
            dict_entities
            + analytical_entities
            + extraction_entities
            + development_entities
            + season_entities
            + species_entities
            + chemical_entities
            + bioactivity_entities
            + llm_entities
        )

        # 6. Normalize all entities to canonical form
        from backend.gazetteer.plant_part_matcher import (
            get_matcher as get_plant_matcher,
        )
        from backend.gazetteer.analytical_technique_matcher import (
            get_matcher as get_analytical_matcher,
        )

        plant_matcher = get_plant_matcher()
        analytical_matcher = get_analytical_matcher()
        from backend.gazetteer.extraction_method_matcher import (
            get_matcher as get_extraction_matcher,
        )

        extraction_matcher = get_extraction_matcher()
        from backend.gazetteer.development_stage_matcher import (
            get_matcher as get_development_matcher,
        )

        development_matcher = get_development_matcher()
        from backend.gazetteer.season_matcher import (
            get_matcher as get_season_matcher,
        )

        season_matcher = get_season_matcher()
        from backend.gazetteer.species_matcher import get_matcher as get_species_matcher
        from backend.gazetteer.chemical_matcher import (
            get_matcher as get_chemical_matcher,
        )

        species_matcher = get_species_matcher()
        chemical_matcher = get_chemical_matcher()

        for e in all_entities:
            text_lower = e.get("text", "").lower()
            label = e.get("label", "")
            if label == "SPECIES":
                species_metadata = species_matcher.lookup(e.get("text", ""))
                if species_metadata:
                    for key, value in species_metadata.items():
                        if key in {"text", "span", "start", "end"}:
                            continue
                        if value not in (None, "", []):
                            e[key] = value
                if not e.get("canonical"):
                    e["canonical"] = species_matcher.canonical_map.get(
                        text_lower, e.get("text", "")
                    )
            elif label == "CHEMICAL":
                enrich_chemical_like_entity(e, chemical_matcher)
            elif e.get("canonical"):
                continue
            elif label == "PLANT PART":
                e["canonical"] = plant_matcher.canonical_map.get(
                    text_lower, e.get("text", "")
                )
            elif label == "ANALYTICAL TECHNIQUE":
                e["canonical"] = analytical_matcher.canonical_map.get(
                    text_lower, e.get("text", "")
                )
            elif label == "EXTRACTION METHOD":
                e["canonical"] = extraction_matcher.canonical_map.get(
                    text_lower, e.get("text", "")
                )
            elif label == "DEVELOPMENT STAGE":
                e["canonical"] = development_matcher.canonical_map.get(
                    text_lower, e.get("text", "")
                )
            elif label == "SEASON":
                e["canonical"] = season_matcher.canonical_map.get(
                    text_lower, e.get("text", "")
                )

        summary, filtered = self.deduplicate(all_entities, text)
        return summary, filtered

    async def process_sections(
        self, sections: List[Dict[str, str]]
    ) -> tuple:
        """Process paper by sections for better entity locality.

        Args:
            sections: List of dicts with 'title' and 'content' keys.
                     E.g., [{"title": "Abstract", "content": "..."}, {"title": "Methods", "content": "..."}]

        Returns:
            (summary, entities) - same as process_text()
        """
        if not sections:
            return {}, []

        # Run dictionary matching on full text (no section tracking for dict)
        all_dict_entities = []

        # Run LLM per section for better context
        all_llm_entities = []
        for section in sections:
            section_title = section.get("title", "Unknown")
            section_text = section.get("content", "")

            if not section_text or not section_text.strip():
                continue

            # Dictionary matching on this section
            for ent in self._match_dictionary_in_text(section_text):
                ent["section"] = section_title
                all_dict_entities.append(ent)

            # LLM extraction on this section — uses validation-retry
            # so semantic LLM failures (wrong type labels, empty
            # arrays, object-wrapped responses) get re-prompted
            # instead of silently dropping all entities for the
            # section.
            try:
                parsed = await self._extract_entities_with_retry(section_text)
                for e in parsed:
                    e["section"] = section_title
                    all_llm_entities.append(e)
            except Exception as e:
                logger.warning(f"LLM extraction failed for section '{section_title}': {e}")

        # Normalize entities
        normalized = self._normalize_entities(all_dict_entities + all_llm_entities)

        # Reconstruct full text for hallucination checking
        full_text = " ".join([s.get("content", "") for s in sections])

        summary, filtered = self.deduplicate(normalized, full_text)
        return summary, filtered

    def _match_dictionary_in_text(self, text: str) -> List[Dict[str, Any]]:
        """Run all dictionary matchers on text and return normalized entities."""
        entities = []

        # 1. Plant parts
        for e in match_plant_parts(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "PLANT PART")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            })

        # 2. Analytical techniques
        for e in match_analytical_techniques(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "ANALYTICAL TECHNIQUE")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            })

        # 3. Extraction methods
        from backend.gazetteer.extraction_method_matcher import match_extraction_methods
        for e in match_extraction_methods(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "EXTRACTION METHOD")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            })

        # 4. Development stages
        from backend.gazetteer.development_stage_matcher import match_development_stages
        for e in match_development_stages(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "DEVELOPMENT STAGE")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            })

        # 5. Seasons
        from backend.gazetteer.season_matcher import match_seasons
        for e in match_seasons(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "SEASON")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "aliases": e.get("aliases"),
            })

        # 6. Species
        for e in match_species(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "SPECIES")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
            })

        # 7. Chemicals
        for e in match_chemicals(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "CHEMICAL")),
                "score": e.get("score", 1.0),
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
        from backend.gazetteer.bioactivity_matcher import match_bioactivities
        for e in match_bioactivities(text):
            entities.append({
                "text": e.get("span", e.get("text", "")),
                "label": e.get("type", e.get("label", "BIOACTIVITY")),
                "score": e.get("score", 1.0),
                "canonical": e.get("canonical"),
                "synonyms": e.get("synonyms"),
            })

        return entities

    def _normalize_entities(self, all_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize entities to canonical forms."""
        from backend.gazetteer.plant_part_matcher import get_matcher as get_plant_matcher
        from backend.gazetteer.analytical_technique_matcher import get_matcher as get_analytical_matcher
        from backend.gazetteer.extraction_method_matcher import get_matcher as get_extraction_matcher
        from backend.gazetteer.development_stage_matcher import get_matcher as get_development_matcher
        from backend.gazetteer.season_matcher import get_matcher as get_season_matcher
        from backend.gazetteer.species_matcher import get_matcher as get_species_matcher
        from backend.gazetteer.chemical_matcher import get_matcher as get_chemical_matcher

        plant_matcher = get_plant_matcher()
        analytical_matcher = get_analytical_matcher()
        extraction_matcher = get_extraction_matcher()
        development_matcher = get_development_matcher()
        season_matcher = get_season_matcher()
        species_matcher = get_species_matcher()
        chemical_matcher = get_chemical_matcher()

        for e in all_entities:
            text_lower = e.get("text", "").lower()
            label = e.get("label", "")

            if label == "SPECIES":
                species_metadata = species_matcher.lookup(e.get("text", ""))
                if species_metadata:
                    for key, value in species_metadata.items():
                        if key in {"text", "span", "start", "end"}:
                            continue
                        if value not in (None, "", []):
                            e[key] = value
                if not e.get("canonical"):
                    e["canonical"] = species_matcher.canonical_map.get(text_lower, e.get("text", ""))
            elif label == "CHEMICAL":
                chemical_metadata = chemical_matcher.lookup(e.get("text", ""))
                if chemical_metadata:
                    for key, value in chemical_metadata.items():
                        if key in {"text", "span", "start", "end"}:
                            continue
                        if value not in (None, "", []):
                            e[key] = value
                if not e.get("canonical"):
                    e["canonical"] = chemical_matcher.canonical_map.get(text_lower, e.get("text", ""))
            elif label == "PLANT PART":
                e["canonical"] = plant_matcher.canonical_map.get(text_lower, e.get("text", ""))
            elif label == "ANALYTICAL TECHNIQUE":
                e["canonical"] = analytical_matcher.canonical_map.get(text_lower, e.get("text", ""))
            elif label == "EXTRACTION METHOD":
                e["canonical"] = extraction_matcher.canonical_map.get(text_lower, e.get("text", ""))
            elif label == "DEVELOPMENT STAGE":
                e["canonical"] = development_matcher.canonical_map.get(text_lower, e.get("text", ""))
            elif label == "SEASON":
                e["canonical"] = season_matcher.canonical_map.get(text_lower, e.get("text", ""))

        return all_entities

    def split_into_word_chunks(
        self, text: str, chunk_size: int = NER_CHUNK_SIZE_WORDS
    ) -> List[str]:
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i : i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks

    async def call_llm(
        self,
        text_chunk: str,
        error_hint: Optional[str] = None,
    ) -> str:
        """Call LLM for NER: Ollama first, then OpenRouter fallback.

        ``error_hint`` — when set, prepended to the user message so
        the model can see what went wrong with its previous attempt
        (validation-retry pattern). ``None`` preserves the original
        single-shot behavior for callers that don't need retry.
        """
        provider = None
        try:
            provider = get_active_provider()
        except Exception as e:
            logger.error(f"NER provider config error: {e}")
            return ""

        # User-facing message body. The error hint is prepended as a
        # correction block so it's the first thing the model attends
        # to; the original "Extract entities from..." instruction
        # remains stable so the system prompt + few-shot examples
        # still apply cleanly.
        if error_hint:
            user_content = (
                "Your previous response was rejected for the "
                f"following reason:\n  {error_hint}\n\n"
                "Retry with a corrected response that follows the "
                "schema from the system prompt.\n\n"
                f"Extract entities from:\n\n{text_chunk}\n\n/no_think"
            )
        else:
            user_content = f"Extract entities from:\n\n{text_chunk}\n\n/no_think"

        # Try self-hosted OpenAI-compatible (llama.cpp / vLLM / LM
        # Studio) first when configured — same wire format as
        # Groq/OpenRouter so we delegate to ``_call_openai_compatible``
        # with the normalized endpoint URL.
        if provider == "llamacpp" and NER_LLAMACPP_URL:
            content = await self._call_openai_compatible(
                provider_name="llamacpp",
                url=_normalize_openai_compat_url(NER_LLAMACPP_URL),
                api_key=NER_LLAMACPP_API_KEY or "",
                model=NER_LLAMACPP_MODEL,
                text_chunk=text_chunk,
                error_hint=error_hint,
            )
            if content:
                return content
            # If llama.cpp didn't return content (server down, model
            # crashed, etc.), fall through to the cloud fallbacks.

        # Try Ollama first (primary)
        if provider == "ollama":
            payload = {
                "model": NER_OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "seed": 42,
                    "num_ctx": 8192,
                    "num_predict": 4096,
                },
            }
            try:
                client = await HttpClientManager.get_client()
                response = await client.post(
                    f"{NER_OLLAMA_URL}/api/chat", json=payload, timeout=300.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"Error calling Ollama LLM: {e}")
                # Fall through to try OpenRouter

        # Try Groq (fallback) — OpenAI-compatible wire format
        if NER_GROQ_API_KEY:
            content = await self._call_openai_compatible(
                provider_name="Groq",
                url=GROQ_URL,
                api_key=NER_GROQ_API_KEY,
                model=NER_GROQ_MODEL,
                text_chunk=text_chunk,
                error_hint=error_hint,
            )
            if content:
                return content

        # Try OpenRouter (fallback) — OpenAI-compatible wire format
        if NER_OPENROUTER_API_KEY:
            content = await self._call_openai_compatible(
                provider_name="OpenRouter",
                url=OPENROUTER_URL,
                api_key=NER_OPENROUTER_API_KEY,
                model=NER_OPENROUTER_MODEL,
                text_chunk=text_chunk,
                error_hint=error_hint,
            )
            if content:
                return content

        return ""

    async def _extract_entities_with_retry(
        self,
        text_chunk: str,
        max_attempts: int = 2,
    ) -> List[Dict[str, Any]]:
        """LLM entity extraction with validation-retry.

        Mirrors the validation-retry pattern of
        ``rag_engine._select_used_chunks``: on each attempt, calls
        the LLM, runs ``json_repair`` for syntax recovery, then
        validates shape (array of objects with at least one
        recognized ``type``/``span`` pair). On semantic failure
        (empty array, all-wrong-type entries, wrong root shape),
        re-prompts the LLM with a specific error so it can correct
        itself rather than silently returning ``[]``.

        Returns a list of entity dicts in the internal format
        ``{"text", "label", "score", "name_type", "linked_to"}``.
        Empty list on terminal failure; callers retain the same
        fallback semantics (dictionary entities still apply).
        """
        if not text_chunk or not text_chunk.strip():
            return []

        try:
            from json_repair import repair_json
        except ImportError:
            # Defensive — json-repair is in requirements but if the
            # package is missing at runtime fall back to single-shot
            # parse via the existing helper.
            raw = await self.call_llm(text_chunk)
            return self.parse_llm_response(raw)

        llm_types = {"CHEMICAL", "SPECIES", "LOCATION", "BIOACTIVITY", "DISEASE"}
        error_hint: Optional[str] = None

        for attempt in range(max_attempts):
            raw = await self.call_llm(text_chunk, error_hint=error_hint)
            if not raw:
                error_hint = (
                    "Your previous response was empty. Return a JSON "
                    'array of entities like '
                    '[{"span":"...", "type":"CHEMICAL", "score":0.9}].'
                )
                continue

            # Strip reasoning blocks before parsing (Qwen-style CoT
            # wrappers — emitted even when /no_think is requested).
            cleaned = re.sub(
                r"<reasoning>.*?</reasoning>", "", raw, flags=re.DOTALL
            ).strip()
            if not cleaned:
                error_hint = (
                    "Your response contained only a <reasoning> block. "
                    "Return the JSON array directly, not inside reasoning tags."
                )
                continue

            try:
                parsed = repair_json(cleaned, return_objects=True)
            except Exception as e:
                logger.warning(
                    f"NER JSON parse failed (attempt {attempt + 1}): {e}"
                )
                error_hint = (
                    "Your previous response could not be parsed as "
                    "JSON. Return a single array, no prose around it."
                )
                continue

            # Locate the entity array — accept both bare-list and
            # object-wrapped shapes.
            entities = None
            if isinstance(parsed, list):
                entities = parsed
            elif isinstance(parsed, dict):
                for key in ("entities", "data", "results", "items"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        entities = value
                        break
                if entities is None:
                    visible_keys = ", ".join(
                        sorted(str(k) for k in parsed.keys())[:6]
                    ) or "<none>"
                    error_hint = (
                        "Expected a top-level JSON array, e.g. "
                        '[{"span":"...", "type":"CHEMICAL"}]. Your '
                        f"object had keys: {visible_keys}."
                    )
                    continue
            else:
                error_hint = (
                    f"Expected a JSON array of entities; got a "
                    f"{type(parsed).__name__} instead."
                )
                continue

            # Convert + validate items. Apply the same filters as the
            # legacy ``parse_llm_response`` so output is byte-identical
            # to the existing pipeline when the first attempt succeeds.
            remap = {"DRUG": "CHEMICAL"}
            result: List[Dict[str, Any]] = []
            for e in entities:
                if not isinstance(e, dict):
                    continue
                text = str(e.get("span", e.get("text", ""))).strip()
                label = str(e.get("type", e.get("label", ""))).strip().upper()
                label = remap.get(label, label)
                if label not in llm_types:
                    continue
                if not text or label not in self.all_labels:
                    continue
                score = float(e.get("score", 1.0))
                result.append({
                    "text": text,
                    "label": label,
                    "score": score,
                    "name_type": e.get("name_type"),
                    "linked_to": e.get("linked_to"),
                })

            if not result:
                # Array shape was right but nothing in it survived
                # validation. Most common cause on small models:
                # wrong type labels (e.g. "Chemical" vs "CHEMICAL",
                # or invented categories like "MOLECULE").
                error_hint = (
                    "None of the entities you returned passed validation. "
                    'Use exact uppercase types from this set: '
                    f"{sorted(llm_types)}. Each item needs "
                    '"span" (entity text) and "type" (one of the labels).'
                )
                continue

            if attempt > 0:
                logger.info(
                    f"NER extraction succeeded on attempt {attempt + 1}/"
                    f"{max_attempts} after validation-retry "
                    f"({len(result)} entities)"
                )
            return result

        logger.warning(
            f"NER extraction: all {max_attempts} attempts failed; "
            f"final error hint: {error_hint!r}"
        )
        return []

    async def _call_openai_compatible(
        self,
        provider_name: str,
        url: str,
        api_key: str,
        model: str,
        text_chunk: str,
        error_hint: Optional[str] = None,
    ) -> str:
        """Generic OpenAI-compatible chat completion call (Groq, OpenRouter).

        Both providers expose the same wire format: POST {model, messages,
        temperature} to /chat/completions, Bearer auth, response shape
        ``{ choices: [ { message: { content } } ] }``.

        ``error_hint`` — when set, prepended to the user message so
        the model can see what went wrong with its previous attempt
        (validation-retry pattern; mirrors ``call_llm``).
        """
        if error_hint:
            user_content = (
                "Your previous response was rejected for the "
                f"following reason:\n  {error_hint}\n\n"
                "Retry with a corrected response that follows the "
                "schema from the system prompt.\n\n"
                f"Extract entities from:\n\n{text_chunk}\n\n/no_think"
            )
        else:
            user_content = f"Extract entities from:\n\n{text_chunk}\n\n/no_think"

        # Only include Authorization when an API key is set. Self-
        # hosted servers (llama.cpp, vLLM) usually run without one —
        # ``Bearer `` with empty token is malformed and some HTTP
        # stacks reject it before reaching the server.
        headers: Dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "stream": False,
            "temperature": 0.0,
        }

        max_retries = 3
        base_delay = 2.0
        for attempt in range(max_retries):
            try:
                client = await HttpClientManager.get_client()
                response = await client.post(
                    url, json=payload, headers=headers, timeout=30.0
                )

                # Handle rate limiting (429) with retry
                if response.status_code == 429:
                    retry_after = int(
                        response.headers.get(
                            "retry-after", base_delay * (2**attempt)
                        )
                    )
                    logger.warning(
                        f"{provider_name} rate limited (429). Retrying after "
                        f"{retry_after}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_after)
                    continue

                if response.status_code == 200:
                    result = response.json()
                    return (
                        result.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                # Non-200 non-429 → fall through to next retry/provider
                logger.warning(
                    f"{provider_name} returned status {response.status_code}: "
                    f"{response.text[:200]}"
                )
            except Exception as e:
                err_str = str(e)
                if "WRONG_VERSION_NUMBER" in err_str or "SSL" in err_str.upper():
                    logger.error(
                        f"SSL handshake failed calling {provider_name} at {url}: {e}. "
                        f"Check that the URL scheme matches the server."
                    )
                else:
                    logger.error(f"Error calling {provider_name} LLM: {e}")
                await asyncio.sleep(base_delay * (2**attempt))
                continue

        return ""

    def parse_llm_response(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse response: strip <reasoning> block, extract JSON,
        and map span/type to text/label for internal compatibility.

        Uses ``json_repair`` for parsing — same pattern as
        ``rag_engine._select_used_chunks``. Small models in JSON mode
        commonly emit trailing commas, single quotes, smart quotes,
        unclosed brackets, or prose preamble around the array; the
        repair pass recovers all of these without a follow-up LLM
        call. The downstream ``isinstance(e, dict)`` guard already
        rejects anything that lands on the wrong shape, so a
        repaired-but-invalid response degrades to ``[]`` exactly as
        a hard-failed parse did before.
        """
        # Strip <reasoning>...</reasoning> block (Qwen-style chain-of-
        # thought wrappers we don't want fed into the JSON parser).
        raw_text = re.sub(
            r"<reasoning>.*?</reasoning>", "", raw_text, flags=re.DOTALL
        ).strip()
        if not raw_text:
            return []

        try:
            from json_repair import repair_json
            parsed = repair_json(raw_text, return_objects=True)
        except Exception:
            return []

        # The model is asked to return a top-level array. Accept that
        # directly; also accept an object whose ``entities``/``data``
        # field is the array (a common drift the prompt doesn't
        # forbid). Anything else degrades to ``[]``.
        if isinstance(parsed, list):
            entities = parsed
        elif isinstance(parsed, dict):
            entities = None
            for key in ("entities", "data", "results", "items"):
                value = parsed.get(key)
                if isinstance(value, list):
                    entities = value
                    break
            if entities is None:
                return []
        else:
            return []

        try:
            result = []
            for e in entities:
                if not isinstance(e, dict):
                    continue
                # Handle both new format (span/type) and legacy format (text/label)
                text = str(e.get("span", e.get("text", ""))).strip()
                label = str(e.get("type", e.get("label", ""))).strip().upper()
                # Remap LLM labels to webapp labels
                remap = {
                    "DRUG": "CHEMICAL",
                }
                label = remap.get(label, label)

                # Only accept LLM-extracted types
                llm_types = {"CHEMICAL", "SPECIES", "LOCATION", "BIOACTIVITY", "DISEASE"}
                if label not in llm_types:
                    continue

                score = float(e.get("score", 1.0))
                if text and label in self.all_labels:
                    result.append(
                        {
                            "text": text,
                            "label": label,
                            "score": score,
                            "name_type": e.get("name_type"),
                            "linked_to": e.get("linked_to"),
                        }
                    )
            return result
        except Exception:
            return []

    def deduplicate(
        self,
        all_entities: List[Dict[str, Any]],
        full_text: str,
        threshold: float = NER_CONFIDENCE_THRESHOLD,
    ):
        filtered = [e for e in all_entities if e["score"] >= threshold]

        # 1. Identify unique entity text->label mappings (case-insensitive)
        # Store scores and a tally of casing variations for each lower-case identity
        id_map = defaultdict(lambda: {"scores": [], "variants": defaultdict(int)})

        for e in filtered:
            text = e["text"].strip()
            label = e["label"]
            lower_text = text.lower()
            key = (lower_text, label)
            id_map[key]["scores"].append(e["score"])
            id_map[key]["variants"][text] += 1

        summary = defaultdict(list)
        full_text_lower = full_text.lower()

        # 2. For each identity, pick the most frequent casing and scan text for counts
        for (lower_text, label), data in id_map.items():
            avg_score = sum(data["scores"]) / len(data["scores"])

            # Pick the casing variant that appeared most often in AI extractions
            display_text = max(data["variants"].items(), key=lambda x: x[1])[0]

            # Fast whole-word matching using regex (case-insensitive because text is lowercased)
            try:
                escaped_text = re.escape(lower_text)
                pattern = r"(?<!\w)" + escaped_text + r"(?!\w)"
                true_count = len(re.findall(pattern, full_text_lower))
            except:
                true_count = full_text_lower.count(lower_text)

            # Discard if it's a hallucination (count 0)
            if true_count > 0:
                summary[label].append(
                    {
                        "text": display_text,
                        "count": true_count,
                        "avg_score": round(avg_score, 2),
                    }
                )

        for label in summary:
            summary[label].sort(key=lambda x: x["count"], reverse=True)

        return dict(summary), filtered


# Singleton instance
ner_service = NERService()
