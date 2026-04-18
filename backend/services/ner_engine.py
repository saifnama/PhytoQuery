import os
import json
import spacy
import re
import asyncio
import logging
import warnings
from typing import List, Dict, Any
from collections import defaultdict
from backend.core.http_client import HttpClientManager

# Import dictionary-based matchers
from backend.gazetteer.plant_part_matcher import match_plant_parts
from backend.gazetteer.analytical_technique_matcher import match_analytical_techniques
from backend.gazetteer.species_matcher import match_species
from backend.gazetteer.chemical_matcher import match_chemicals
from backend.gazetteer.bioactivity_matcher import match_bioactivities

# Suppress spaCy FutureWarning about set union in tokenizer
warnings.filterwarnings("ignore", category=FutureWarning, module="spacy.language")

logger = logging.getLogger(__name__)

# --- Import from NER config ---
from backend.config_ner import (
    NER_OLLAMA_URL,
    NER_OLLAMA_MODEL,
    NER_OPENROUTER_API_KEY,
    NER_OPENROUTER_MODEL,
    NER_CONFIDENCE_THRESHOLD,
    NER_CHUNK_SIZE_WORDS,
    get_ner_provider,
)


def get_active_provider():
    """Determine which LLM provider to use based on config.

    Fallback order: OpenRouter -> Ollama.
    OpenRouter has top priority (better quality), Ollama as fallback.
    """
    if NER_OPENROUTER_API_KEY:
        return "openrouter"
    if NER_OLLAMA_URL:
        return "ollama"
    raise ValueError(
        "No LLM provider configured. Set NER_OPENROUTER_API_KEY or NER_OLLAMA_URL"
    )


# --- System Prompt (Llama 3.1 8B Ethnobotany NER) ---
SYSTEM_PROMPT = """You are a highly precise Named Entity Recognition (NER) specialist for phytochemical and ethnobotanical research.

Your task is to extract named entities from scientific text and return them as a structured JSON array.

════════════════════════════════════════
ENTITY TYPES & DEFINITIONS
════════════════════════════════════════

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

3. EXTRACTION METHOD
   - Physical or mechanical process used to obtain crude extract
   - Examples: maceration, cold percolation, Soxhlet extraction, cold press, solvent extraction

5. LOCATION
   - Geographic region where organism was collected, studied, or reported
   - Includes: countries, states, districts, forest names, altitude descriptions
   - Examples: Western Ghats, Wayanad, Kerala, Himalayan foothills

6. BIOACTIVITY
   - Biological, pharmacological, or chemical activity/property of a compound.
   - Examples: antimicrobial, antioxidant, anti-inflammatory, cytotoxic, antifungal, larvicidal

7. ANALYTICAL TECHNIQUE
   - Specific separation or isolation technique (more specific than EXTRACTION METHOD)
   - Examples: steam distillation, hydrodistillation, supercritical CO₂ extraction, column chromatography, HPLC, fractional distillation, liquid-liquid partitioning

8. DISEASE
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

# --- Label Definitions (Exactly as requested: Spaces instead of underscores) ---
LABEL_DEFINITIONS = {
    "CHEMICAL": "Chemical compounds, natural molecules, phytochemicals, metabolites.",
    "SPECIES": "Any living organism (plants, bacteria, fungi, animals, parasites) and taxonomic groups.",
    # PLANT PART - handled by dictionary matching (no LLM needed)
    "EXTRACTION METHOD": "Physical or mechanical process used to obtain crude extract.",
    "LOCATION": "Geographic locations, countries, regions, institutions.",
    "BIOACTIVITY": "Biological or chemical activities and effects of substances.",
    "ANALYTICAL TECHNIQUE": "Specific separation or isolation technique.",
    "DISEASE": "Diseases, medical conditions, infections, disorders.",
}

RULER_PATTERNS = [
    # ANALYTICAL TECHNIQUE patterns removed - now handled by dictionary
]


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
        # Load SciSpaCy base once
        self.nlp = spacy.load("en_core_sci_md", disable=["ner", "parser"])
        if "sentencizer" not in self.nlp.pipe_names:
            self.nlp.add_pipe("sentencizer", first=True)

        ruler = self.nlp.add_pipe("entity_ruler", after="sentencizer")
        ruler.add_patterns(RULER_PATTERNS)

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
        # Even if LLM fails, we still have dictionary + ruler entities
        llm_entities = []
        try:
            for chunk in chunks:
                raw_response = await self.call_llm(chunk)
                parsed = self.parse_llm_response(raw_response)
                llm_entities.extend(parsed)
        except Exception as e:
            logger.warning(
                f"LLM extraction failed: {e}. Using dictionary + ruler entities only."
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

    def apply_entity_ruler(self, text: str) -> List[Dict[str, Any]]:
        """Disabled - using dictionary matchers instead."""
        return []

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

    async def call_llm(self, text_chunk: str) -> str:
        """Call LLM for NER: Ollama first, then OpenRouter fallback."""
        provider = None
        try:
            provider = get_active_provider()
        except Exception as e:
            logger.error(f"NER provider config error: {e}")
            return ""

        # Try Ollama first (primary)
        if provider == "ollama":
            payload = {
                "model": NER_OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Extract entities from:\n\n{text_chunk}\n\n/no_think",
                    },
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

        # Try OpenRouter (fallback)
        if NER_OPENROUTER_API_KEY:
            url = "https://openrouter.ai/api/v1/chat/completions"
            headers = {"Authorization": f"Bearer {NER_OPENROUTER_API_KEY}"}
            payload = {
                "model": NER_OPENROUTER_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Extract entities from:\n\n{text_chunk}\n\n/no_think",
                    },
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
                            f"Rate limited (429). Retrying after {retry_after}s (attempt {attempt + 1}/{max_retries})"
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
                except Exception as e:
                    logger.error(f"Error calling OpenRouter LLM: {e}")
                    await asyncio.sleep(base_delay * (2**attempt))
                    continue

        return ""

    def parse_llm_response(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse response: strip <reasoning> block, extract JSON,
        and map span/type to text/label for internal compatibility."""
        # Strip <reasoning>...</reasoning> block
        raw_text = re.sub(
            r"<reasoning>.*?</reasoning>", "", raw_text, flags=re.DOTALL
        ).strip()
        # Strip markdown code fences if present
        raw_text = re.sub(r"```json|```", "", raw_text).strip()

        match = re.search(r"\[.*\]", raw_text, re.DOTALL)
        if not match:
            return []
        try:
            entities = json.loads(match.group())
            result = []
            for e in entities:
                if not isinstance(e, dict):
                    continue
                # Handle both new format (span/type) and legacy format (text/label)
                text = str(e.get("span", e.get("text", ""))).strip()
                label = str(e.get("type", e.get("label", ""))).strip().upper()
                # Remap system prompt labels to webapp labels (exact requested spaces)
                remap = {
                    "EXTRACTION_METHOD": "EXTRACTION METHOD",
                    "BIOACTIVITY": "BIOACTIVITY",
                    "ANALYTICAL_TECHNIQUE": "ANALYTICAL TECHNIQUE",
                    "DRUG": "CHEMICAL",
                }
                label = remap.get(label, label)

                if label == "PERCENTAGE":
                    continue  # Skip percentages, not displayed in webapp
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
