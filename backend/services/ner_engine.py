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

    Fallback order: Ollama -> OpenRouter.
    """
    if NER_NER_OLLAMA_URL:
        return "ollama"
    if NER_NER_OPENROUTER_API_KEY:
        return "openrouter"
    raise ValueError(
        "No LLM provider configured. Set NER_NER_OLLAMA_URL or NER_NER_OPENROUTER_API_KEY"
    )


def get_active_provider():
    """Determine which LLM provider to use based on config.

    Fallback order: Ollama -> OpenRouter.
    If Ollama URL is configured (non-empty), use Ollama. Otherwise, check OpenRouter.
    """
    # Ollama has top priority when configured
    if NER_OLLAMA_URL:
        return "ollama"
    # OpenRouter as OpenAI-compatible API
    if NER_OPENROUTER_API_KEY:
        return "openrouter"
    raise ValueError(
        "No LLM provider configured. Set NER_OLLAMA_URL or NER_OPENROUTER_API_KEY"
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

3. PLANT PART
   - The specific anatomical part of the plant or organism used
   - Examples: leaves, bark, roots, seeds, flowers, aerial parts, rhizome, heartwood, fruit peel

4. EXTRACTION METHOD
   - Physical or mechanical process used to obtain crude extract
   - Examples: maceration, cold percolation, Soxhlet extraction, cold press, solvent extraction

5. LOCATION
   - Geographic region where organism was collected, studied, or reported
   - Includes: countries, states, districts, forest names, altitude descriptions
   - Examples: Western Ghats, Wayanad, Kerala, Himalayan foothills

6. CHEMICAL ACTIVITY
   - Biological, pharmacological, or chemical activity/property of a compound
   - Examples: antimicrobial, antioxidant, anti-inflammatory, cytotoxic, antifungal, larvicidal

7. ISOLATION METHOD
   - Specific separation or isolation technique (more specific than EXTRACTION METHOD)
   - Examples: steam distillation, hydrodistillation, supercritical CO₂ extraction, column chromatography, HPLC, fractional distillation, liquid-liquid partitioning

8. DISEASE
   - Medical condition, disease, or pathological state
   - Examples: malaria, diabetes, tuberculosis, leishmaniasis, oxidative stress

9. DRUG
   - Named pharmaceutical drug or traditional medicine formulation
   - Use DRUG (not CHEMICAL) when clinical or therapeutic context is dominant
   - Examples: aspirin, artemisinin, Chyawanprash formulation, morphine

10. CHEMICAL LIGAND
    - Molecular, cellular, or biological target of a compound
    - Includes: enzymes, receptors, ligands, proteins, pathways
    - Examples: COX-2 enzyme, ACE receptor, EGFR ligand, acetylcholinesterase, DNA gyrase

════════════════════════════════════════
DISAMBIGUATION RULES
════════════════════════════════════════

- CHEMICAL vs DRUG: If the compound is discussed in a clinical/therapeutic treatment context, label it DRUG. If discussed as a constituent or in activity screening, label it CHEMICAL.
- ISOLATION METHOD vs EXTRACTION METHOD: Always prefer ISOLATION METHOD when the technique is a specific separation process (e.g., steam distillation, HPLC). Use EXTRACTION METHOD for bulk crude extraction processes (e.g., Soxhlet, maceration).
- SPECIES (scientific) vs SPECIES (common): Same entity type — distinguish only via the "name_type" field. Both "tulsi" and "Ocimum sanctum" → SPECIES.
- Nested entities are allowed. "leaves of Cinnamomum verum" → extract PLANT PART ("leaves") and SPECIES ("Cinnamomum verum") as separate entities.
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
- "linked_to" is required for CHEMICAL ACTIVITY (link to the chemical performing the activity). Set null otherwise.
- Return ONLY the <reasoning> block followed by the JSON array. No extra text, no markdown code fences.

════════════════════════════════════════
FEW-SHOT EXAMPLES
════════════════════════════════════════

── EXAMPLE 1 ──

Input:
"Essential oil of Cinnamomum verum bark collected from Wayanad, Kerala was obtained by steam distillation. Eugenol (72.4%) exhibited strong antimicrobial activity against Staphylococcus aureus."

<reasoning>
1. Candidate spans: Cinnamomum verum (species - scientific), bark (plant part), Wayanad, Kerala (location), steam distillation (isolation method), Eugenol (chemical), 72.4% (percentage), antimicrobial (chemical activity), Staphylococcus aureus (species - scientific)
2. Ambiguities: None.
3. Nested entities: None.
</reasoning>

[
  {"span": "Cinnamomum verum",    "type": "SPECIES",           "start": 17,  "end": 33,  "name_type": "scientific", "linked_to": null},
  {"span": "bark",                "type": "PLANT PART",        "start": 34,  "end": 38,  "name_type": null,         "linked_to": null},
  {"span": "Wayanad, Kerala",     "type": "LOCATION",          "start": 54,  "end": 69,  "name_type": null,         "balanced": null},
  {"span": "steam distillation",  "type": "ISOLATION METHOD",  "start": 86,  "end": 103, "name_type": null,         "linked_to": null},
  {"span": "Eugenol",             "type": "CHEMICAL",          "start": 105, "end": 112, "name_type": null,         "linked_to": null},
  {"span": "antimicrobial",       "type": "CHEMICAL ACTIVITY", "start": 140, "end": 153, "name_type": null,         "linked_to": "Eugenol"},
  {"span": "Staphylococcus aureus","type": "SPECIES",          "start": 162, "end": 183, "name_type": "scientific", "linked_to": null}
]

── EXAMPLE 2 ──

Input:
"Leaves and roots of neem collected from Tamil Nadu were subjected to Soxhlet extraction using methanol. The crude extract showed antifungal activity against Candida albicans and antidiabetic potential in streptozotocin-induced diabetes models. Nimbolide (3.8%) was isolated via column chromatography and found to inhibit COX-2 enzyme."

<reasoning>
1. Candidate spans: Leaves, roots (plant parts), neem (species - common), Tamil Nadu (location), Soxhlet extraction (extraction method), methanol (chemical - solvent), antifungal (chemical activity), Candida albicans (species - scientific), antidiabetic (chemical activity), diabetes (disease), streptozotocin (drug - used as disease model inducer), Nimbolide (chemical), 3.8% (percentage), column chromatography (isolation method), COX-2 enzyme (chemical ligand)
2. Ambiguities: streptozotocin — used as a pharmacological tool to induce diabetes, label DRUG. methanol — solvent used in extraction, label CHEMICAL.
3. Nested entities: "Leaves and roots" → two separate PLANT PART entities.
</reasoning>

[
  {"span": "Leaves",                "type": "PLANT PART",        "start": 0,   "end": 6,   "name_type": null,         "linked_to": null},
  {"span": "roots",                 "type": "PLANT PART",        "start": 11,  "end": 16,  "name_type": null,         "linked_to": null},
  {"span": "neem",                  "type": "SPECIES",           "start": 20,  "end": 24,  "name_type": "common",     "linked_to": null},
  {"span": "Tamil Nadu",            "type": "LOCATION",          "start": 40,  "end": 50,  "name_type": null,         "linked_to": null},
  {"span": "Soxhlet extraction",    "type": "EXTRACTION METHOD", "start": 68,  "end": 85,  "name_type": null,         "linked_to": null},
  {"span": "methanol",              "type": "CHEMICAL",          "start": 92,  "end": 100, "name_type": null,         "linked_to": null},
  {"span": "antifungal",            "type": "CHEMICAL ACTIVITY", "start": 120, "end": 130, "name_type": null,         "linked_to": null},
  {"span": "Candida albicans",      "type": "SPECIES",           "start": 139, "end": 155, "name_type": "scientific", "linked_to": null},
  {"span": "antidiabetic",          "type": "CHEMICAL ACTIVITY", "start": 160, "end": 172, "name_type": null,         "linked_to": null},
  {"span": "streptozotocin",        "type": "DRUG",              "start": 176, "end": 190, "name_type": null,         "linked_to": null},
  {"span": "diabetes",              "type": "DISEASE",           "start": 199, "end": 207, "name_type": null,         "linked_to": null},
  {"span": "Nimbolide",             "type": "CHEMICAL",          "start": 216, "end": 225, "name_type": null,         "linked_to": null},
  {"span": "column chromatography", "type": "ISOLATION METHOD",  "start": 247, "end": 267, "name_type": null,         "linked_to": null},
  {"span": "COX-2 enzyme",          "type": "CHEMICAL LIGAND",   "start": 287, "end": 303, "name_type": null,         "linked_to": null}
]
"""

# --- Label Definitions (Exactly as requested: Spaces instead of underscores) ---
LABEL_DEFINITIONS = {
    "CHEMICAL": "Chemical compounds, natural molecules, phytochemicals, metabolites.",
    "SPECIES": "Any living organism (plants, bacteria, fungi, animals, parasites) and taxonomic groups.",
    "PLANT PART": "Parts of a plant used in extraction or referenced as source material.",
    "EXTRACTION METHOD": "Physical or mechanical process used to obtain crude extract.",
    "LOCATION": "Geographic locations, countries, regions, institutions.",
    "CHEMICAL ACTIVITY": "Biological or chemical activities and effects of substances.",
    "ISOLATION METHOD": "Specific separation or isolation technique.",
    "DISEASE": "Diseases, medical conditions, infections, disorders.",
    "DRUG": "Pharmaceutical drugs and synthetic medicines.",
    "CHEMICAL LIGAND": "Molecular, cellular, or biological target of a compound (enzymes, receptors, ligands, proteins, pathways).",
}

RULER_PATTERNS = [
    {"label": "ISOLATION METHOD", "pattern": [{"LOWER": "hplc"}]},
    {"label": "ISOLATION METHOD", "pattern": [{"LOWER": "gc-ms"}]},
    {
        "label": "ISOLATION METHOD",
        "pattern": [{"LOWER": "gc"}, {"TEXT": "-"}, {"LOWER": "ms"}],
    },
    {"label": "ISOLATION METHOD", "pattern": [{"LOWER": "tlc"}]},
    {"label": "ISOLATION METHOD", "pattern": [{"LOWER": "nmr"}]},
    {"label": "ISOLATION METHOD", "pattern": [{"LOWER": "ftir"}]},
    {"label": "ISOLATION METHOD", "pattern": [{"TEXT": "HPLC-MS"}]},
    {"label": "ISOLATION METHOD", "pattern": [{"LOWER": "lc-ms"}]},
    {
        "label": "ISOLATION METHOD",
        "pattern": [{"LOWER": "lc"}, {"TEXT": "-"}, {"LOWER": "ms"}],
    },
    {"label": "ISOLATION METHOD", "pattern": [{"LOWER": "hplc-dad"}]},
    {"label": "ISOLATION METHOD", "pattern": [{"LOWER": "uplc"}]},
    {
        "label": "ISOLATION METHOD",
        "pattern": [{"LOWER": "ms"}, {"TEXT": "/"}, {"LOWER": "ms"}],
    },
    {
        "label": "ISOLATION METHOD",
        "pattern": [{"LOWER": "column"}, {"LOWER": "chromatography"}],
    },
    {
        "label": "ISOLATION METHOD",
        "pattern": [{"LOWER": "thin"}, {"LOWER": "layer"}, {"LOWER": "chromatography"}],
    },
]


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

    async def process_text(self, text: str) -> List[Dict[str, Any]]:
        """Main entry point for NER processing with parallelized chunking."""
        # 1. Rule-based extraction (sequential)
        ruler_entities = self.apply_entity_ruler(text)

        # 2. Chunking
        chunks = self.split_into_word_chunks(text)

        # 3. LLM extraction (parallel)
        async def process_chunk(chunk):
            raw_response = await self.call_llm(chunk)
            return self.parse_llm_response(raw_response)

        tasks = [process_chunk(chunk) for chunk in chunks]
        results_list = await asyncio.gather(*tasks)

        llm_entities = []
        for res in results_list:
            llm_entities.extend(res)

        # 4. Combine and deduplicate
        all_entities = ruler_entities + llm_entities
        summary, filtered = self.deduplicate(
            all_entities, text
        )  # Pass original full text
        return summary, filtered

    def apply_entity_ruler(self, text: str) -> List[Dict[str, Any]]:
        doc = self.nlp(text)
        entities = []
        seen = set()
        for ent in doc.ents:
            key = (ent.start_char, ent.end_char)
            if key not in seen:
                entities.append(
                    {"text": ent.text.strip(), "label": ent.label_, "score": 1.0}
                )
                seen.add(key)
        return entities

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
        """Call Qwen3.5:9b via /api/chat with /no_think to disable built-in thinking."""
        # Build provider-determined payloads and endpoints
        provider = None
        try:
            provider = get_active_provider()
        except Exception as e:
            logger.error(f"NER provider config error: {e}")
            return ""

        # Common payload structure, adapted per provider as needed
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
                "options": {"temperature": 0.0, "seed": 42, "num_ctx": 8192},
            }
            try:
                client = await HttpClientManager.get_client()
                response = await client.post(
                    f"{NER_OLLAMA_URL}/api/chat", json=payload, timeout=120.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return result.get("message", {}).get("content", "")
            except Exception as e:
                logger.error(f"Error calling Ollama LLM: {e}")
            return ""

        # OpenRouter provider (fallback)
        if provider == "openrouter":
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
            try:
                client = await HttpClientManager.get_client()
                response = await client.post(
                    url, json=payload, headers=headers, timeout=120.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return (
                        result.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
            except Exception as e:
                logger.error(f"Error calling OpenRouter LLM: {e}")
            return ""

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
                    "TARGET": "CHEMICAL LIGAND",
                    "CHEMICAL_LIGAND": "CHEMICAL LIGAND",
                    "PLANT_PART": "PLANT PART",
                    "EXTRACTION_METHOD": "EXTRACTION METHOD",
                    "CHEMICAL_ACTIVITY": "CHEMICAL ACTIVITY",
                    "ISOLATION_METHOD": "ISOLATION METHOD",
                }
                label = remap.get(label, label)

                if label == "PERCENTAGE":
                    continue  # Skip percentages, not displayed in webapp
                score = float(e.get("score", 1.0))
                if text and label in self.all_labels:
                    result.append({"text": text, "label": label, "score": score})
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
