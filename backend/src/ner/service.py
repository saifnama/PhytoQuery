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

from backend.src.ner.dictionary import _DictionaryMixin, enrich_chemical_like_entity
from backend.src.ner.llm import LABEL_DEFINITIONS, _LLMMixin
from backend.src.settings import (
    NER_BUDGET_SECONDS,
    NER_CONFIDENCE_THRESHOLD,
    NER_HYBRID,
    _safe_int,
)


class NERService(_DictionaryMixin, _LLMMixin):
    """Hybrid NER. Dictionary + LLM paths live in the _*Mixin splits."""

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
        # 1. Dictionary-based extraction — all 8 matchers run once, with
        #    dedup (longest overlapping span wins) applied automatically.
        dict_entities = self._match_dictionary_in_text(text)

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
        # the specific error so a small local model can correct itself
        # instead of silently returning []. Dictionary entities are
        # still the safety net if the LLM path errors out entirely.
        # NER_HYBRID=false skips this whole phase (dictionary-only).
        llm_entities = []
        if not NER_HYBRID:
            logger.info("NER_HYBRID=false — dictionary-only extraction.")
        else:
            try:
                for chunk in chunks:
                    parsed = await self._extract_entities_with_retry(chunk)
                    llm_entities.extend(parsed)
            except Exception as e:
                logger.warning(
                    f"LLM extraction failed: {e}. Using dictionary entities only."
                )

        # 5. Combine dict entities (already dedup'd) + LLM entities
        all_entities = dict_entities + llm_entities

        # 6. Normalize all entities to canonical form
        from backend.src.ner.dictionaries.plant_part import (
            get_matcher as get_plant_matcher,
        )
        from backend.src.ner.dictionaries.analytical_technique import (
            get_matcher as get_analytical_matcher,
        )

        plant_matcher = get_plant_matcher()
        analytical_matcher = get_analytical_matcher()
        from backend.src.ner.dictionaries.extraction_method import (
            get_matcher as get_extraction_matcher,
        )

        extraction_matcher = get_extraction_matcher()
        from backend.src.ner.dictionaries.development_stage import (
            get_matcher as get_development_matcher,
        )

        development_matcher = get_development_matcher()
        from backend.src.ner.dictionaries.season import (
            get_matcher as get_season_matcher,
        )

        season_matcher = get_season_matcher()
        from backend.src.ner.dictionaries.species import get_matcher as get_species_matcher
        from backend.src.ner.dictionaries.chemical import (
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

        # Filter empty sections once so the parallel LLM batch isn't
        # padded with no-op coroutines and the dictionary loop skips
        # them too.
        valid_sections = [
            s for s in sections
            if (s.get("content", "") or "").strip()
        ]
        if not valid_sections:
            return {}, []

        # Dictionary matching stays sequential — it's sync CPU work and
        # already fast at typical paper sizes; thread-offload would add
        # more orchestration than it saves here.
        all_dict_entities: List[Dict[str, Any]] = []
        for section in valid_sections:
            section_title = section.get("title", "Unknown")
            section_text = section.get("content", "")
            for ent in self._match_dictionary_in_text(section_text):
                ent["section"] = section_title
                all_dict_entities.append(ent)

        # LLM extraction in parallel across sections, bounded by a
        # semaphore so we don't overwhelm a single-GPU llama.cpp server
        # with concurrent requests. Default 1 for local GPU; set
        # NER_CONCURRENCY env var to raise for cloud providers.
        # NER_HYBRID=false skips this whole phase (dictionary-only).
        all_llm_entities: List[Dict[str, Any]] = []
        if not NER_HYBRID:
            logger.info("NER_HYBRID=false — dictionary-only extraction.")
            skipped_sections = 0
        else:
            _llm_concurrency = max(1, _safe_int("NER_CONCURRENCY", 1))
            sem = asyncio.Semaphore(_llm_concurrency)

            # Wall-clock budget for the LLM phase. Unreliable/slow providers
            # (e.g. rate-limited free tiers) used to stall the whole
            # /paper/json request past the frontend timeout; once the budget
            # is spent the remaining sections simply keep their dictionary
            # entities instead of waiting on the LLM.
            budget_seconds = NER_BUDGET_SECONDS
            deadline = (
                time.perf_counter() + budget_seconds if budget_seconds > 0 else None
            )
            skipped_sections = 0

            async def _llm_for_section(section: Dict[str, str]) -> List[Dict[str, Any]]:
                nonlocal skipped_sections
                section_title = section.get("title", "Unknown")
                section_text = section.get("content", "")
                async with sem:
                    # Re-check inside the semaphore: with concurrency 1 the
                    # queued sections only resume after earlier calls finish,
                    # long past the budget deadline they saw at gather time.
                    if deadline is not None and time.perf_counter() > deadline:
                        skipped_sections += 1
                        return []
                    try:
                        parsed = await self._extract_entities_with_retry(section_text)
                    except Exception as exc:
                        logger.warning(
                            f"LLM extraction failed for section '{section_title}': {exc}"
                        )
                        return []
                return [{**e, "section": section_title} for e in parsed]

            # return_exceptions=True keeps one section's hard failure from
            # cancelling the rest of the batch — default gather() would
            # propagate the first exception and cancel siblings mid-flight,
            # losing their results.
            section_results = await asyncio.gather(
                *[_llm_for_section(s) for s in valid_sections],
                return_exceptions=True,
            )

            for result in section_results:
                if isinstance(result, BaseException):
                    logger.warning(f"LLM section task raised: {result}")
                    continue
                all_llm_entities.extend(result)

            if skipped_sections:
                logger.warning(
                    f"NER LLM budget ({NER_BUDGET_SECONDS:.0f}s) exhausted: "
                    f"{skipped_sections}/{len(valid_sections)} sections fell back "
                    f"to dictionary-only entities"
                )

        # Normalize entities
        normalized = self._normalize_entities(all_dict_entities + all_llm_entities)

        # Reconstruct full text for hallucination checking
        full_text = " ".join([s.get("content", "") for s in sections])

        summary, filtered = self.deduplicate(normalized, full_text)
        return summary, filtered


    def _normalize_entities(self, all_entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Normalize entities to canonical forms."""
        from backend.src.ner.dictionaries.plant_part import get_matcher as get_plant_matcher
        from backend.src.ner.dictionaries.analytical_technique import get_matcher as get_analytical_matcher
        from backend.src.ner.dictionaries.extraction_method import get_matcher as get_extraction_matcher
        from backend.src.ner.dictionaries.development_stage import get_matcher as get_development_matcher
        from backend.src.ner.dictionaries.season import get_matcher as get_season_matcher
        from backend.src.ner.dictionaries.species import get_matcher as get_species_matcher
        from backend.src.ner.dictionaries.chemical import get_matcher as get_chemical_matcher

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


ner_service = NERService()


