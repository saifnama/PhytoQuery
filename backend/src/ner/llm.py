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
    LLMConfigError,
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
    LLMTimeoutError,
    LLMUpstreamError,
    get_llm_client,
)

class _LLMMixin:
    pass  # methods attached below (verbatim moves)

    async def call_llm(
        self,
        text_chunk: str,
        error_hint: Optional[str] = None,
    ) -> str:
        """Call the shared unified LLM for NER extraction.

        ``error_hint`` — when set, prepended to the user message so
        the model can see what went wrong with its previous attempt
        (validation-retry pattern). ``None`` preserves the original
        single-shot behavior for callers that don't need retry.

        Returns ``""`` on config/auth/timeout/upstream failures.
        Re-raises ``LLMRateLimitError`` so the retry loop can fail fast
        instead of re-prompting a throttled server.
        """
        # User-facing message body. The error hint is prepended as a
        # correction block so it's the first thing the model attends
        # to; the original "Extract entities from..." instruction
        # remains stable so the system prompt still applies cleanly.
        if error_hint:
            user_content = (
                "Your previous response was rejected for the "
                f"following reason:\n  {error_hint}\n\n"
                "Retry with a corrected response that follows the "
                "schema from the system prompt.\n\n"
                f"Extract entities from:\n\n{text_chunk}"
            )
        else:
            user_content = f"Extract entities from:\n\n{text_chunk}"

        try:
            client = get_llm_client()
        except LLMConfigError as e:
            logger.error(f"NER LLM config error: {e}")
            return ""

        try:
            response = await client.invoke(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=2048,
                timeout_seconds=120.0,
                json_mode=True,
            )
        except LLMRateLimitError:
            raise
        except LLMUpstreamError as e:
            # Some servers reject response_format (HTTP 400 on unknown
            # fields). Retry once in free-text mode rather than losing
            # the section to dictionary-only entities.
            if "400" not in str(e):
                logger.warning(f"NER LLM call failed: {e}")
                return ""
            logger.warning(f"NER JSON mode unsupported ({e}); retrying plain")
            try:
                response = await client.invoke(
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    temperature=0.0,
                    max_tokens=2048,
                    timeout_seconds=120.0,
                )
            except LLMRateLimitError:
                raise
            except (LLMAuthError, LLMTimeoutError, LLMUpstreamError) as retry_e:
                logger.warning(f"NER LLM call failed: {retry_e}")
                return ""
        except (LLMAuthError, LLMTimeoutError) as e:
            logger.warning(f"NER LLM call failed: {e}")
            return ""
        return response.content or ""


    async def _extract_entities_with_retry(
        self,
        text_chunk: str,
        max_attempts: Optional[int] = None,
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

        After ``max_attempts`` failed attempts (default
        ``NER_MAX_ATTEMPTS``), returns ``[]`` — callers keep their
        dictionary-extracted entities, which are always merged in
        ``process_sections`` regardless of the LLM outcome.

        Returns a list of entity dicts in the internal format
        ``{"text", "label", "score", "name_type", "linked_to"}``.
        """
        if not text_chunk or not text_chunk.strip():
            return []

        if max_attempts is None:
            max_attempts = max(1, NER_MAX_ATTEMPTS)

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
            t0 = time.perf_counter()
            try:
                raw = await self.call_llm(text_chunk, error_hint=error_hint)
            except LLMRateLimitError as e:
                # Throttled — re-prompting cannot succeed. Fail fast so
                # the section falls back to dictionary entities instead
                # of burning the time budget on doomed retries.
                logger.warning(f"NER rate limited, skipping retries: {e}")
                return []
            llm_ms = (time.perf_counter() - t0) * 1000
            if not raw:
                error_hint = (
                    "Your previous response was empty. Return a JSON "
                    'array of entities like '
                    '[{"span":"...", "type":"CHEMICAL", "score":0.9}].'
                )
                continue

            # Strip reasoning blocks before parsing (some models emit
            # chain-of-thought wrappers despite the JSON-only instruction).
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
            # object-wrapped shapes. A JSON-encoded *string* (the model
            # quoting the whole array) gets one more repair pass on its
            # inner content before it counts as a failure.
            entities = None
            if isinstance(parsed, str) and parsed.strip():
                try:
                    parsed = repair_json(parsed.strip(), return_objects=True)
                except Exception:
                    pass
            if isinstance(parsed, list):
                entities = parsed
            elif isinstance(parsed, dict):
                if "span" in parsed or "text" in parsed:
                    # Single bare entity object — wrap it.
                    entities = [parsed]
                else:
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
                    "start": e.get("start"),
                    "end": e.get("end"),
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
                    f"({len(result)} entities, {llm_ms:.0f}ms LLM)"
                )
            else:
                logger.info(
                    f"NER extraction: {len(result)} entities in {llm_ms:.0f}ms"
                )
            return result

        logger.warning(
            f"NER extraction: all {max_attempts} attempts failed; "
            f"final error hint: {error_hint!r}"
        )
        return []


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
        # directly, a single bare entity object, or an object whose
        # ``entities``/``data`` field is the array (common drifts the
        # prompt doesn't forbid). Anything else degrades to ``[]``.
        if isinstance(parsed, list):
            entities = parsed
        elif isinstance(parsed, dict):
            if "span" in parsed or "text" in parsed:
                entities = [parsed]
            else:
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


SYSTEM_PROMPT = """You are a precise Named Entity Recognition (NER) specialist for phytochemical and ethnobotanical research.

Extract named entities from scientific text and return ONLY a JSON array — no prose, no markdown fences.

ENTITY TYPES (exactly five — extract nothing else)

1. CHEMICAL — named compounds, phytoconstituents, solvents, reagents. NOT bulk mixtures (essential oil, crude extract).
   Examples: eugenol, quercetin, methanol, streptozotocin

2. SPECIES — plant binomial name only (genus + species). NO common names, NO non-plant organisms.
   Examples: Ocimum sanctum, Cinnamomum verum, Azadirachta indica

3. LOCATION — geographic region, country, state, district, forest.
   Examples: Western Ghats, Wayanad, Kerala, Tamil Nadu

4. BIOACTIVITY — biological or pharmacological activity.
   Examples: antimicrobial, antioxidant, anti-inflammatory, cytotoxic

5. DISEASE — named clinical/veterinary condition (NOT mechanisms like apoptosis or oxidative stress).
   Examples: malaria, diabetes, tuberculosis, cancer

RULES
- Solvents/pharmacological inducers are always CHEMICAL.
- Extract nested entities separately: "streptozotocin-induced diabetes" → CHEMICAL + DISEASE.
- Never include concentration values in spans: "Eugenol (72.4%)" → span is "Eugenol".
- linked_to is only for BIOACTIVITY → name of the performing CHEMICAL, or null.

OUTPUT SCHEMA — JSON array, each object:
{"span":"verbatim substring","type":"CHEMICAL|SPECIES|LOCATION|BIOACTIVITY|DISEASE","start":0,"end":7,"name_type":"scientific|null","linked_to":"chemical span|null"}

name_type is "scientific" for SPECIES, null for all others.
linked_to is only populated for BIOACTIVITY when the performing chemical is named in the text.

Return ONLY the JSON array."""


LABEL_DEFINITIONS = {
    # Dictionary-only types (handled by dictionary, no LLM needed)
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


