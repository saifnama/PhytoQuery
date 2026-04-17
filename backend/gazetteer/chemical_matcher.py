"""
Dictionary-based Chemical Matcher using spaCy PhraseMatcher.

Matches preferred chemical names and synonyms with high precision and enriches
chemical entities with metadata from the chemical gazetteer.
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import spacy
from spacy.matcher import PhraseMatcher

logger = logging.getLogger(__name__)

ENTITY_TYPE = "CHEMICAL"

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "gazetteer" / "data"
BUILD_DIR = BASE_DIR / "gazetteer" / "build"
DATA_FILE = DATA_DIR / "chemical.csv"
CACHE_FILE = BUILD_DIR / "chemical_cache.pkl"
CACHE_VERSION = "v4"


def _normalize_alias(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower().strip())


def _alias_variants(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    variants = [text]
    hyphen_to_space = re.sub(r"[-_/]+", " ", text)
    if hyphen_to_space != text:
        variants.append(re.sub(r"\s+", " ", hyphen_to_space).strip())

    punctuation_removed = re.sub(r"[^A-Za-z0-9\s]", "", text)
    punctuation_removed = re.sub(r"\s+", " ", punctuation_removed).strip()
    if punctuation_removed and punctuation_removed not in variants:
        variants.append(punctuation_removed)

    return list(dict.fromkeys(v for v in variants if v))


def _primary_name(row: Dict[str, Any]) -> str:
    return ((row.get("term") or row.get("preferred_name") or "")).strip()


def build_chemical_cache_data(
    rows: List[Dict[str, Any]],
    log_collision: Optional[Callable[[str, str, str], None]] = None,
) -> Dict[str, Any]:
    terms: set[str] = set()
    metadata_map: Dict[str, Dict[str, Any]] = {}
    canonical_map: Dict[str, str] = {}
    aliases_by_canonical: Dict[str, List[str]] = {}

    alias_candidates: Dict[str, List[Dict[str, Any]]] = {}

    prepared_rows: List[Dict[str, Any]] = []

    for row in rows:
        preferred_name = _primary_name(row)
        synonyms_raw = (row.get("synonyms") or "").strip()
        inchikey = (row.get("inchikey") or "").strip()
        smiles = (row.get("smiles") or "").strip()
        molecular_formula = (row.get("molecular_formula") or "").strip()
        source_db = (row.get("source_db") or "").strip()
        source_url = (row.get("source_url") or "").strip()

        canonical = preferred_name
        if not canonical:
            continue

        primary_aliases = _alias_variants(preferred_name)
        synonyms = [s.strip() for s in synonyms_raw.split("|") if s.strip()]
        synonym_aliases: List[str] = []
        for alias in synonyms:
            synonym_aliases.extend(_alias_variants(alias))
        synonym_aliases = [
            alias for alias in list(dict.fromkeys(synonym_aliases)) if alias not in primary_aliases
        ]
        aliases = primary_aliases + synonym_aliases

        prepared_rows.append(
            {
                "canonical": canonical,
                "preferred_name": preferred_name,
                "aliases": aliases,
                "inchikey": inchikey,
                "smiles": smiles,
                "molecular_formula": molecular_formula,
                "source_db": source_db,
                "source_url": source_url,
            }
        )

        for index, alias in enumerate(primary_aliases):
            alias_candidates.setdefault(alias.lower(), []).append(
                {
                    "canonical": canonical,
                    "alias": alias,
                    "priority": 2 if index == 0 else 1,
                }
            )
        for alias in synonym_aliases:
            alias_candidates.setdefault(alias.lower(), []).append(
                {"canonical": canonical, "alias": alias, "priority": 0}
            )

    alias_owner: Dict[str, str] = {}
    invalid_alias_keys: set[str] = set()

    for alias_key, candidates in alias_candidates.items():
        distinct_candidates: List[Dict[str, Any]] = []
        seen_pairs = set()
        for candidate in candidates:
            pair = (candidate["canonical"], candidate["priority"])
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            distinct_candidates.append(candidate)

        max_priority = max(candidate["priority"] for candidate in distinct_candidates)
        highest_priority = [
            candidate
            for candidate in distinct_candidates
            if candidate["priority"] == max_priority
        ]

        if len(highest_priority) == 1:
            alias_owner[alias_key] = highest_priority[0]["canonical"]
            continue

        invalid_alias_keys.add(alias_key)
        winner = highest_priority[0]
        for loser in highest_priority[1:]:
            if log_collision is not None:
                log_collision(winner["alias"], winner["canonical"], loser["canonical"])

    for prepared in prepared_rows:
        canonical = prepared["canonical"]
        accepted_aliases = [
            alias
            for alias in prepared["aliases"]
            if alias.lower() not in invalid_alias_keys
            and alias_owner.get(alias.lower()) == canonical
        ]

        if not accepted_aliases:
            continue

        aliases_by_canonical[canonical] = accepted_aliases
        metadata = {
            "canonical": canonical,
            "preferred_name": prepared["preferred_name"],
            "aliases": accepted_aliases,
            "inchikey": prepared["inchikey"],
            "smiles": prepared["smiles"],
            "molecular_formula": prepared["molecular_formula"],
            "source_db": prepared["source_db"],
            "source_url": prepared["source_url"],
        }

        for alias in accepted_aliases:
            alias_key = alias.lower()
            terms.add(alias_key)
            canonical_map[alias_key] = canonical
            metadata_map[alias_key] = metadata

    return {
        "terms": sorted(terms),
        "canonical_map": canonical_map,
        "metadata_map": metadata_map,
        "aliases_by_canonical": aliases_by_canonical,
    }


class ChemicalMatcher:
    """High-precision chemical matcher backed by gazetteer data."""

    def __init__(self, nlp: Any = None):
        self.nlp = nlp or spacy.blank("en")
        self.matcher: Optional[PhraseMatcher] = None
        self.canonical_map: Dict[str, str] = {}
        self.metadata_map: Dict[str, Dict[str, Any]] = {}
        self.aliases_by_canonical: Dict[str, List[str]] = {}
        self._load_or_build()

    def _load_or_build(self) -> None:
        if CACHE_FILE.exists() and DATA_FILE.exists():
            try:
                if CACHE_FILE.stat().st_mtime < DATA_FILE.stat().st_mtime:
                    logger.info("[ChemicalMatcher] Cache is stale; rebuilding from CSV")
                    self._build_from_csv()
                    return

                with open(CACHE_FILE, "rb") as f:
                    cache = pickle.load(f)

                if ENTITY_TYPE in cache:
                    data = cache[ENTITY_TYPE]
                    if data.get("cache_version") != CACHE_VERSION:
                        logger.info(
                            "[ChemicalMatcher] Cache version mismatch; rebuilding from CSV"
                        )
                        self._build_from_csv()
                        return
                    if not data.get("metadata_map") or not data.get("canonical_map"):
                        logger.info(
                            "[ChemicalMatcher] Cache schema is incomplete; rebuilding from CSV"
                        )
                        self._build_from_csv()
                        return

                    terms = data["terms"]
                    self.canonical_map = data.get("canonical_map", {})
                    self.metadata_map = data.get("metadata_map", {})
                    self.aliases_by_canonical = data.get("aliases_by_canonical", {})

                    matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
                    matcher.add(ENTITY_TYPE, [self.nlp.make_doc(t) for t in terms])
                    self.matcher = matcher
                    logger.info(
                        "[ChemicalMatcher] Loaded %s patterns from cache", len(terms)
                    )
                    return
            except Exception as e:
                logger.warning("Chemical cache load failed: %s", e)

        self._build_from_csv()

    def _build_from_csv(self) -> None:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"Chemical gazetteer not found: {DATA_FILE}")

        BUILD_DIR.mkdir(parents=True, exist_ok=True)

        with open(DATA_FILE, "r", encoding="utf-8", errors="ignore", newline="") as f:
            import csv

            rows = list(csv.DictReader(f))

        cache_data = build_chemical_cache_data(
            rows,
            log_collision=lambda alias, existing, canonical: logger.warning(
                "[ChemicalMatcher] Duplicate alias collision for '%s': '%s' vs '%s'. Keeping neither.",
                alias,
                existing,
                canonical,
            ),
        )

        self.canonical_map = cache_data["canonical_map"]
        self.metadata_map = cache_data["metadata_map"]
        self.aliases_by_canonical = cache_data["aliases_by_canonical"]

        patterns = [self.nlp.make_doc(term) for term in cache_data["terms"]]
        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        matcher.add(ENTITY_TYPE, patterns)
        self.matcher = matcher

        cache = {
            ENTITY_TYPE: {
                **cache_data,
                "cache_version": CACHE_VERSION,
                "source_mtime": DATA_FILE.stat().st_mtime,
            }
        }
        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)

        logger.info("[ChemicalMatcher] Built %s chemical patterns", len(patterns))

    def lookup(self, text: str) -> Optional[Dict[str, Any]]:
        stripped = text.lower().strip()
        lookup = self.metadata_map.get(stripped)
        if not lookup:
            lookup = self.metadata_map.get(_normalize_alias(text))
        if not lookup:
            return None

        return {
            "text": text,
            "span": text,
            "canonical": lookup.get("preferred_name")
            or lookup.get("canonical")
            or text,
            "preferred_name": lookup.get("preferred_name")
            or lookup.get("canonical")
            or text,
            "aliases": lookup.get("aliases", []),
            "type": ENTITY_TYPE,
            "label": ENTITY_TYPE,
            "score": 1.0,
            "name_type": None,
            "linked_to": None,
            "inchikey": lookup.get("inchikey"),
            "smiles": lookup.get("smiles"),
            "molecular_formula": lookup.get("molecular_formula"),
            "source_db": lookup.get("source_db"),
            "source_url": lookup.get("source_url"),
        }

    def match(self, text: str) -> List[Dict[str, Any]]:
        if not text or not text.strip() or self.matcher is None:
            return []

        doc = self.nlp(text)
        entities: List[Dict[str, Any]] = []
        seen = set()

        for _, start, end in self.matcher(doc):
            span = doc[start:end]
            key = (span.start_char, span.end_char)
            if key in seen:
                continue
            seen.add(key)

            enriched = self.lookup(span.text)
            if not enriched:
                continue

            enriched.update(
                {
                    "text": span.text,
                    "span": span.text,
                    "start": span.start_char,
                    "end": span.end_char,
                }
            )
            entities.append(enriched)

        return entities


_matcher: Optional[ChemicalMatcher] = None


def get_matcher() -> ChemicalMatcher:
    global _matcher
    if _matcher is None:
        _matcher = ChemicalMatcher()
    return _matcher


def match_chemicals(text: str) -> List[Dict[str, Any]]:
    return get_matcher().match(text)
