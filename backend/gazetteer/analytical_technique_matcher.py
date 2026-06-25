"""
Dictionary-based Analytical Technique Matcher using spaCy PhraseMatcher.

Fast dictionary lookup for analytical/identification techniques - no heavy models needed.
Uses spaCy blank model with PhraseMatcher for efficient matching.
"""

import csv
import pickle
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
import spacy
from spacy.matcher import PhraseMatcher
import logging

logger = logging.getLogger(__name__)

# Entity type label
ENTITY_TYPE = "ANALYTICAL TECHNIQUE"

# Paths
BASE_DIR = Path(__file__).parent.parent  # backend/
DATA_DIR = BASE_DIR / "gazetteer" / "data"


def _normalize_dashes(text: str) -> str:
    return re.sub(r'[\u2013\u2014\u2015\u2212]', '-', text)
BUILD_DIR = BASE_DIR / "gazetteer" / "build"
CACHE_FILE = BUILD_DIR / "analytical_technique_cache.pkl"


class AnalyticalTechniqueMatcher:
    """Fast dictionary matcher for analytical techniques using spaCy PhraseMatcher."""

    def __init__(self, nlp: Any = None):
        """Initialize matcher."""
        self.nlp = nlp or spacy.blank("en")
        self.matcher = None
        self.canonical_map = {}  # alias -> canonical term mapping
        self._load_or_build()

    def _load_or_build(self):
        """Load from cache or build new."""
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "rb") as f:
                    cache = pickle.load(f)

                if ENTITY_TYPE in cache:
                    data = cache[ENTITY_TYPE]
                    terms = data["terms"]
                    canonical_map = data.get("canonical_map", {})

                    # Create patterns
                    patterns = [self.nlp.make_doc(t) for t in terms]
                    self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
                    self.matcher.add(ENTITY_TYPE, patterns)

                    self.canonical_map = canonical_map

                    logger.info(
                        f"[AnalyticalTechniqueMatcher] Loaded {len(patterns)} patterns from cache"
                    )
                    return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")

        # Build from CSV
        self._build_from_csv()

    def _build_from_csv(self):
        """Build matcher from CSV file with aliases support.

        CSV format:
        term,aliases
        GC-MS,gcms|gc ms|GC; GC-MS
        """
        csv_path = DATA_DIR / "analytical_technique.csv"

        enc = "utf-8"
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                f.read()
        except UnicodeDecodeError:
            enc = "cp1252"
        terms = []
        with open(csv_path, "r", encoding=enc) as f:
            reader = csv.reader(f)
            headers = next(reader, [])

            # Check for synonyms column (term,synonyms format)
            has_synonyms = headers and len(headers) > 1 and "synonym" in headers[1].lower()

            for row in reader:
                if not row or not row[0].strip():
                    continue

                # Skip comments
                if row[0].strip().startswith("#"):
                    continue

                # Primary term
                terms.append(row[0].strip().lower())

                # Synonyms from second column
                if has_synonyms and len(row) > 1 and row[1].strip():
                    for synonym in row[1].strip().split("|"):
                        if synonym.strip():
                            terms.append(synonym.strip().lower())

        # Deduplicate
        terms = list(set(terms))

        # Build canonical map
        canonical_map = {}
        with open(csv_path, "r", encoding=enc) as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            for row in reader:
                if not row or not row[0].strip() or row[0].strip().startswith("#"):
                    continue
                primary = row[0].strip().lower()
                canonical_map[primary] = primary
                if len(row) > 1 and row[1].strip():
                    for alias in row[1].strip().split("|"):
                        if alias.strip():
                            canonical_map[alias.strip().lower()] = primary

        self.canonical_map = canonical_map

        # Create matcher — normalize dashes so en-dash/em-dash match hyphen
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(_normalize_dashes(t)) for t in terms if t]
        self.matcher.add(ENTITY_TYPE, patterns)

        logger.info(
            f"[AnalyticalTechniqueMatcher] Built {len(patterns)} patterns from CSV (with aliases)"
        )

    def get_aliases_for_canonical(self, canonical: str) -> List[str]:
        """Get all aliases (variations) for a canonical term."""
        aliases = []
        for variation, can in self.canonical_map.items():
            if can == canonical:
                aliases.append(variation)
        return aliases

    def match(self, text: str) -> List[Dict[str, Any]]:
        """Find all isolation methods in text.

        Args:
            text: Input text to match

        Returns:
            List of entity dicts with fields:
            - span: matched text
            - type: entity type (ISOLATION METHOD)
            - start: start position
            - end: end position
            - name_type: None
            - linked_to: None
            - label: for compatibility
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(_normalize_dashes(text))
        entities = []
        seen = set()

        for match_id, start, end in self.matcher(doc):
            span = doc[start:end]
            key = (span.start_char, span.end_char)

            if key in seen:
                continue
            seen.add(key)

            # Use canonical term if available (use lowercase for lookup)
            canonical = self.canonical_map.get(span.text.lower(), span.text)

            # Get all aliases for this canonical term
            aliases = self.get_aliases_for_canonical(canonical)

            entities.append(
                {
                    "span": span.text,
                    "canonical": canonical,
                    "aliases": aliases,
                    "type": ENTITY_TYPE,
                    "start": span.start_char,
                    "end": span.end_char,
                    "name_type": None,
                    "linked_to": None,
                    "label": ENTITY_TYPE,
                    "score": 1.0,
                }
            )

        # Remove shorter entities nested inside longer ones at the same position
        entities.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
        deduped = []
        for ent in entities:
            if not any(ent["start"] >= other["start"] and ent["end"] <= other["end"]
                       and (ent["end"] - ent["start"]) < (other["end"] - other["start"])
                       for other in deduped):
                deduped.append(ent)

        return deduped


# Singleton instance
_matcher: Optional[AnalyticalTechniqueMatcher] = None


def get_matcher() -> AnalyticalTechniqueMatcher:
    """Get singleton matcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = AnalyticalTechniqueMatcher()
    return _matcher


def match_analytical_techniques(text: str) -> List[Dict[str, Any]]:
    """Convenience function to match analytical techniques in text.

    Args:
        text: Input text

    Returns:
        List of analytical technique entities
    """
    return get_matcher().match(text)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Test
    test_text = """
    The essential oil was analyzed by GC-MS and GC-FID.
    NMR spectroscopy was used for structure elucidation.
    HPLC was used for compound isolation.
    TLC was performed for preliminary screening.
    """

    entities = match_analytical_techniques(test_text)
    print(f"\nFound {len(entities)} analytical techniques:")
    for e in entities:
        print(f"  - '{e['span']}' ({e['start']}-{e['end']})")
