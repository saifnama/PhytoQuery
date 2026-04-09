"""
Dictionary-based Extraction Method Matcher using spaCy PhraseMatcher.

Fast dictionary lookup for plant extraction methods - no heavy models needed.
Uses spaCy blank model with PhraseMatcher for efficient matching.
"""

import csv
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import spacy
from spacy.matcher import PhraseMatcher
import logging

logger = logging.getLogger(__name__)

# Entity type label
ENTITY_TYPE = "EXTRACTION METHOD"

# Paths
BASE_DIR = Path(__file__).parent.parent  # backend/
DATA_DIR = BASE_DIR / "gazetteer" / "data"
BUILD_DIR = BASE_DIR / "gazetteer" / "build"
CACHE_FILE = BUILD_DIR / "extraction_method_cache.pkl"


class ExtractionMethodMatcher:
    """Fast dictionary matcher for extraction methods using spaCy PhraseMatcher."""

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
                        f"[ExtractionMethodMatcher] Loaded {len(patterns)} patterns from cache"
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
        soxhlet extraction,soxhlet|hot continuous extraction
        maceration,macerated
        """
        csv_path = DATA_DIR / "extraction_method.csv"

        terms = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])

            for row in reader:
                if not row or not row[0].strip():
                    continue

                term = row[0].strip()

                # Skip comments
                if term.startswith("#"):
                    continue

                terms.append(term.lower())

                # Build canonical map: each variation maps to the main term
                # The main term itself maps to itself
                self.canonical_map[term.lower()] = term

                # If there are aliases, map them to the main term
                if len(row) > 1 and row[1].strip():
                    aliases = row[1].strip()
                    for alias in aliases.split("|"):
                        alias = alias.strip()
                        if alias:
                            terms.append(alias.lower())
                            self.canonical_map[alias.lower()] = term

        terms = list(dict.fromkeys(terms))

        # Create PhraseMatcher patterns
        patterns = [self.nlp.make_doc(t) for t in terms]
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        self.matcher.add(ENTITY_TYPE, patterns)

        # Save to cache
        self._save_cache(terms)

        logger.info(
            f"[ExtractionMethodMatcher] Built {len(patterns)} patterns from CSV (with aliases)"
        )

    def _save_cache(self, terms: List[str]):
        """Save matcher to cache file."""
        BUILD_DIR.mkdir(parents=True, exist_ok=True)

        cache = {
            ENTITY_TYPE: {
                "terms": terms,
                "canonical_map": self.canonical_map,
            }
        }

        with open(CACHE_FILE, "wb") as f:
            pickle.dump(cache, f)

        logger.info(f"[ExtractionMethodMatcher] Cache saved to {CACHE_FILE}")

    def get_aliases_for_canonical(self, canonical: str) -> List[str]:
        """Get all aliases (variations) for a canonical term."""
        aliases = []
        for variation, can in self.canonical_map.items():
            if can == canonical:
                aliases.append(variation)
        return aliases

    def match(self, text: str) -> List[Dict[str, Any]]:
        """Find all extraction methods in text.

        Args:
            text: Input text to match

        Returns:
            List of entity dicts with fields:
            - span: matched text
            - type: entity type (EXTRACTION METHOD)
            - start: start position
            - end: end position
            - name_type: None
            - linked_to: None
            - label: for compatibility
        """
        if not text or not text.strip():
            return []

        doc = self.nlp(text)
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
                    "aliases": aliases,  # All variations for counting
                    "type": ENTITY_TYPE,
                    "start": span.start_char,
                    "end": span.end_char,
                    "name_type": None,
                    "linked_to": None,
                    "label": ENTITY_TYPE,
                    "score": 1.0,  # Dictionary matches are 100% confident
                }
            )

        return entities


# Singleton instance
_matcher: Optional[ExtractionMethodMatcher] = None


def get_matcher() -> ExtractionMethodMatcher:
    """Get singleton matcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = ExtractionMethodMatcher()
    return _matcher


def match_extraction_methods(text: str) -> List[Dict[str, Any]]:
    """Convenience function to match extraction methods in text.

    Args:
        text: Input text

    Returns:
        List of extraction method entities
    """
    return get_matcher().match(text)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Test
    test_text = """
    The plant material was extracted using soxhlet extraction with methanol.
    Maceration was performed for 24 hours.
    Ultrasound assisted extraction was also used.
    Supercritical fluid extraction gave better yields.
    """

    entities = match_extraction_methods(test_text)
    for e in entities:
        print(f"  {e['span']} -> {e['canonical']}")
