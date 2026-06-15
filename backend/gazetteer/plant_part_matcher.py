"""
Dictionary-based Plant Part Matcher using spaCy PhraseMatcher.

Fast dictionary lookup for plant parts - no heavy models needed.
Uses spaCy blank model with PhraseMatcher for efficient matching.
"""

import csv
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional, Any
import spacy
from spacy.matcher import PhraseMatcher
import logging

logger = logging.getLogger(__name__)

# Paths - FIXED
BASE_DIR = Path(__file__).parent.parent  # backend/
DATA_DIR = BASE_DIR / "gazetteer" / "data"
BUILD_DIR = BASE_DIR / "gazetteer" / "build"
CACHE_FILE = BUILD_DIR / "plant_part_cache.pkl"

ENTITY_TYPE = "PLANT PART"


class PlantPartMatcher:
    """Fast dictionary matcher for plant parts using spaCy PhraseMatcher."""

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

                    # Store canonical map for lookup
                    self.canonical_map = canonical_map

                    logger.info(
                        f"[PlantPartMatcher] Loaded {len(patterns)} patterns from cache"
                    )
                    return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")

        # Build from CSV
        self._build_from_csv()

    def _build_from_csv(self):
        """Build matcher from CSV file with aliases support.

        CSV format (new):
        term,aliases
        leaf,leaves
        rhizome,rhizomes|horizontal stem

        Aliases separated by | in second column.
        """
        csv_path = DATA_DIR / "plant_part.csv"

        terms = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])

            # Check if using new format (term,aliases|synonyms)
            second_header = headers[1].lower() if headers and len(headers) > 1 else ""
            has_aliases = "alias" in second_header or "synonym" in second_header

            for row in reader:
                if not row or not row[0].strip():
                    continue

                # Skip comments
                if row[0].strip().startswith("#"):
                    continue

                # Primary term
                terms.append(row[0].strip().lower())

                # Aliases from second column
                if has_aliases and len(row) > 1 and row[1].strip():
                    for alias in row[1].strip().split("|"):
                        if alias.strip():
                            terms.append(alias.strip().lower())
                elif not has_aliases:
                    # Old format: just term in column 1
                    terms.append(row[0].strip().lower())

        # Deduplicate
        terms = list(set(terms))

        # Build canonical map: alias -> primary term (singular form)
        canonical_map = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            for row in reader:
                if not row or not row[0].strip() or row[0].strip().startswith("#"):
                    continue
                primary = row[0].strip().lower()
                canonical_map[primary] = primary  # primary -> itself
                # Map all aliases to primary
                if len(row) > 1 and row[1].strip():
                    for alias in row[1].strip().split("|"):
                        if alias.strip():
                            canonical_map[alias.strip().lower()] = primary

        self.canonical_map = canonical_map

        # Create matcher
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(t) for t in terms if t]
        self.matcher.add(ENTITY_TYPE, patterns)

        logger.info(
            f"[PlantPartMatcher] Built {len(patterns)} patterns from CSV (with aliases)"
        )

    def get_aliases_for_canonical(self, canonical: str) -> List[str]:
        """Get all aliases (variations) for a canonical term."""
        aliases = []
        for variation, can in self.canonical_map.items():
            if can == canonical:
                aliases.append(variation)
        return aliases

    def match(self, text: str) -> List[Dict[str, Any]]:
        """Find all plant parts in text.

        Args:
            text: Input text to match

        Returns:
            List of entity dicts with fields:
            - span: matched text
            - type: entity type (PLANT PART)
            - start: start position
            - end: end position
            - name_type: None for plant parts
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

            # Use canonical term if available, otherwise use matched text
            canonical = self.canonical_map.get(span.text.lower(), span.text)

            # Get all aliases for this canonical term
            aliases = self.get_aliases_for_canonical(canonical)

            entities.append(
                {
                    "span": span.text,
                    "canonical": canonical,  # Normalized form for display
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
_matcher: Optional[PlantPartMatcher] = None


def get_matcher() -> PlantPartMatcher:
    """Get singleton matcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = PlantPartMatcher()
    return _matcher


def match_plant_parts(text: str) -> List[Dict[str, Any]]:
    """Convenience function to match plant parts in text.

    Args:
        text: Input text

    Returns:
        List of plant part entities
    """
    return get_matcher().match(text)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Test
    test_text = """
    The leaves and bark of Cinnamomum verum were collected from Kerala.
    Fresh rhizomes and roots were used for extraction.
    Essential oil from flower and stem showed antimicrobial activity.
    """

    entities = match_plant_parts(test_text)
    print(f"\nFound {len(entities)} plant parts:")
    for e in entities:
        print(f"  - '{e['span']}' ({e['start']}-{e['end']})")
