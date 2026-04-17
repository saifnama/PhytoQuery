"""
Dictionary-based Bioactivity Matcher using spaCy PhraseMatcher.

Fast dictionary lookup for bioactivity/pharmacological activity terms - no heavy models needed.
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
ENTITY_TYPE = "BIOACTIVITY"

# Paths
BASE_DIR = Path(__file__).parent.parent  # backend/
DATA_DIR = BASE_DIR / "gazetteer" / "data"
BUILD_DIR = BASE_DIR / "gazetteer" / "build"
CACHE_FILE = BUILD_DIR / "bioactivity_cache.pkl"


class BioactivityMatcher:
    """Fast dictionary matcher for bioactivity terms using spaCy PhraseMatcher."""

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
                        f"[BioactivityMatcher] Loaded {len(patterns)} patterns from cache"
                    )
                    return
            except Exception as e:
                logger.warning(f"Cache load failed: {e}")

        # Build from CSV
        self._build_from_csv()

    def _build_from_csv(self):
        """Build matcher from CSV file with aliases support.

        CSV format:
        term,synonyms
        antimicrobial,antibacterial|antimicrobial
        antioxidant,free radical scavenger
        """
        csv_path = DATA_DIR / "bioactivity.csv"

        terms = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])

            # Check if second column contains synonyms
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

        # Build canonical map: alias -> primary term
        canonical_map = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            for row in reader:
                if not row or not row[0].strip() or row[0].strip().startswith("#"):
                    continue
                primary = row[0].strip().lower()
                canonical_map[primary] = primary  # primary -> itself
                # Map all synonyms to primary
                if len(row) > 1 and row[1].strip():
                    for synonym in row[1].strip().split("|"):
                        if synonym.strip():
                            canonical_map[synonym.strip().lower()] = primary

        self.canonical_map = canonical_map

        # Create matcher
        self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        patterns = [self.nlp.make_doc(t) for t in terms if t]
        self.matcher.add(ENTITY_TYPE, patterns)

        logger.info(
            f"[BioactivityMatcher] Built {len(patterns)} patterns from CSV (with synonyms)"
        )

    def get_synonyms_for_canonical(self, canonical: str) -> List[str]:
        """Get all synonyms (variations) for a canonical term."""
        synonyms = []
        for variation, can in self.canonical_map.items():
            if can == canonical:
                synonyms.append(variation)
        return synonyms

    def match(self, text: str) -> List[Dict[str, Any]]:
        """Find all bioactivity terms in text.

        Args:
            text: Input text to match

        Returns:
            List of entity dicts with fields:
            - span: matched text
            - type: entity type (BIOACTIVITY)
            - start: start position
            - end: end position
            - name_type: None
            - linked_to: None
            - canonical: canonical term
            - synonyms: all variations
            - score: 1.0 (dictionary matches are 100% confident)
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

            # Use canonical term if available
            canonical = self.canonical_map.get(span.text.lower(), span.text)

            # Get all synonyms for this canonical term
            synonyms = self.get_synonyms_for_canonical(canonical)

            entities.append(
                {
                    "span": span.text,
                    "canonical": canonical,
                    "synonyms": synonyms,
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
_matcher: Optional[BioactivityMatcher] = None


def get_matcher() -> BioactivityMatcher:
    """Get singleton matcher instance."""
    global _matcher
    if _matcher is None:
        _matcher = BioactivityMatcher()
    return _matcher


def match_bioactivities(text: str) -> List[Dict[str, Any]]:
    """Convenience function to match bioactivity terms in text.

    Args:
        text: Input text

    Returns:
        List of bioactivity entities
    """
    return get_matcher().match(text)


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)

    # Test
    test_text = """
    The extract showed antimicrobial and antioxidant activities.
    It exhibited antifungal properties and anti-inflammatory effects.
    The compound has cytotoxic activity against cancer cells.
    """

    entities = match_bioactivities(test_text)
    print(f"\nFound {len(entities)} bioactivity terms:")
    for e in entities:
        print(f"  - '{e['span']}' ({e['start']}-{e['end']})")