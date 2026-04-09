#!/usr/bin/env python3
"""
Build script for dictionary-based NER matcher.

Compiles gazetteer CSV files into binary matcher for fast loading.
"""

import csv
import os
import pickle
import time
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "gazetteer" / "data"
BUILD_DIR = BASE_DIR / "gazetteer" / "build"

# Entity types mapping
CATEGORIES = {
    "PLANT PART": "plant_part.csv",
    "ANALYTICAL TECHNIQUE": "analytical_technique.csv",
    "EXTRACTION METHOD": "extraction_method.csv",
    "DEVELOPMENT STAGE": "development_stage.csv",
    "SEASON": "season.csv",
}


def build_matcher(category: str, csv_file: Path) -> dict:
    """Build matcher for a category."""
    print(f"\n[{category}] Loading terms...")

    terms = []
    canonical_map = {}

    with open(csv_file, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        headers = next(reader)  # Skip header

        # Check for aliases column
        has_aliases = headers and len(headers) > 1 and "alias" in headers[1].lower()

        for row in reader:
            if not row or not row[0].strip():
                continue
            # Skip comments
            if row[0].strip().startswith("#"):
                continue

            # Primary term
            term = row[0].strip().lower()
            terms.append(term)
            canonical_map[term] = term  # primary -> itself

            # Aliases
            if has_aliases and len(row) > 1 and row[1].strip():
                for alias in row[1].strip().split("|"):
                    alias_clean = alias.strip().lower()
                    if alias_clean:
                        terms.append(alias_clean)
                        canonical_map[alias_clean] = term  # alias -> primary

    print(f"[{category}] {len(terms)} terms (with aliases)")
    return {
        "category": category,
        "terms": terms,
        "count": len(terms),
        "canonical_map": canonical_map,
    }

    print(f"[{category}] {len(terms)} terms (with aliases)")
    return {
        "category": category,
        "terms": terms,
        "count": len(terms),
        "canonical_map": canonical_map,
    }


def main():
    """Build all matchers."""
    start_time = time.time()

    print("=" * 60)
    print("Dictionary NER Matcher Builder")
    print("=" * 60)

    # Ensure build directory exists
    BUILD_DIR.mkdir(parents=True, exist_ok=True)

    # Build each category
    results = {}
    for category, filename in CATEGORIES.items():
        csv_path = DATA_DIR / filename
        if not csv_path.exists():
            print(f"[WARNING] {filename} not found at {csv_path}, skipping...")
            continue

        result = build_matcher(category, csv_path)
        results[category] = result

    # Save each category to its own cache file
    for category, result in results.items():
        # Convert category name to filename: "PLANT PART" -> "plant_part_cache.pkl"
        category_slug = category.lower().replace(" ", "_")
        cache_file = BUILD_DIR / f"{category_slug}_cache.pkl"

        cache = {category: result}
        with open(cache_file, "wb") as f:
            pickle.dump(cache, f)

        print(f"[{category}] Cache saved to: {cache_file}")

    elapsed = time.time() - start_time
    total_terms = sum(r["count"] for r in results.values())

    print("=" * 60)
    print(f"Done. {total_terms} terms compiled")
    print(f"Cache saved to: {cache_file}")
    print(f"Time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
