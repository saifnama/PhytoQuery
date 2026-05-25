"""
SQLite migration v2: OLD 2-table layout -> 15-table per-type schema.

OLD shape (what's in db_data/phytoquery.sqlite right now):
    papers(id, doi, title, journal, year, is_open_access, entity_count)
    paper_entities(id, paper_id, label, canonical_text, mention_count, metadata_json)

NEW shape (per-type, one table per entity category + one junction per category):
    papers                                              ← unchanged + created_at/updated_at
    chemicals             + paper_chemicals             ← rich chemistry metadata
    species               + paper_species               ← rich taxonomy metadata
    locations             + paper_locations             ← country/state preserved
    plant_parts           + paper_plant_parts           ← bare entity table
    analytical_techniques + paper_analytical_techniques ← bare entity table
    extraction_methods    + paper_extraction_methods    ← bare entity table
    development_stages    + paper_development_stages    ← bare entity table

What the migration does:
    1. Back up the DB with a timestamp.
    2. Auto-clean any leftover empty per-type tables from a previous failed run.
    3. Refuse if the migration already succeeded (chemicals table has rows).
    4. Create every new table inside a single transaction.
    5. Group old rows by (label, lowercase canonical_text).
    6. For each group:
         - Pick the most-common original casing as display_text.
         - Merge metadata from SQLite (LOCATION country/state, SPECIES family/
           common_name) plus an optional gazetteer lookup for CHEMICAL/SPECIES.
         - SQLite values OVERRIDE gazetteer on key overlap.
         - INSERT into the per-type entity table.
    7. Build a map (label, canonical_lower) -> new entity id per type.
    8. Walk every old paper_entities row, look up its new (table, entity_id),
       and INSERT into the matching junction. Re-imports of casing-drift
       siblings sum into the same junction via ON CONFLICT DO UPDATE.
    9. Recompute papers.entity_count across all 7 junctions.
   10. Validate (papers count, sum(mention_count), distinct entities, 5-paper
       spot checks). ROLLBACK on any failure.
   11. DROP old paper_entities, COMMIT.
   12. VACUUM outside the transaction.

Run from the repo root (use the project venv that has spaCy):
    pq/Scripts/python.exe -m backend.db.migrate_schema_v2 --dry-run
    pq/Scripts/python.exe -m backend.db.migrate_schema_v2

A timestamped backup ``phytoquery.sqlite.bak.YYYYMMDD_HHMMSS`` is taken
BEFORE any writes. If anything fails, ROLLBACK; the original DB is intact
and the backup is your safety net.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "db_data" / "phytoquery.sqlite"

# Maps the old paper_entities.label string to:
#   (entity_table_name, junction_table_name, junction_entity_column)
LABEL_TO_TABLE: Dict[str, Tuple[str, str, str]] = {
    "CHEMICAL":             ("chemicals",             "paper_chemicals",             "chemical_id"),
    "SPECIES":              ("species",               "paper_species",               "species_id"),
    "LOCATION":             ("locations",             "paper_locations",             "location_id"),
    "PLANT PART":           ("plant_parts",           "paper_plant_parts",           "plant_part_id"),
    "ANALYTICAL TECHNIQUE": ("analytical_techniques", "paper_analytical_techniques", "analytical_technique_id"),
    "EXTRACTION METHOD":    ("extraction_methods",    "paper_extraction_methods",    "extraction_method_id"),
    "DEVELOPMENT STAGE":    ("development_stages",    "paper_development_stages",    "development_stage_id"),
}

# Columns (in INSERT order) for each per-type entity table.
ENTITY_INSERT_COLUMNS: Dict[str, List[str]] = {
    "CHEMICAL": [
        "canonical_text", "display_text",
        "inchikey", "smiles", "molecular_formula", "preferred_name",
        "source_db", "source_url",
        "aliases_json",
    ],
    "SPECIES": [
        "canonical_text", "display_text",
        "accepted_scientific_name", "common_name", "family", "taxon_id", "name_type",
        "source_db", "source_url",
        "aliases_json",
    ],
    "LOCATION": [
        "canonical_text", "display_text",
        "country", "state",
        "aliases_json",
    ],
    "PLANT PART":           ["canonical_text", "display_text", "aliases_json"],
    "ANALYTICAL TECHNIQUE": ["canonical_text", "display_text", "aliases_json"],
    "EXTRACTION METHOD":    ["canonical_text", "display_text", "aliases_json"],
    "DEVELOPMENT STAGE":    ["canonical_text", "display_text", "aliases_json"],
}

# Matcher-internal keys we never carry into the entity row.
MATCHER_INTERNAL_KEYS = {
    "text", "span", "canonical", "type", "label", "score", "linked_to",
    "match_status", "review_required", "scientific_name_verified",
}


# ─────────────────────────────────────────────────────────────────────────────
# DDL — one statement per execute() to keep transactional integrity
# ─────────────────────────────────────────────────────────────────────────────

DDL_CHEMICALS = """
CREATE TABLE chemicals (
    id INTEGER PRIMARY KEY,
    canonical_text TEXT NOT NULL,
    display_text TEXT NOT NULL,
    inchikey TEXT,
    smiles TEXT,
    molecular_formula TEXT,
    preferred_name TEXT,
    source_db TEXT,
    source_url TEXT,
    aliases_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    CONSTRAINT uq_chemicals_canonical UNIQUE (canonical_text)
);
"""
DDL_CHEMICALS_INDEXES = [
    "CREATE INDEX idx_chemicals_inchikey ON chemicals(inchikey);",
    "CREATE INDEX idx_chemicals_molecular_formula ON chemicals(molecular_formula);",
]

DDL_SPECIES = """
CREATE TABLE species (
    id INTEGER PRIMARY KEY,
    canonical_text TEXT NOT NULL,
    display_text TEXT NOT NULL,
    accepted_scientific_name TEXT,
    common_name TEXT,
    family TEXT,
    taxon_id TEXT,
    name_type TEXT,
    source_db TEXT,
    source_url TEXT,
    aliases_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    CONSTRAINT uq_species_canonical UNIQUE (canonical_text)
);
"""
DDL_SPECIES_INDEXES = [
    "CREATE INDEX idx_species_family ON species(family);",
    "CREATE INDEX idx_species_taxon_id ON species(taxon_id);",
]

DDL_LOCATIONS = """
CREATE TABLE locations (
    id INTEGER PRIMARY KEY,
    canonical_text TEXT NOT NULL,
    display_text TEXT NOT NULL,
    country TEXT,
    state TEXT,
    aliases_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    CONSTRAINT uq_locations_canonical UNIQUE (canonical_text)
);
"""
DDL_LOCATIONS_INDEXES = [
    "CREATE INDEX idx_locations_country ON locations(country);",
    "CREATE INDEX idx_locations_state ON locations(state);",
]

# Template for the 4 "simple" entity tables.
DDL_SIMPLE_TEMPLATE = """
CREATE TABLE {table} (
    id INTEGER PRIMARY KEY,
    canonical_text TEXT NOT NULL,
    display_text TEXT NOT NULL,
    aliases_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    CONSTRAINT uq_{table}_canonical UNIQUE (canonical_text)
);
"""

# Junction template — composite PK + WITHOUT ROWID + cascade FKs.
DDL_JUNCTION_TEMPLATE = """
CREATE TABLE {junction} (
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    {entity_col} INTEGER NOT NULL REFERENCES {entity_table}(id) ON DELETE CASCADE,
    mention_count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (paper_id, {entity_col})
) WITHOUT ROWID;
"""

DDL_JUNCTION_INDEX_TEMPLATE = "CREATE INDEX idx_{junction}_{entity_col} ON {junction}({entity_col});"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_metadata(raw: Any) -> Dict[str, Any]:
    """Parse a paper_entities.metadata_json value into a dict.

    Most rows store the literal string "null". Only SPECIES (family/
    common_name) and LOCATION (country/state) carry real JSON in your DB.
    Anything that isn't a dict becomes {}.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    s = raw.strip()
    if not s or s.lower() == "null":
        return {}
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _strip_matcher_internal(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop matcher-internal keys + None values + empty aliases from a lookup result."""
    if not d:
        return {}
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k in MATCHER_INTERNAL_KEYS:
            continue
        if v is None:
            continue
        if k == "aliases" and isinstance(v, list) and len(v) == 0:
            continue
        out[k] = v
    return out


def _merge_sqlite_meta(
    sqlite_meta: Dict[str, Any], row_meta: Dict[str, Any], label: str
) -> Dict[str, Any]:
    """Accumulate per-row metadata into a single dict for the entity group.

    LOCATION: union of (country, state) — first non-empty wins per key.
    SPECIES:  first non-empty (family, common_name) wins.
    Everything else: ignored — those labels have no curated metadata in
    your DB right now (metadata_json is "null" for all).
    """
    if label == "LOCATION":
        for k in ("country", "state"):
            v = row_meta.get(k)
            if v and not sqlite_meta.get(k):
                sqlite_meta[k] = v
    elif label == "SPECIES":
        for k in ("family", "common_name"):
            v = row_meta.get(k)
            if v and not sqlite_meta.get(k):
                sqlite_meta[k] = v
    return sqlite_meta


def _build_entity_params(
    label: str,
    canonical_lower: str,
    display_text: str,
    merged: Dict[str, Any],
) -> Tuple[Any, ...]:
    """Project the merged dict into the INSERT parameter tuple for this label."""
    columns = ENTITY_INSERT_COLUMNS[label]
    aliases = merged.get("aliases")
    aliases_json = (
        json.dumps(aliases)
        if isinstance(aliases, (list, tuple)) and aliases
        else None
    )

    values: List[Any] = []
    for col in columns:
        if col == "canonical_text":
            values.append(canonical_lower)
        elif col == "display_text":
            values.append(display_text)
        elif col == "aliases_json":
            values.append(aliases_json)
        else:
            v = merged.get(col)
            if v == "":
                v = None
            values.append(v)
    return tuple(values)


def _build_entity_insert_sql(label: str) -> str:
    """Build INSERT INTO <entity_table> (col1, col2, ...) VALUES (?, ?, ...)."""
    entity_table, _, _ = LABEL_TO_TABLE[label]
    columns = ENTITY_INSERT_COLUMNS[label]
    column_list = ", ".join(columns)
    placeholders = ", ".join(["?"] * len(columns))
    return f"INSERT INTO {entity_table} ({column_list}) VALUES ({placeholders});"


def _build_junction_insert_sql(label: str) -> str:
    """UPSERT into per-type junction; SUM mention_count on conflict (handles
    casing-drift siblings collapsing into one junction row)."""
    _, junction, entity_col = LABEL_TO_TABLE[label]
    return (
        f"INSERT INTO {junction} (paper_id, {entity_col}, mention_count) "
        "VALUES (?, ?, ?) "
        f"ON CONFLICT(paper_id, {entity_col}) DO UPDATE SET "
        f"mention_count = {junction}.mention_count + excluded.mention_count;"
    )


def _load_matchers():
    """Import and instantiate the two gazetteer matchers once.

    Requires spaCy in the active env. Run with the project venv's python:
        pq/Scripts/python.exe -m backend.db.migrate_schema_v2
    """
    from backend.gazetteer.chemical_matcher import get_matcher as get_chemical_matcher
    from backend.gazetteer.species_matcher import get_matcher as get_species_matcher

    print("  Loading chemical matcher ...")
    chem = get_chemical_matcher()
    print("  Loading species matcher ...")
    spec = get_species_matcher()
    return chem, spec


def _lookup_safe(matcher, display_text: str, canonical_lower: str) -> Dict[str, Any]:
    """Look up in the matcher, trying both display + canonical forms."""
    try:
        hit = matcher.lookup(display_text)
        if hit is None and display_text.lower() != canonical_lower:
            hit = matcher.lookup(canonical_lower)
        return _strip_matcher_internal(hit)
    except Exception as exc:  # noqa: BLE001 -- per-row resilience
        print(f"    [warn] matcher lookup failed for {display_text!r}: {exc}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# State detection + partial-cleanup
# ─────────────────────────────────────────────────────────────────────────────

NEW_ENTITY_TABLES = [
    "chemicals", "species", "locations", "plant_parts",
    "analytical_techniques", "extraction_methods", "development_stages",
]
NEW_JUNCTION_TABLES = [
    "paper_chemicals", "paper_species", "paper_locations", "paper_plant_parts",
    "paper_analytical_techniques", "paper_extraction_methods", "paper_development_stages",
]
ALL_NEW_TABLES = NEW_ENTITY_TABLES + NEW_JUNCTION_TABLES


def _detect_state(cur) -> Tuple[str, set]:
    """Return (state, existing_table_set).

    state ∈ {"fresh", "partial", "migrated"}.
    """
    rows = cur.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
    ).fetchall()
    tables = {row[0] for row in rows}

    has_new = any(t in tables for t in ALL_NEW_TABLES)
    has_old = "paper_entities" in tables

    if not has_new:
        return "fresh", tables

    # Any new tables present — check if migration finished or only started.
    if not has_old:
        # New tables exist AND old paper_entities is gone → finished migration.
        return "migrated", tables

    # New tables present BUT old paper_entities still here → partial state.
    return "partial", tables


def _cleanup_partial(cur, tables: set) -> None:
    """Drop any empty leftover NEW tables from a previous failed run.

    Junction tables go first (they have FKs to entity tables). Refuses to
    drop any table that has rows — that would be data loss.
    """
    print("  Detected partial state from a previous failed run.")
    for t in NEW_JUNCTION_TABLES + NEW_ENTITY_TABLES:
        if t in tables:
            cnt = cur.execute(f'SELECT COUNT(*) FROM "{t}";').fetchone()[0]
            if cnt != 0:
                raise RuntimeError(
                    f"Partial-state table {t!r} has {cnt} rows; refusing to "
                    "auto-clean. Investigate manually."
                )
            cur.execute(f"DROP TABLE {t};")
            print(f"  Cleaned dangling empty table: {t}")


# ─────────────────────────────────────────────────────────────────────────────
# Migration
# ─────────────────────────────────────────────────────────────────────────────

def migrate(db_path: Path, dry_run: bool) -> int:
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 2

    # ─── Stage 1: backup ────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.bak.{timestamp}")
    size_before = os.path.getsize(db_path)

    print("=" * 70)
    print("[Stage 1/7] Backup")
    print(f"  Source: {db_path}")
    print(f"  Backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    print(f"  Backup size: {os.path.getsize(backup_path):,} bytes")

    # ─── Stage 2: connect + state check + counter snapshot ─────────────
    conn = sqlite3.connect(str(db_path))
    # Autocommit mode so the explicit BEGIN/COMMIT/ROLLBACK we issue covers
    # ALL statements including DDL. Python's sqlite3 default isolation_level
    # ("" / deferred) auto-commits before DDL, which silently breaks rollback
    # for transactional DDL. With isolation_level=None and SQL-level
    # transaction control via cur.execute(), every statement (DDL + DML)
    # belongs to the manual transaction.
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=OFF;")

    state, tables = _detect_state(cur)
    if state == "migrated":
        print("Already migrated (new per-type tables present, no old paper_entities). Exiting.")
        conn.close()
        return 0

    # Snapshot counters for end-of-run validation.
    papers_before = cur.execute("SELECT COUNT(*) FROM papers;").fetchone()[0]
    pe_rows_before = cur.execute("SELECT COUNT(*) FROM paper_entities;").fetchone()[0]
    mention_sum_before = cur.execute(
        "SELECT COALESCE(SUM(mention_count), 0) FROM paper_entities;"
    ).fetchone()[0]
    print(f"  Papers: {papers_before:,}")
    print(f"  paper_entities rows: {pe_rows_before:,}")
    print(f"  Sum(mention_count) before: {mention_sum_before:,}")

    chem_total = chem_enriched = spec_total = spec_enriched = 0
    new_junction_total = 0

    try:
        # ─── Stage 3: schema creation ──────────────────────────────────
        print("=" * 70)
        print("[Stage 3/7] Schema creation")
        cur.execute("BEGIN;")

        # Partial cleanup if needed (must run inside transaction).
        if state == "partial":
            _cleanup_partial(cur, tables)

        # Single-statement executes so each participates in the transaction.
        cur.execute(DDL_CHEMICALS)
        for stmt in DDL_CHEMICALS_INDEXES:
            cur.execute(stmt)
        cur.execute(DDL_SPECIES)
        for stmt in DDL_SPECIES_INDEXES:
            cur.execute(stmt)
        cur.execute(DDL_LOCATIONS)
        for stmt in DDL_LOCATIONS_INDEXES:
            cur.execute(stmt)
        for simple in ("plant_parts", "analytical_techniques",
                       "extraction_methods", "development_stages"):
            cur.execute(DDL_SIMPLE_TEMPLATE.format(table=simple))

        # All 7 junctions + their entity-side indexes.
        for label, (entity_table, junction, entity_col) in LABEL_TO_TABLE.items():
            cur.execute(DDL_JUNCTION_TEMPLATE.format(
                junction=junction, entity_col=entity_col, entity_table=entity_table
            ))
            cur.execute(DDL_JUNCTION_INDEX_TEMPLATE.format(
                junction=junction, entity_col=entity_col
            ))

        print(f"  Created {len(NEW_ENTITY_TABLES)} entity tables + "
              f"{len(NEW_JUNCTION_TABLES)} junction tables + indexes")

        # Make sure papers has created_at/updated_at columns. SQLite forbids
        # non-constant DEFAULT in ALTER TABLE ADD COLUMN, so we add without
        # default and backfill with UPDATE.
        paper_cols = {row[1] for row in cur.execute("PRAGMA table_info(papers);").fetchall()}
        if "created_at" not in paper_cols:
            cur.execute("ALTER TABLE papers ADD COLUMN created_at TEXT;")
            cur.execute("UPDATE papers SET created_at = datetime('now') WHERE created_at IS NULL;")
            print("  Added papers.created_at (backfilled)")
        if "updated_at" not in paper_cols:
            cur.execute("ALTER TABLE papers ADD COLUMN updated_at TEXT;")
            cur.execute("UPDATE papers SET updated_at = datetime('now') WHERE updated_at IS NULL;")
            print("  Added papers.updated_at (backfilled)")

        # ─── Stage 4: load matchers + read old rows ────────────────────
        print("=" * 70)
        print("[Stage 4/7] Loading matchers + reading old paper_entities")
        chem_matcher, spec_matcher = _load_matchers()

        t0 = time.time()
        cur.execute(
            "SELECT id, paper_id, label, canonical_text, mention_count, metadata_json "
            "FROM paper_entities;"
        )
        old_rows: List[sqlite3.Row] = cur.fetchall()
        print(f"  Loaded {len(old_rows):,} rows in {time.time() - t0:.2f}s")

        # ─── Stage 5: group + enrich + insert entities ─────────────────
        print("=" * 70)
        print("[Stage 5/7] Deduplicating + enriching + inserting entities")

        # group_key -> {label, canonical_lower, display_counter, sqlite_meta}
        groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
        unknown_labels: Counter = Counter()
        for r in old_rows:
            label = r["label"]
            if label not in LABEL_TO_TABLE:
                unknown_labels[label] += 1
                continue
            ct = (r["canonical_text"] or "").strip()
            if not ct:
                continue
            canonical_lower = ct.lower()
            key = (label, canonical_lower)
            g = groups.get(key)
            if g is None:
                g = {
                    "label": label,
                    "canonical_lower": canonical_lower,
                    "display_counter": Counter(),
                    "sqlite_meta": {},
                }
                groups[key] = g
            g["display_counter"][ct] += 1
            row_meta = _parse_metadata(r["metadata_json"])
            if row_meta:
                _merge_sqlite_meta(g["sqlite_meta"], row_meta, label)

        if unknown_labels:
            print(f"  WARNING: skipped rows with unknown labels: {dict(unknown_labels)}")
        print(f"  {len(groups):,} distinct (label, canonical_lower) groups")

        # entity_id_map[(label, canonical_lower)] = new entity id (per type)
        entity_id_map: Dict[Tuple[str, str], int] = {}

        for key, g in groups.items():
            label = g["label"]
            canonical_lower = g["canonical_lower"]
            display_text = g["display_counter"].most_common(1)[0][0]
            sqlite_meta = g["sqlite_meta"]

            gazetteer_meta: Dict[str, Any] = {}
            if label == "CHEMICAL":
                chem_total += 1
                gazetteer_meta = _lookup_safe(chem_matcher, display_text, canonical_lower)
                if gazetteer_meta:
                    chem_enriched += 1
            elif label == "SPECIES":
                spec_total += 1
                gazetteer_meta = _lookup_safe(spec_matcher, display_text, canonical_lower)
                if gazetteer_meta:
                    spec_enriched += 1

            # SQLite values OVERRIDE gazetteer on key overlap (your curation wins).
            merged = {**gazetteer_meta, **sqlite_meta}
            params = _build_entity_params(label, canonical_lower, display_text, merged)
            cur.execute(_build_entity_insert_sql(label), params)
            entity_id_map[key] = cur.lastrowid

        # Per-table count summary.
        for label, (entity_table, _, _) in LABEL_TO_TABLE.items():
            n = cur.execute(f'SELECT COUNT(*) FROM "{entity_table}";').fetchone()[0]
            print(f"    {entity_table:25s} {n:>6,} rows")

        if chem_total:
            print(f"  Chemical gazetteer hits: {chem_enriched:,} / {chem_total:,}  "
                  f"({chem_enriched / chem_total * 100:.1f}%)")
        if spec_total:
            print(f"  Species gazetteer hits:  {spec_enriched:,} / {spec_total:,}  "
                  f"({spec_enriched / spec_total * 100:.1f}%)")

        # ─── Stage 6: rewrite junctions ────────────────────────────────
        print("=" * 70)
        print("[Stage 6/7] Rewriting junctions (per-type)")
        t0 = time.time()
        junction_insert_sql_by_label = {
            label: _build_junction_insert_sql(label) for label in LABEL_TO_TABLE
        }
        skipped_rows = 0
        for r in old_rows:
            label = r["label"]
            if label not in LABEL_TO_TABLE:
                skipped_rows += 1
                continue
            canonical_lower = (r["canonical_text"] or "").strip().lower()
            if not canonical_lower:
                skipped_rows += 1
                continue
            entity_id = entity_id_map[(label, canonical_lower)]
            cur.execute(
                junction_insert_sql_by_label[label],
                (r["paper_id"], entity_id, int(r["mention_count"] or 1)),
            )

        if skipped_rows:
            print(f"  Skipped {skipped_rows} rows with unknown/empty labels")

        # Refresh papers.entity_count from the new junctions (sum across 7).
        total_count_sql = " + ".join(
            f"(SELECT COUNT(*) FROM {junction} WHERE paper_id = papers.id)"
            for _, (_, junction, _) in LABEL_TO_TABLE.items()
        )
        cur.execute(f"UPDATE papers SET entity_count = {total_count_sql};")

        # Per-junction row counts.
        per_table_counts: Dict[str, int] = {}
        for label, (_, junction, _) in LABEL_TO_TABLE.items():
            per_table_counts[junction] = cur.execute(
                f'SELECT COUNT(*) FROM "{junction}";'
            ).fetchone()[0]
        new_junction_total = sum(per_table_counts.values())
        for j, n in per_table_counts.items():
            print(f"    {j:35s} {n:>7,} rows")
        print(
            f"  Junction total: {new_junction_total:,}  "
            f"(was {pe_rows_before:,}; collapse delta: {pe_rows_before - new_junction_total:,})"
            f" in {time.time() - t0:.2f}s"
        )

        # ─── Stage 7: validation + swap ────────────────────────────────
        print("=" * 70)
        print("[Stage 7/7] Validation + swap")

        # Validation 1: papers count unchanged.
        papers_after = cur.execute("SELECT COUNT(*) FROM papers;").fetchone()[0]
        if papers_after != papers_before:
            raise RuntimeError(
                f"papers count drift: before={papers_before} after={papers_after}"
            )
        print(f"  papers count unchanged: {papers_after:,}")

        # Validation 2: sum(mention_count) preserved across all junctions.
        sum_sql = " + ".join(
            f"(SELECT COALESCE(SUM(mention_count), 0) FROM {junction})"
            for _, (_, junction, _) in LABEL_TO_TABLE.items()
        )
        mention_sum_after = cur.execute(f"SELECT {sum_sql};").fetchone()[0]
        if mention_sum_after != mention_sum_before:
            raise RuntimeError(
                f"mention_count sum drift: before={mention_sum_before} after={mention_sum_after}"
            )
        print(f"  sum(mention_count) preserved: {mention_sum_after:,}")

        # Validation 3: entities count across all per-type tables == distinct groups.
        entities_total = sum(
            cur.execute(f'SELECT COUNT(*) FROM "{entity_table}";').fetchone()[0]
            for _, (entity_table, _, _) in LABEL_TO_TABLE.items()
        )
        if entities_total != len(groups):
            raise RuntimeError(
                f"entities total {entities_total} != distinct groups {len(groups)}"
            )
        print(f"  entities total matches distinct groups: {entities_total:,}")

        # Validation 4: per-paper spot check on 5 random papers.
        sample_ids = [
            row[0] for row in cur.execute(
                "SELECT id FROM papers ORDER BY RANDOM() LIMIT 5;"
            ).fetchall()
        ]
        for pid in sample_ids:
            old_distinct = cur.execute(
                "SELECT COUNT(*) FROM (SELECT DISTINCT label, LOWER(canonical_text) "
                "FROM paper_entities WHERE paper_id = ?);",
                (pid,),
            ).fetchone()[0]
            new_total = sum(
                cur.execute(
                    f'SELECT COUNT(*) FROM "{junction}" WHERE paper_id = ?;', (pid,)
                ).fetchone()[0]
                for _, (_, junction, _) in LABEL_TO_TABLE.items()
            )
            if old_distinct != new_total:
                raise RuntimeError(
                    f"paper {pid}: old distinct={old_distinct} new total junction={new_total}"
                )
            print(f"  paper {pid}: {old_distinct} distinct entities (match)")

        # Drop the old paper_entities table — we no longer need it.
        cur.execute("DROP TABLE paper_entities;")
        print("  Dropped old paper_entities")

        if dry_run:
            print("  [DRY RUN] ROLLBACK instead of COMMIT")
            cur.execute("ROLLBACK;")
        else:
            cur.execute("COMMIT;")
            print("  COMMIT")

    except Exception:
        print("ERROR during migration; rolling back.", file=sys.stderr)
        traceback.print_exc()
        try:
            cur.execute("ROLLBACK;")
        except sqlite3.Error:
            pass
        conn.close()
        return 1

    # ─── Post: FK re-enable + VACUUM ────────────────────────────────────
    cur.execute("PRAGMA foreign_keys=ON;")
    if not dry_run:
        print("  Running VACUUM ...")
        cur.execute("VACUUM;")

    size_after = os.path.getsize(db_path)
    conn.close()

    # ─── Summary ────────────────────────────────────────────────────────
    print("=" * 70)
    print("Summary")
    print(f"  Mode:                 {'DRY RUN' if dry_run else 'COMMITTED'}")
    print(f"  Backup file:          {backup_path}")
    print(f"  Papers:               {papers_before:,}")
    print(f"  paper_entities rows:  {pe_rows_before:,} -> {new_junction_total:,}  (across 7 junctions)")
    print(f"  Entities created:     {len(entity_id_map):,}  (across 7 per-type tables)")
    if chem_total:
        print(f"  CHEMICAL enriched:    {chem_enriched:,} / {chem_total:,}")
    if spec_total:
        print(f"  SPECIES enriched:     {spec_enriched:,} / {spec_total:,}")
    print(f"  DB size before:       {size_before:,} bytes")
    print(f"  DB size after:        {size_after:,} bytes")
    print("=" * 70)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate phytoquery.sqlite from the OLD 2-table layout "
            "(papers + paper_entities-with-metadata_json) to the NEW 15-table "
            "per-type schema (papers + 7 entity tables + 7 junction tables)."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the full migration inside a transaction but ROLLBACK "
             "instead of COMMIT. Prints the same summary so you can "
             "preview the result without persisting anything.",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=str(DB_PATH),
        help=f"Path to the SQLite DB (default: {DB_PATH}).",
    )
    args = parser.parse_args(argv)

    return migrate(Path(args.db_path), args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
