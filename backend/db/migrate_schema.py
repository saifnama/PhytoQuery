"""
One-shot SQLite migration: 2-table layout -> 3-table star schema.

OLD schema:
    papers(id, doi, title, journal, year, is_open_access, entity_count)
    paper_entities(id, paper_id, label, canonical_text, mention_count, metadata_json)

NEW schema:
    papers           -- unchanged (created_at/updated_at added if missing)
    entities         -- one row per UNIQUE(label, canonical_text), typed
                        columns for SPECIES/CHEMICAL/LOCATION + provenance +
                        aliases_json + metadata_json catch-all
    paper_entities   -- junction (paper_id, entity_id, mention_count),
                        composite PK, WITHOUT ROWID
    schema_version   -- migration tracker

Casing rule: ``canonical_text`` is stored LOWERCASED in entities, so casing
drift collapses ("Lavandula stoechas" + "lavandula stoechas" -> one row,
mention_counts summed on the junction). ``display_text`` keeps the most
common original casing for UI rendering.

CHEMICAL and SPECIES entities are enriched from the in-process gazetteers
(``backend.gazetteer.chemical_matcher`` and ``backend.gazetteer.species_matcher``).
LOCATION is enriched purely from the curated ``metadata_json`` blob on
paper_entities. All other labels (PLANT PART, ANALYTICAL TECHNIQUE, ...)
leave the enrichment columns NULL.

Run from the repo root:
    python -m backend.db.migrate_schema
    python -m backend.db.migrate_schema --dry-run

A timestamped backup ``phytoquery.sqlite.bak.YYYYMMDD_HHMMSS`` is taken
before any writes. The migration runs inside a single transaction and
ROLLs BACK on any validation failure.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import sqlite3
import sys
import time
import traceback
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = REPO_ROOT / "db_data" / "phytoquery.sqlite"

# Columns we promote from the merged (gazetteer + sqlite) blob into typed
# entity columns. Anything left over after promotion lands in metadata_json.
PROMOTED_KEYS = {
    "accepted_scientific_name",
    "common_name",
    "family",
    "taxon_id",
    "name_type",
    "inchikey",
    "smiles",
    "molecular_formula",
    "preferred_name",
    "country",
    "state",
    "source_db",
    "source_url",
    "aliases",  # -> aliases_json
}

# Matcher-internal keys we never carry into the entity row.
MATCHER_INTERNAL_KEYS = {
    "text",
    "span",
    "canonical",
    "type",
    "label",
    "score",
    "linked_to",
    "match_status",
    "review_required",
    "scientific_name_verified",
}


# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

DDL_ENTITIES = """
CREATE TABLE entities (
    id INTEGER PRIMARY KEY,
    label TEXT NOT NULL,
    canonical_text TEXT NOT NULL,
    display_text TEXT NOT NULL,
    accepted_scientific_name TEXT,
    common_name TEXT,
    family TEXT,
    taxon_id TEXT,
    name_type TEXT,
    inchikey TEXT,
    smiles TEXT,
    molecular_formula TEXT,
    preferred_name TEXT,
    country TEXT,
    state TEXT,
    source_db TEXT,
    source_url TEXT,
    aliases_json TEXT,
    metadata_json TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    UNIQUE(label, canonical_text)
);
"""

DDL_ENTITY_INDEXES = [
    "CREATE INDEX idx_entities_label ON entities(label);",
    "CREATE INDEX idx_entities_family ON entities(family);",
    "CREATE INDEX idx_entities_country ON entities(country);",
    "CREATE INDEX idx_entities_state ON entities(state);",
    "CREATE INDEX idx_entities_inchikey ON entities(inchikey);",
    "CREATE INDEX idx_entities_taxon_id ON entities(taxon_id);",
    "CREATE INDEX idx_entities_molecular_formula ON entities(molecular_formula);",
]

DDL_PAPER_ENTITIES_NEW = """
CREATE TABLE paper_entities_new (
    paper_id INTEGER NOT NULL REFERENCES papers(id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    mention_count INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (paper_id, entity_id)
) WITHOUT ROWID;
"""

DDL_PAPER_ENTITIES_INDEX = "CREATE INDEX idx_pe_entity_id ON paper_entities_new(entity_id);"

DDL_SCHEMA_VERSION = """
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT DEFAULT (datetime('now')),
    description TEXT NOT NULL
);
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_metadata(raw: Any) -> Dict[str, Any]:
    """Parse a paper_entities.metadata_json value into a dict.

    The column holds the literal string ``"null"`` for ~95% of rows; only
    SPECIES (family/common_name) and LOCATION (country/state) carry real
    JSON. Anything that isn't a dict becomes an empty dict.
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
    if isinstance(parsed, dict):
        return parsed
    return {}


def _strip_matcher_internal(d: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Drop matcher-internal keys and None values from a gazetteer result."""
    if not d:
        return {}
    out: Dict[str, Any] = {}
    for k, v in d.items():
        if k in MATCHER_INTERNAL_KEYS:
            continue
        if v is None:
            continue
        # An empty aliases list is uninteresting; skip it.
        if k == "aliases" and isinstance(v, list) and len(v) == 0:
            continue
        out[k] = v
    return out


def _merge_sqlite_meta(
    sqlite_meta: Dict[str, Any], row_meta: Dict[str, Any], label: str
) -> Dict[str, Any]:
    """Accumulate per-row metadata into a single dict for the entity group.

    LOCATION: union of (country, state) -- first non-empty value wins.
    SPECIES:  first non-empty (family, common_name) wins.
    Everything else: ignored.
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


def _build_entity_row(
    label: str,
    canonical_lower: str,
    display_text: str,
    merged: Dict[str, Any],
) -> Tuple[Any, ...]:
    """Project the merged dict into the entities INSERT parameter tuple."""
    aliases = merged.get("aliases")
    aliases_json = (
        json.dumps(aliases) if isinstance(aliases, (list, tuple)) and aliases else None
    )

    leftover = {
        k: v
        for k, v in merged.items()
        if k not in PROMOTED_KEYS and v is not None and v != ""
    }
    metadata_json = json.dumps(leftover) if leftover else None

    return (
        label,
        canonical_lower,
        display_text,
        merged.get("accepted_scientific_name"),
        merged.get("common_name"),
        merged.get("family"),
        merged.get("taxon_id"),
        merged.get("name_type"),
        merged.get("inchikey"),
        merged.get("smiles"),
        merged.get("molecular_formula"),
        merged.get("preferred_name"),
        merged.get("country"),
        merged.get("state"),
        merged.get("source_db"),
        merged.get("source_url"),
        aliases_json,
        metadata_json,
    )


def _load_matchers():
    """Import and instantiate the two gazetteer matchers once."""
    from backend.gazetteer.chemical_matcher import get_matcher as get_chemical_matcher
    from backend.gazetteer.species_matcher import get_matcher as get_species_matcher

    print("  Loading chemical matcher ...")
    chem = get_chemical_matcher()
    print("  Loading species matcher ...")
    spec = get_species_matcher()
    return chem, spec


def _lookup_safe(matcher, display_text: str, canonical_lower: str) -> Dict[str, Any]:
    """Look up a term in a matcher, trying both display and canonical forms."""
    try:
        hit = matcher.lookup(display_text)
        if hit is None and display_text.lower() != canonical_lower:
            hit = matcher.lookup(canonical_lower)
        return _strip_matcher_internal(hit)
    except Exception as exc:  # noqa: BLE001 -- per-row resilience
        print(f"    [warn] matcher lookup failed for {display_text!r}: {exc}")
        return {}


# ---------------------------------------------------------------------------
# Migration
# ---------------------------------------------------------------------------

def migrate(db_path: Path, dry_run: bool) -> int:
    if not db_path.exists():
        print(f"ERROR: DB not found at {db_path}", file=sys.stderr)
        return 2

    # ---- Pre-flight: backup ------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.with_name(f"{db_path.name}.bak.{timestamp}")
    size_before = os.path.getsize(db_path)

    print("=" * 70)
    print(f"[Stage 1/7] Backup")
    print(f"  Source: {db_path}")
    print(f"  Backup: {backup_path}")
    shutil.copy2(db_path, backup_path)
    print(f"  Backup size: {os.path.getsize(backup_path):,} bytes")

    conn = sqlite3.connect(str(db_path))
    # Autocommit mode so the explicit BEGIN/COMMIT/ROLLBACK we issue covers
    # ALL statements including DDL. Python's sqlite3 default ("" — deferred)
    # auto-commits before DDL, which means rollback() only undoes DML, not
    # CREATE/ALTER. With isolation_level=None we get a single real transaction.
    conn.isolation_level = None
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("PRAGMA foreign_keys=OFF;")

    # ---- Idempotency check + partial-state cleanup ------------------------
    # An ``entities`` table by itself isn't conclusive — a previous failed
    # run may have created the new tables but never swapped paper_entities.
    # Distinguish: if paper_entities still has the OLD ``label`` column, the
    # migration is incomplete and we should clean up dangling empty tables
    # and proceed. Otherwise (paper_entities is the junction), refuse.
    tables = {
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
        ).fetchall()
    }
    if "entities" in tables:
        pe_cols = {row[1] for row in cur.execute("PRAGMA table_info(paper_entities);").fetchall()}
        if "label" not in pe_cols:
            print("Already migrated (paper_entities is the junction). Exiting.")
            conn.close()
            return 0
        # Partial state: clean up empty new tables before retry.
        print("  Detected partial state from a previous failed run.")
        for t in ("entities", "paper_entities_new", "schema_version"):
            if t in tables:
                cnt = cur.execute(f'SELECT COUNT(*) FROM "{t}";').fetchone()[0]
                if cnt != 0:
                    print(
                        f"ERROR: dangling table {t!r} has {cnt} rows; refusing to "
                        f"auto-clean. Investigate manually before re-running.",
                        file=sys.stderr,
                    )
                    conn.close()
                    return 3
                cur.execute(f"DROP TABLE {t};")
                print(f"  Cleaned dangling empty table: {t}")

    # Snapshot counters used by validation later.
    cur.execute("SELECT COUNT(*) FROM papers;")
    papers_before = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM paper_entities;")
    pe_rows_before = cur.fetchone()[0]
    cur.execute("SELECT COALESCE(SUM(mention_count), 0) FROM paper_entities;")
    mention_sum_before = cur.fetchone()[0]
    print(f"  Papers: {papers_before:,}")
    print(f"  paper_entities rows: {pe_rows_before:,}")
    print(f"  Sum(mention_count) before: {mention_sum_before:,}")

    try:
        # ---- Stage 2: schema creation -------------------------------------
        print("=" * 70)
        print("[Stage 2/7] Schema creation")
        cur.execute("BEGIN;")
        cur.executescript(DDL_ENTITIES)
        for stmt in DDL_ENTITY_INDEXES:
            cur.execute(stmt)
        cur.executescript(DDL_PAPER_ENTITIES_NEW)
        cur.execute(DDL_PAPER_ENTITIES_INDEX)
        cur.executescript(DDL_SCHEMA_VERSION)
        # Make sure papers has created_at/updated_at (older DBs lack them).
        # SQLite forbids non-constant DEFAULTs in ALTER TABLE ADD COLUMN
        # (datetime('now') is a function call, not a constant). Add the
        # column with no default and backfill existing rows with a single
        # UPDATE. Future INSERTs from the app populate the value via
        # SQLAlchemy's Python-side ``default=`` callable in models.py.
        cur.execute("PRAGMA table_info(papers);")
        paper_cols = {row[1] for row in cur.fetchall()}
        if "created_at" not in paper_cols:
            cur.execute("ALTER TABLE papers ADD COLUMN created_at TEXT;")
            cur.execute(
                "UPDATE papers SET created_at = datetime('now') WHERE created_at IS NULL;"
            )
            print("  Added papers.created_at (backfilled to datetime('now'))")
        if "updated_at" not in paper_cols:
            cur.execute("ALTER TABLE papers ADD COLUMN updated_at TEXT;")
            cur.execute(
                "UPDATE papers SET updated_at = datetime('now') WHERE updated_at IS NULL;"
            )
            print("  Added papers.updated_at (backfilled to datetime('now'))")
        print("  Created entities, paper_entities_new, schema_version")

        # ---- Load matchers (after schema; before heavy loop) --------------
        chem_matcher, spec_matcher = _load_matchers()

        # ---- Stage 3: read all old paper_entities rows --------------------
        print("=" * 70)
        print("[Stage 3/7] Reading old paper_entities into memory")
        t0 = time.time()
        cur.execute(
            "SELECT id, paper_id, label, canonical_text, mention_count, metadata_json "
            "FROM paper_entities;"
        )
        old_rows: List[sqlite3.Row] = cur.fetchall()
        print(f"  Loaded {len(old_rows):,} rows in {time.time() - t0:.2f}s")

        # ---- Stage 4: group + enrich + insert entities --------------------
        print("=" * 70)
        print("[Stage 4/7] Deduplicating + enriching entities")

        # group_key -> {
        #   "label": ..., "canonical_lower": ...,
        #   "display_counter": Counter, "sqlite_meta": dict
        # }
        groups: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in old_rows:
            label = r["label"]
            ct = (r["canonical_text"] or "").strip()
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

        print(f"  {len(groups):,} distinct (label, canonical_lower) groups")

        # Build & insert entities.
        entity_id_map: Dict[Tuple[str, str], int] = {}
        chem_total = chem_enriched = 0
        spec_total = spec_enriched = 0

        for key, g in groups.items():
            label = g["label"]
            canonical_lower = g["canonical_lower"]
            display_text = g["display_counter"].most_common(1)[0][0]
            sqlite_meta = g["sqlite_meta"]

            gazetteer_meta: Dict[str, Any] = {}
            if label == "CHEMICAL":
                chem_total += 1
                gazetteer_meta = _lookup_safe(
                    chem_matcher, display_text, canonical_lower
                )
                if gazetteer_meta:
                    chem_enriched += 1
            elif label == "SPECIES":
                spec_total += 1
                gazetteer_meta = _lookup_safe(
                    spec_matcher, display_text, canonical_lower
                )
                if gazetteer_meta:
                    spec_enriched += 1

            # SQLite values OVERRIDE gazetteer on key overlap.
            merged = {**gazetteer_meta, **sqlite_meta}
            params = _build_entity_row(label, canonical_lower, display_text, merged)
            cur.execute(
                "INSERT INTO entities ("
                "label, canonical_text, display_text, "
                "accepted_scientific_name, common_name, family, taxon_id, name_type, "
                "inchikey, smiles, molecular_formula, preferred_name, "
                "country, state, "
                "source_db, source_url, aliases_json, metadata_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                params,
            )
            entity_id_map[key] = cur.lastrowid

        print(f"  Inserted {len(entity_id_map):,} entities")
        print(
            f"  Chemical gazetteer hits: {chem_enriched:,} / {chem_total:,}"
            + (f"  ({chem_enriched / chem_total * 100:.1f}%)" if chem_total else "")
        )
        print(
            f"  Species gazetteer hits:  {spec_enriched:,} / {spec_total:,}"
            + (f"  ({spec_enriched / spec_total * 100:.1f}%)" if spec_total else "")
        )

        # ---- Stage 5: rewrite junction ------------------------------------
        print("=" * 70)
        print("[Stage 5/7] Rewriting paper_entities junction")
        t0 = time.time()
        insert_sql = (
            "INSERT INTO paper_entities_new(paper_id, entity_id, mention_count) "
            "VALUES (?, ?, ?) "
            "ON CONFLICT(paper_id, entity_id) DO UPDATE SET "
            "mention_count = paper_entities_new.mention_count + excluded.mention_count;"
        )
        for r in old_rows:
            label = r["label"]
            canonical_lower = (r["canonical_text"] or "").strip().lower()
            entity_id = entity_id_map[(label, canonical_lower)]
            cur.execute(
                insert_sql,
                (r["paper_id"], entity_id, int(r["mention_count"] or 1)),
            )
        cur.execute("SELECT COUNT(*) FROM paper_entities_new;")
        new_junction_count = cur.fetchone()[0]
        print(
            f"  Junction rows: {new_junction_count:,}"
            f"  (was {pe_rows_before:,}, collapse delta: "
            f"{pe_rows_before - new_junction_count:,}) in {time.time() - t0:.2f}s"
        )

        # Refresh papers.entity_count from the new junction.
        cur.execute(
            "UPDATE papers SET entity_count = "
            "(SELECT COUNT(*) FROM paper_entities_new WHERE paper_id = papers.id);"
        )
        print("  Refreshed papers.entity_count")

        # ---- Schema version row -------------------------------------------
        cur.execute(
            "INSERT INTO schema_version (version, description) VALUES (?, ?);",
            (1, "3-table star: papers + entities + paper_entities junction"),
        )

        # ---- Stage 6: validation ------------------------------------------
        print("=" * 70)
        print("[Stage 6/7] Validation")

        cur.execute("SELECT COUNT(*) FROM papers;")
        papers_after = cur.fetchone()[0]
        assert papers_after == papers_before, (
            f"papers count drift: before={papers_before} after={papers_after}"
        )
        print(f"  papers count unchanged: {papers_after:,}")

        cur.execute("SELECT COALESCE(SUM(mention_count), 0) FROM paper_entities_new;")
        mention_sum_after = cur.fetchone()[0]
        assert mention_sum_after == mention_sum_before, (
            f"mention_count sum drift: before={mention_sum_before} "
            f"after={mention_sum_after}"
        )
        print(f"  sum(mention_count) preserved: {mention_sum_after:,}")

        cur.execute("SELECT COUNT(*) FROM entities;")
        entities_count = cur.fetchone()[0]
        assert entities_count == len(groups), (
            f"entities count {entities_count} != distinct groups {len(groups)}"
        )
        print(f"  entities count matches distinct groups: {entities_count:,}")

        # Per-paper spot check on 5 random papers.
        cur.execute("SELECT id FROM papers ORDER BY RANDOM() LIMIT 5;")
        sample_ids = [row[0] for row in cur.fetchall()]
        for pid in sample_ids:
            cur.execute(
                "SELECT COUNT(*) FROM ("
                "SELECT DISTINCT label, LOWER(canonical_text) "
                "FROM paper_entities WHERE paper_id = ?);",
                (pid,),
            )
            old_distinct = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM paper_entities_new WHERE paper_id = ?;",
                (pid,),
            )
            new_count = cur.fetchone()[0]
            assert old_distinct == new_count, (
                f"paper {pid}: old distinct={old_distinct} new junction={new_count}"
            )
            print(f"  paper {pid}: {old_distinct} distinct entities (match)")

        # ---- Stage 7: swap + commit ---------------------------------------
        print("=" * 70)
        print("[Stage 7/7] Swap + commit")
        cur.execute("DROP TABLE paper_entities;")
        cur.execute("ALTER TABLE paper_entities_new RENAME TO paper_entities;")
        print("  Dropped old paper_entities; renamed paper_entities_new -> paper_entities")

        if dry_run:
            print("  [DRY RUN] ROLLBACK instead of COMMIT")
            cur.execute("ROLLBACK;")
        else:
            cur.execute("COMMIT;")
            print("  COMMIT")

    except Exception:
        print("ERROR during migration; rolling back.", file=sys.stderr)
        traceback.print_exc()
        # SQL-level ROLLBACK — conn.rollback() proved unreliable for DDL
        # in autocommit mode. Use cur.execute("ROLLBACK") for full effect.
        try:
            cur.execute("ROLLBACK;")
        except sqlite3.Error:
            pass
        conn.close()
        return 1

    # ---- Post: foreign keys + VACUUM --------------------------------------
    cur.execute("PRAGMA foreign_keys=ON;")
    if not dry_run:
        print("  Running VACUUM ...")
        conn.isolation_level = None  # autocommit so VACUUM can run
        cur.execute("VACUUM;")

    size_after = os.path.getsize(db_path)
    conn.close()

    # ---- Summary ----------------------------------------------------------
    print("=" * 70)
    print("Summary")
    print(f"  Mode:                 {'DRY RUN' if dry_run else 'COMMITTED'}")
    print(f"  Backup file:          {backup_path}")
    print(f"  Papers:               {papers_before:,}")
    print(f"  paper_entities rows:  {pe_rows_before:,} -> {new_junction_count:,}")
    print(f"  entities created:     {len(entity_id_map):,}")
    print(f"  CHEMICAL enriched:    {chem_enriched:,} / {chem_total:,}")
    print(f"  SPECIES enriched:     {spec_enriched:,} / {spec_total:,}")
    print(f"  DB size before:       {size_before:,} bytes")
    print(f"  DB size after:        {size_after:,} bytes")
    print("=" * 70)
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Migrate phytoquery.sqlite from the 2-table layout "
            "(papers + paper_entities-with-metadata_json) to the 3-table "
            "star schema (papers + entities + paper_entities junction)."
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

    # Deterministic sample for validation step.
    random.seed(0xC0FFEE)
    return migrate(Path(args.db_path), args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
