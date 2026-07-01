"""
evaluate.py

Loads the 3 system config outputs and gold standard, then computes
Precision, Recall, and F1 using nervaluate (SemEval 2013 standard).

Run this after extract.py has produced the 3 config JSON files.
Re-run this as many times as you like -- it reads files, calls no APIs.

Outputs:
    results/results.csv          -- overall P/R/F1 per config
    results/results_per_type.csv -- per entity type F1

Usage:
    python evaluate.py
"""

import json
import csv
import os
from pathlib import Path

# -- Paths (relative to this script's location) --------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_DATA_DIR   = _SCRIPT_DIR / "data"
_RESULTS_DIR = _SCRIPT_DIR / "results"

GOLD_PATH = str(_DATA_DIR / "annotated_data.json")

CONFIGS = {
    "Dictionary": str(_DATA_DIR / "dictionary.json"),
    "LLM":        str(_DATA_DIR / "llm.json"),
    "Hybrid":     str(_DATA_DIR / "hybrid.json"),
}

# -- Entity types --------------------------------------------------------------
ALL_TYPES = [
    "CHEMICAL", "SPECIES", "BIOACTIVITY", "DISEASE", "LOCATION",
    "PLANT PART", "EXTRACTION METHOD", "ANALYTICAL TECHNIQUE",
    "DEVELOPMENT STAGE", "SEASON",
]

CONFIG_TYPES = {
    "Dictionary": [
        "CHEMICAL", "SPECIES", "BIOACTIVITY",
        "ANALYTICAL TECHNIQUE", "EXTRACTION METHOD",
        "PLANT PART", "DEVELOPMENT STAGE", "SEASON",
    ],
    "LLM": [
        "CHEMICAL", "SPECIES", "BIOACTIVITY", "LOCATION", "DISEASE",
    ],
    "Hybrid": [
        "CHEMICAL", "SPECIES", "BIOACTIVITY", "LOCATION", "DISEASE",
        "ANALYTICAL TECHNIQUE", "EXTRACTION METHOD",
        "PLANT PART", "DEVELOPMENT STAGE", "SEASON",
    ],
}


# -----------------------------------------------------------------------------
# Load gold standard
# -----------------------------------------------------------------------------

def load_gold(path: str):
    """
    Load Label Studio JSON export.

    Returns:
        gold:  {doc_id: [{'label':..., 'start':..., 'end':...}]}
        texts: {doc_id: text_string}

    VERIFIED: Label Studio value.start/end are character offsets.
    nervaluate dict loader accepts character offsets directly.
    """
    with open(path, encoding="utf-8") as f:
        tasks = json.load(f)

    gold, texts = {}, {}
    for task in tasks:
        anns = task.get("annotations", [])
        if not anns:
            continue
        doc_id = task["data"].get("doc_id") or task["data"].get("doi", str(task["id"]))
        texts[doc_id] = task["data"]["text"]
        spans = []
        for result in anns[0].get("result", []):
            v = result.get("value", {})
            if "labels" in v and v["labels"]:
                spans.append({
                    "label": v["labels"][0],
                    "start": v["start"],
                    "end":   v["end"],
                })
        gold[doc_id] = spans
    return gold, texts


# -----------------------------------------------------------------------------
# Load system config outputs
# -----------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a config JSON file saved by extract.py."""
    with open(path) as f:
        return json.load(f)


def to_nervaluate(spans: list) -> list:
    """Strip to {label, start, end} and sort exact-matches before overlapping partials.
    nervaluate's strict strategy processes preds greedily — the first pred that
    overlaps a gold entity claims it.  Sorting by (start, -length) ensures
    exact matches on a position are evaluated before shorter overlaps.
    """
    # Sort: same start → longest span first (exact match before overlapping partial)
    spans = sorted(spans, key=lambda s: (s["start"], -(s["end"] - s["start"])))
    return [{"label": s["label"], "start": s["start"], "end": s["end"]} for s in spans]


# -----------------------------------------------------------------------------
# Evaluate
# -----------------------------------------------------------------------------

def evaluate(gold: dict, configs: dict) -> dict:
    """
    Run nervaluate for each config. Returns results dict with strict
    and partial P/R/F1 plus per-entity-type strict F1.

    Each config is evaluated only on entity types it was designed to
    handle (see CONFIG_TYPES).  Gold and predicted spans are filtered
    to those types before evaluation.

    VERIFIED nervaluate v1.0.0+ API:
      result = evaluator.evaluate()   <- single dict, NOT a tuple
      result["overall"]["strict"].f1  <- EvaluationResult attribute access
      result["entities"][type]["strict"].f1
    """
    from nervaluate import Evaluator

    results = {}

    for cfg_name, cfg_data in configs.items():
        cfg_types = CONFIG_TYPES.get(cfg_name, ALL_TYPES)
        type_set  = set(cfg_types)

        # Evaluate only on docs present in BOTH gold and this config's predictions
        doc_ids = sorted(set(gold) & set(cfg_data))
        if not doc_ids:
            print(f"\n  {cfg_name} -- no overlapping docs with gold, skipping")
            continue
        print(f"\n  {cfg_name} -- evaluating on {len(doc_ids)} docs")

        y_true = [
            to_nervaluate([s for s in gold[d] if s["label"] in type_set])
            for d in doc_ids
        ]
        y_pred = [
            to_nervaluate([s for s in cfg_data.get(d, []) if s["label"] in type_set])
            for d in doc_ids
        ]

        evaluator = Evaluator(y_true, y_pred, tags=cfg_types)
        result    = evaluator.evaluate()

        overall = result["overall"]
        per_tag = result["entities"]

        s  = overall["strict"]
        pt = overall["partial"]

        per_type_f1 = {}
        for ent in cfg_types:
            tag   = per_tag.get(ent, {}).get("strict")
            f1    = tag.f1 if tag else 0.0
            per_type_f1[ent] = f1

        # Macro F1: mean of per-type F1, excluding types with zero gold instances
        gold_types_with_data = set()
        for d in doc_ids:
            for gs in gold[d]:
                gold_types_with_data.add(gs["label"])
        macro_f1s = [f1 for ent, f1 in per_type_f1.items() if ent in gold_types_with_data]
        macro_f1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0
        status = "  (excludes zero-gold types)" if len(macro_f1s) < len(per_type_f1) else ""

        print(f"\n{'='*55}")
        print(f"  {cfg_name}")
        print(f"{'='*55}")
        print(f"  Micro-avg Strict  P: {s.precision:.4f}  R: {s.recall:.4f}  F1: {s.f1:.4f}")
        print(f"  Macro-avg Strict  F1: {macro_f1:.4f} {status}")
        print(f"  Partial           P: {pt.precision:.4f}  R: {pt.recall:.4f}  F1: {pt.f1:.4f}")
        print(f"\n  Per entity type -- Strict F1:")
        for ent in cfg_types:
            f1   = per_type_f1[ent]
            bar  = "#" * int(f1 * 25)
            flag = "  <- low" if f1 < 0.3 and f1 > 0 else ""
            print(f"    {ent:26s} {f1:.4f}  {bar}{flag}")

        results[cfg_name] = {
            "strict_P":   s.precision,
            "strict_R":   s.recall,
            "strict_F1":  s.f1,
            "macro_F1":   macro_f1,
            "partial_P":  pt.precision,
            "partial_R":  pt.recall,
            "partial_F1": pt.f1,
            "per_type":   per_type_f1,
        }

    return results


# -----------------------------------------------------------------------------
# Save results
# -----------------------------------------------------------------------------

def save_csv(results: dict, base_path: str | None = None):
    if base_path is None:
        base_path = str(_RESULTS_DIR / "results.csv")
    os.makedirs(os.path.dirname(base_path), exist_ok=True)

    # Overall results
    with open(base_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Config",
                    "Micro_P", "Micro_R", "Micro_F1",
                    "Macro_F1",
                    "Partial_P","Partial_R","Partial_F1"])
        for cfg, r in results.items():
            w.writerow([
                cfg,
                f"{r['strict_P']:.4f}",  f"{r['strict_R']:.4f}",  f"{r['strict_F1']:.4f}",
                f"{r['macro_F1']:.4f}",
                f"{r['partial_P']:.4f}", f"{r['partial_R']:.4f}", f"{r['partial_F1']:.4f}",
            ])
    print(f"\n  Saved {base_path}")

    # Per entity type
    per_type_path = base_path.replace(".csv", "_per_type.csv")
    with open(per_type_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Entity_Type"] + list(results.keys()))
        for ent in ALL_TYPES:
            row = [ent] + [
                f"{results[cfg]['per_type'].get(ent, 0):.4f}" for cfg in results
            ]
            w.writerow(row)
    print(f"  Saved {per_type_path}")


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 55)
    print("  Evaluate -- NER Configurations")
    print("=" * 55)

    # -- Load gold -------------------------------------------------------------
    print(f"\nLoading gold standard from {GOLD_PATH}...")
    gold, texts = load_gold(GOLD_PATH)
    total_gold  = sum(len(v) for v in gold.values())
    print(f"  {len(gold)} documents  |  {total_gold} gold entities")

    # -- Check all config files exist -----------------------------------------
    print("\nChecking config files...")
    for cfg_name, path in CONFIGS.items():
        exists = os.path.exists(path)
        total  = sum(len(v) for v in load_config(path).values()) if exists else 0
        status = f"{total} entities" if exists else "NOT FOUND -- run extract.py first"
        print(f"  {'[OK]' if exists else '[MISS]'} {path:35s} {status}")

    # -- Load configs ----------------------------------------------------------
    loaded_configs = {}
    for cfg_name, path in CONFIGS.items():
        if os.path.exists(path):
            loaded_configs[cfg_name] = load_config(path)
        else:
            print(f"  Skipping {cfg_name} -- {path} not found")

    # -- Evaluate --------------------------------------------------------------
    print("\nEvaluating with nervaluate...")
    results = evaluate(gold, loaded_configs)

    # -- Save ------------------------------------------------------------------
    save_csv(results)

    # -- Summary ---------------------------------------------------------------
    print()
    print("=" * 55)
    print("  SUMMARY")
    print("=" * 55)
    for cfg, r in results.items():
        mibar = "#" * int(r["strict_F1"] * 30)
        mabar = "#" * int(r["macro_F1"]  * 30) if r["macro_F1"] else "N/A"
        print(f"  {cfg:32s} Micro-F1: {r['strict_F1']:.4f}  {mibar}")
        print(f"  {'':32s} Macro-F1: {r['macro_F1']:.4f}  {mabar}")

    print()
    print(f"  Copy to paper from:")
    print(f"    {_RESULTS_DIR / 'results.csv'}")
    print(f"    {_RESULTS_DIR / 'results_per_type.csv'}")
