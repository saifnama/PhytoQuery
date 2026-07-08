#!/usr/bin/env python3
"""Convert a JSON QA dataset to CSV files for eval scripts.

Usage:
    python scripts/json_to_csv.py <input.json> [output_dir]

Writes MCQ items to <output_dir>/mcq.csv and open-ended items to
<output_dir>/open_ended.csv.  Skips types not present in the JSON.

If output_dir is omitted, defaults to backend/evals/rag/data/.
"""
import json
import csv
import sys
from pathlib import Path

MCQ_FIELDS = ["id", "question", "option_a", "option_b", "option_c",
              "option_d", "correct_option", "difficulty"]
OPEN_ENDED_FIELDS = ["id", "question", "reference", "difficulty"]

DEFAULT_OUTPUT_DIR = Path("backend/evals/rag/data")


def load_json(input_path: Path) -> list:
    with open(input_path, encoding="utf-8") as f:
        raw = f.read()

    # Strip trailing comma before closing bracket (invalid but common)
    raw = raw.rstrip()
    if raw.endswith(","):
        raw = raw[:-1]
    if not raw.endswith("]"):
        raw += "]"

    return json.loads(raw)


def write_csv(items: list, output_path: Path, fieldnames: list):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in items:
            writer.writerow(item)
    print(f"  {len(items)} items -> {output_path}")


def convert(input_path: Path, output_dir: Path):
    data = load_json(input_path)

    mcq_items = [item for item in data if item.get("type") == "mcq"]
    open_ended_items = [item for item in data if item.get("type") == "open_ended"]

    if not mcq_items and not open_ended_items:
        print(f"No MCQ or open-ended items found in {input_path}")
        sys.exit(1)

    print(f"Converting {input_path.name} ({len(data)} total items):")

    if mcq_items:
        write_csv(mcq_items, output_dir / "mcq.csv", MCQ_FIELDS)

    if open_ended_items:
        write_csv(open_ended_items, output_dir / "open_ended.csv", OPEN_ENDED_FIELDS)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/json_to_csv.py <input.json> [output_dir]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_directory = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT_DIR

    convert(input_file, output_directory)
