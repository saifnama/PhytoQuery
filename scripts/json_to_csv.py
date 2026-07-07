#!/usr/bin/env python3
"""Convert a JSON QA dataset to CSV format for rag_mcq.py and eval_mcq.py.

Usage:
    python scripts/json_to_csv.py <input.json> [output.csv]

If output is omitted, writes to backend/evals/rag/data/mcq.csv.
"""
import json
import csv
import sys
from pathlib import Path


def convert(input_path: Path, output_path: Path):
    with open(input_path, encoding="utf-8") as f:
        raw = f.read()

    # Strip trailing comma before closing bracket (invalid but common)
    raw = raw.rstrip()
    if raw.endswith(","):
        raw = raw[:-1]
    if not raw.endswith("]"):
        raw += "]"

    data = json.loads(raw)

    # Filter to MCQ items only
    mcq_items = [item for item in data if item.get("type") == "mcq"]

    if not mcq_items:
        print(f"No MCQ items found in {input_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["id", "question", "option_a", "option_b", "option_c", "option_d", "correct_option", "difficulty"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for item in mcq_items:
            writer.writerow(item)

    print(f"Converted {len(mcq_items)} MCQ items → {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/json_to_csv.py <input.json> [output.csv]")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    output_file = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("backend/evals/rag/data/mcq.csv")

    convert(input_file, output_file)
