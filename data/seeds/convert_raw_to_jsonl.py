#!/usr/bin/env python3
"""
MediCS — Convert raw chatbot-generated seeds to JSONL format.

Reads raw_*.txt and twins_*.txt files from data/seeds/ and produces:
  - raw_seeds.jsonl   (500 harmful medical prompts)
  - benign_twins.jsonl (500 matched safe questions)

Usage:
  python data/seeds/convert_raw_to_jsonl.py

Expected input files (in same directory as this script):
  raw_TOX.txt, raw_SH.txt, raw_MIS.txt, raw_ULP.txt, raw_PPV.txt, raw_UCA.txt
  twins_TOX.txt, twins_SH.txt, twins_MIS.txt, twins_ULP.txt, twins_PPV.txt, twins_UCA.txt
"""

import json
import os
import re
import sys
from datetime import datetime

# Category definitions (must match experiment_config.yaml)
CATEGORIES = {
    "TOX": {"name": "Toxicology / Drug Synthesis", "count": 84},
    "SH":  {"name": "Self-Harm / Suicide Methods", "count": 83},
    "MIS": {"name": "Medical Misinformation", "count": 83},
    "ULP": {"name": "Unlicensed Medical Practice", "count": 84},
    "PPV": {"name": "Patient Privacy Violations", "count": 83},
    "UCA": {"name": "Unethical Clinical Advice", "count": 83},
}

# Model used for each category (for metadata — update if you used a fallback)
CATEGORY_MODELS = {
    "TOX": "GPT-5.2",
    "SH":  "GPT-5.2",
    "MIS": "Claude Opus 4.6",
    "ULP": "GPT-5.2",
    "PPV": "Gemini 3.1 Pro",
    "UCA": "Claude Opus 4.6",
}


def parse_numbered_list(filepath: str) -> list[str]:
    """
    Parse a numbered list from a text file.
    Handles formats like:
      1. Query text here
      1) Query text here
      1: Query text here
      1 Query text here

    Also handles multi-line queries (continuation lines not starting with a number).
    """
    if not os.path.exists(filepath):
        print(f"  ERROR: File not found: {filepath}")
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split into lines
    lines = content.strip().split("\n")

    queries = []
    current_query = ""
    # Pattern: starts with a number, optionally followed by . or ) or : or space
    number_pattern = re.compile(r"^\s*(\d+)\s*[.):]\s*(.*)")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        match = number_pattern.match(line)
        if match:
            # Save previous query if exists
            if current_query:
                queries.append(current_query.strip())
            current_query = match.group(2)
        else:
            # Continuation of previous query (multi-line)
            if current_query:
                current_query += " " + line

    # Don't forget the last query
    if current_query:
        queries.append(current_query.strip())

    # Filter out empty strings and obvious non-query lines
    queries = [q for q in queries if len(q) > 10]

    return queries


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    print("=" * 60)
    print("MediCS — Raw Seeds to JSONL Converter")
    print("=" * 60)

    # ── Step 1: Parse harmful seeds ──────────────────────────
    print("\n--- Parsing harmful seeds ---")
    all_seeds = []
    errors = []

    for cat_code, cat_info in CATEGORIES.items():
        filepath = os.path.join(script_dir, f"raw_{cat_code}.txt")
        queries = parse_numbered_list(filepath)

        expected = cat_info["count"]
        actual = len(queries)
        status = "OK" if actual >= expected else f"SHORT ({actual}/{expected})"
        model = CATEGORY_MODELS.get(cat_code, "unknown")

        print(f"  {cat_code} ({model}): {actual} queries [{status}]")

        if actual < expected:
            errors.append(f"{cat_code}: got {actual}, need {expected}")
        if actual > expected:
            print(f"    Trimming {actual} -> {expected} (taking first {expected})")
            queries = queries[:expected]

        for i, query in enumerate(queries):
            all_seeds.append({
                "id": f"{cat_code}_{i:03d}",
                "category": cat_code,
                "category_name": cat_info["name"],
                "original_en": query,
                "source_model": model,
                "timestamp": datetime.now().isoformat(),
            })

    print(f"\n  Total seeds: {len(all_seeds)}")

    # ── Step 2: Parse benign twins ───────────────────────────
    print("\n--- Parsing benign twins ---")
    all_twins = []

    for cat_code, cat_info in CATEGORIES.items():
        filepath = os.path.join(script_dir, f"twins_{cat_code}.txt")
        twins = parse_numbered_list(filepath)

        # Get the seeds for this category
        cat_seeds = [s for s in all_seeds if s["category"] == cat_code]
        expected = len(cat_seeds)
        actual = len(twins)
        status = "OK" if actual >= expected else f"SHORT ({actual}/{expected})"

        print(f"  {cat_code}: {actual} twins [{status}]")

        if actual < expected:
            errors.append(f"twins_{cat_code}: got {actual}, need {expected}")
        if actual > expected:
            twins = twins[:expected]

        for i, (twin_text, seed) in enumerate(zip(twins, cat_seeds)):
            all_twins.append({
                "seed_id": seed["id"],
                "category": cat_code,
                "harmful_query": seed["original_en"],
                "benign_question": twin_text,
                "timestamp": datetime.now().isoformat(),
            })

    print(f"\n  Total twins: {len(all_twins)}")

    # ── Step 3: Report errors ────────────────────────────────
    if errors:
        print("\n" + "!" * 60)
        print("WARNINGS — Some files have fewer items than expected:")
        for err in errors:
            print(f"  - {err}")
        print("\nYou can still proceed, but the dataset will be smaller than 500.")
        print("Generate more seeds to fill the gaps, or adjust counts in config.")
        print("!" * 60)

        response = input("\nContinue anyway? (y/n): ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(1)

    # ── Step 4: Write JSONL files ────────────────────────────
    seeds_path = os.path.join(script_dir, "raw_seeds.jsonl")
    twins_path = os.path.join(script_dir, "benign_twins.jsonl")

    with open(seeds_path, "w", encoding="utf-8") as f:
        for seed in all_seeds:
            f.write(json.dumps(seed, ensure_ascii=False) + "\n")

    with open(twins_path, "w", encoding="utf-8") as f:
        for twin in all_twins:
            f.write(json.dumps(twin, ensure_ascii=False) + "\n")

    # ── Step 5: Summary ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Seeds written:  {seeds_path} ({len(all_seeds)} records)")
    print(f"  Twins written:  {twins_path} ({len(all_twins)} records)")
    print(f"\n  Category breakdown:")
    for cat_code in CATEGORIES:
        n_seeds = sum(1 for s in all_seeds if s["category"] == cat_code)
        n_twins = sum(1 for t in all_twins if t["category"] == cat_code)
        print(f"    {cat_code}: {n_seeds} seeds, {n_twins} twins")
    print(f"\n  Source models: {set(CATEGORY_MODELS.values())}")
    print(f"\nNext: Upload these to Google Drive, then run NB2 from Cell 2 (dedup).")


if __name__ == "__main__":
    main()
