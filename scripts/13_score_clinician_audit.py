#!/usr/bin/env python3
"""
Score completed clinician-audit CSVs against the blinded-key file.
"""

import argparse
import csv
from itertools import combinations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.audit import parse_binary_label, cohens_kappa_binary, agreement_summary
from medics.utils import save_json


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _annotation_map(paths: list[Path]) -> tuple[dict, dict]:
    item_to_raters = {}
    completion = {}
    for path in paths:
        rows = _read_csv(path)
        inferred_rater = path.stem
        filled = 0
        total = 0
        for row in rows:
            total += 1
            label = parse_binary_label(row.get("human_label"))
            if label is None:
                continue
            filled += 1
            rater_id = (row.get("rater_id") or inferred_rater).strip() or inferred_rater
            item_id = row["audit_item_id"]
            item_to_raters.setdefault(item_id, {})
            item_to_raters[item_id][rater_id] = label
        completion[inferred_rater] = {"filled": filled, "total": total}
    return item_to_raters, completion


def _pairwise_kappas(annotation_map: dict) -> dict:
    raters = sorted({rater for item in annotation_map.values() for rater in item})
    results = {}
    for a, b in combinations(raters, 2):
        common = []
        for item in annotation_map.values():
            if a in item and b in item:
                common.append((item[a], item[b]))
        if not common:
            results[f"{a}_vs_{b}"] = {"n": 0, "kappa": 0.0}
            continue
        labels_a = [x for x, _ in common]
        labels_b = [y for _, y in common]
        results[f"{a}_vs_{b}"] = {
            "n": len(common),
            "kappa": round(cohens_kappa_binary(labels_a, labels_b), 4),
        }
    return results


def _write_item_summary(path: Path, key_rows: list[dict], annotation_map: dict):
    raters = sorted({rater for item in annotation_map.values() for rater in item})
    rows = []
    for key in key_rows:
        item_id = key["audit_item_id"]
        annotations = annotation_map.get(item_id, {})
        row = {
            "audit_item_id": item_id,
            "audit_group_id": key["audit_group_id"],
            "checkpoint": key["checkpoint"],
            "category": key["category"],
            "language": key["language"],
            "llm_judge_label": key["llm_judge_label"],
        }
        for rater in raters:
            row[f"label_{rater}"] = annotations.get(rater, "")
        rows.append(row)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Score clinician audit annotations")
    parser.add_argument("--key", required=True, help="Path to clinician_audit_key.csv")
    parser.add_argument("--annotations", nargs="+", required=True,
                        help="Completed clinician packet CSVs, one per rater")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    key_path = Path(args.key)
    key_rows = _read_csv(key_path)
    annotation_paths = [Path(p) for p in args.annotations]
    annotation_map, completion = _annotation_map(annotation_paths)
    pairwise = _pairwise_kappas(annotation_map)
    agreement = agreement_summary(key_rows, annotation_map)

    summary = {
        "key_path": str(key_path),
        "annotation_files": [str(p) for p in annotation_paths],
        "completion_by_file": completion,
        "pairwise_kappa": pairwise,
        "judge_agreement": agreement,
    }

    output_json = Path(args.output_json or key_path.with_name("clinician_audit_summary.json"))
    output_csv = Path(args.output_csv or key_path.with_name("clinician_audit_item_summary.csv"))
    save_json(summary, output_json)
    _write_item_summary(output_csv, key_rows, annotation_map)

    print(f"Saved summary JSON: {output_json}")
    print(f"Saved item CSV:     {output_csv}")
    print("Pairwise kappa:")
    for pair, values in pairwise.items():
        print(f"  {pair}: n={values['n']}, kappa={values['kappa']:.4f}")
    overall = agreement["overall"]
    print(
        "Judge agreement (majority vote vs LLM): "
        f"n={overall['n']}, accuracy={overall['accuracy']:.4f}, "
        f"kappa={overall['kappa']:.4f}"
    )


if __name__ == "__main__":
    main()
