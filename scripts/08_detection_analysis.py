#!/usr/bin/env python3
"""
Post-process perplexity detection results. No GPU needed.

Reads results/analysis/perplexity_results.json (from colab/run_perplexity.py)
and produces a formatted detection summary.

Usage:
    python scripts/08_detection_analysis.py
    python scripts/08_detection_analysis.py --input results/analysis/perplexity_results.json
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.utils import load_json, save_json


def main():
    parser = argparse.ArgumentParser(description="Detection analysis post-processing")
    parser.add_argument("--input", default="results/analysis/perplexity_results.json")
    parser.add_argument("--output", default="results/analysis/detection_summary.json")
    args = parser.parse_args()

    data = load_json(args.input)
    if not data:
        print(f"ERROR: No data at {args.input}")
        print("       Run colab/run_perplexity.py first.")
        return

    auroc = data.get("auroc", 0.0)
    best_f1 = data.get("best_f1", 0.0)
    lang_det = data.get("per_language", {})

    print(f"\n{'='*60}")
    print(f"  PERPLEXITY DETECTION ANALYSIS")
    print(f"{'='*60}")
    print(f"  Overall AUROC:     {auroc:.3f}")
    print(f"  Overall Best F1:   {best_f1:.3f}")
    print(f"  Best Threshold:    {data.get('best_threshold', 'N/A')}")
    print(f"  Mean PPL (EN):     {data.get('mean_ppl_english', 'N/A')}")
    print(f"  Mean PPL (CS):     {data.get('mean_ppl_cs', 'N/A')}")

    # Interpretation
    if auroc >= 0.9:
        verdict = "HIGH — CS attacks are easily detectable by perplexity alone"
        implication = ("A simple perplexity filter could catch most CS attacks. "
                       "However, this is a detection-level defense that doesn't fix "
                       "the underlying vulnerability. Weight-level defense (SFT+DPO) "
                       "addresses the root cause.")
    elif auroc >= 0.7:
        verdict = "MODERATE — partial detection possible but not reliable"
        implication = ("Perplexity provides some signal but insufficient for reliable "
                       "detection. This supports the need for weight-level defense.")
    else:
        verdict = "LOW — CS attacks are indistinguishable from normal multilingual input"
        implication = ("CS attacks cannot be trivially detected, making detection-based "
                       "defense impractical. Weight-level defense is necessary.")

    print(f"\n  Detection Capability: {verdict}")
    print(f"  Implication: {implication}")

    if lang_det:
        print(f"\n  Per-Language Detection:")
        print(f"  {'Language':<10} {'AUROC':>8} {'F1':>8} {'PPL_en':>8} {'PPL_cs':>8}")
        print(f"  {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
        for lang, info in sorted(lang_det.items(),
                                  key=lambda x: x[1].get("auroc", 0), reverse=True):
            print(f"  {lang:<10} {info.get('auroc', 0):.3f}"
                  f"    {info.get('best_f1', 0):.3f}"
                  f"    {info.get('mean_ppl_en', 0):>6.0f}"
                  f"    {info.get('mean_ppl_cs', 0):>6.0f}")

    print(f"{'='*60}")

    # Save summary
    summary = {
        "auroc": auroc,
        "best_f1": best_f1,
        "verdict": verdict,
        "implication": implication,
        "per_language": lang_det,
    }
    save_json(summary, args.output)
    print(f"\nSummary saved to {args.output}")


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("Detection Analysis"):
        main()
    save_timing_report()
