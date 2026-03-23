#!/usr/bin/env python3
"""
Aggregate timing reports into a computational cost summary.

Reads results/timing_report.json (accumulated by timed_phase across all scripts)
and produces a formatted summary table suitable for paper inclusion.

Usage:
    python scripts/09_cost_report.py
    python scripts/09_cost_report.py --timing-file results/timing_report.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.timing import compute_gpu_hours
from medics.utils import save_json


def main():
    parser = argparse.ArgumentParser(description="MediCS computational cost report")
    parser.add_argument("--timing-file", default="results/timing_report.json",
                        help="Path to accumulated timing report")
    parser.add_argument("--output", default="results/computational_cost.json",
                        help="Output path for cost summary")
    args = parser.parse_args()

    timing_path = Path(args.timing_file)
    if not timing_path.exists():
        print(f"ERROR: Timing report not found at {timing_path}")
        print("       Timing data is recorded automatically during pipeline execution.")
        return

    with open(timing_path) as f:
        timing_data = json.load(f)

    if not timing_data:
        print("WARNING: Timing report is empty.")
        return

    summary = compute_gpu_hours(timing_data)

    # Print LaTeX-friendly table
    print(f"\n{'='*60}")
    print(f"  COMPUTATIONAL COST REPORT")
    print(f"{'='*60}")
    print(f"  Total wall-clock time: {summary['total_wall_clock_hours']:.2f} hours")
    print(f"  Total GPU time:        {summary['total_gpu_hours']:.2f} hours")
    print(f"  Training GPU time:     {summary['training_hours']:.2f} hours")
    print(f"  Inference GPU time:    {summary['inference_hours']:.2f} hours")
    print(f"\n  Per-phase breakdown:")
    print(f"  {'Phase':<40} {'Hours':>8} {'GPU':>5}")
    print(f"  {'-'*40} {'-'*8} {'-'*5}")
    for phase in summary["per_phase"]:
        gpu_str = "yes" if phase["gpu"] else "no"
        print(f"  {phase['phase']:<40} {phase['duration_hours']:>8.3f} {gpu_str:>5}")
    print(f"{'='*60}")

    # Save
    save_json(summary, args.output)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
