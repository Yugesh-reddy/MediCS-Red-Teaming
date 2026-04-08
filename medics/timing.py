"""
MediCS — Computational Cost Tracking
======================================
Context manager and utilities for tracking wall-clock time across pipeline phases.
Timing data is accumulated in a module-level log and saved to JSON for paper reporting.
"""

import time
import json
from pathlib import Path
from contextlib import contextmanager

# Module-level timing log — accumulated across the session
_timing_log = []


@contextmanager
def timed_phase(phase_name, metadata=None):
    """
    Context manager that records wall-clock time for a pipeline phase.

    Usage:
        with timed_phase("SFT Round 1 Training", {"gpu": True, "round": 1}):
            trainer.train()

    Args:
        phase_name: human-readable phase name (e.g., "Dataset Construction")
        metadata: optional dict with extra info (gpu=True/False, round number, etc.)
    """
    meta = metadata or {}
    start = time.time()
    print(f"[TIMING] Starting: {phase_name}")
    try:
        yield
    finally:
        elapsed = time.time() - start
        entry = {
            "phase": phase_name,
            "duration_sec": round(elapsed, 2),
            "duration_min": round(elapsed / 60, 2),
            "gpu": meta.get("gpu", False),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            **{k: v for k, v in meta.items() if k != "gpu"},
        }
        _timing_log.append(entry)
        print(f"[TIMING] Finished: {phase_name} ({elapsed:.1f}s / {elapsed/60:.1f}min)")


def save_timing_report(output_path="results/timing_report.json"):
    """
    Save accumulated timing data to JSON. Appends to existing file if present.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if path.exists() and path.stat().st_size > 0:
        with open(path) as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []

    combined = existing + _timing_log

    with open(path, "w") as f:
        json.dump(combined, f, indent=2)

    print(f"[TIMING] Report saved to {output_path} ({len(combined)} entries)")


def print_timing_summary():
    """Print formatted timing summary table."""
    if not _timing_log:
        print("[TIMING] No timing data recorded.")
        return

    print(f"\n{'='*60}")
    print(f"  TIMING SUMMARY ({len(_timing_log)} phases)")
    print(f"{'='*60}")
    print(f"  {'Phase':<35} {'Time':>8} {'GPU':>5}")
    print(f"  {'-'*35} {'-'*8} {'-'*5}")

    total_sec = 0
    gpu_sec = 0
    for entry in _timing_log:
        dur = entry["duration_sec"]
        is_gpu = entry.get("gpu", False)
        total_sec += dur
        if is_gpu:
            gpu_sec += dur
        time_str = f"{dur/60:.1f}m" if dur >= 60 else f"{dur:.1f}s"
        print(f"  {entry['phase']:<35} {time_str:>8} {'yes' if is_gpu else 'no':>5}")

    print(f"  {'-'*35} {'-'*8} {'-'*5}")
    print(f"  {'Total wall-clock':<35} {total_sec/60:.1f}m")
    print(f"  {'Total GPU time':<35} {gpu_sec/60:.1f}m")
    print(f"  {'Total GPU hours':<35} {gpu_sec/3600:.2f}h")
    print(f"{'='*60}\n")


def compute_gpu_hours(timing_data):
    """
    Aggregate timing data into GPU-hours breakdown.

    Args:
        timing_data: list of timing entry dicts (from saved JSON)

    Returns:
        dict with total_wall_clock_hours, total_gpu_hours, training_hours,
        inference_hours, per_phase breakdown
    """
    total_sec = sum(e["duration_sec"] for e in timing_data)
    gpu_sec = sum(e["duration_sec"] for e in timing_data if e.get("gpu"))

    training_keywords = ["SFT", "DPO", "Train", "train"]
    inference_keywords = ["Inference", "inference", "Perplexity", "Transfer"]

    training_sec = sum(
        e["duration_sec"] for e in timing_data
        if e.get("gpu") and any(kw in e["phase"] for kw in training_keywords)
    )
    inference_sec = sum(
        e["duration_sec"] for e in timing_data
        if e.get("gpu") and any(kw in e["phase"] for kw in inference_keywords)
    )

    return {
        "total_wall_clock_hours": round(total_sec / 3600, 2),
        "total_gpu_hours": round(gpu_sec / 3600, 2),
        "training_hours": round(training_sec / 3600, 2),
        "inference_hours": round(inference_sec / 3600, 2),
        "n_phases": len(timing_data),
        "per_phase": [
            {
                "phase": e["phase"],
                "duration_hours": round(e["duration_sec"] / 3600, 3),
                "gpu": e.get("gpu", False),
            }
            for e in timing_data
        ],
    }


def reset_timing_log():
    """Clear the module-level timing log (for testing)."""
    global _timing_log
    _timing_log = []
