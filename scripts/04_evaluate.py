#!/usr/bin/env python3
"""
Full evaluation pipeline. No GPU needed — reads saved inference results.

Usage:
  python scripts/04_evaluate.py --checkpoints base,sft,dpo --seeds 42,123,456
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.metrics import (
    compute_all_metrics, mcnemar_test,
    compute_per_category_asr, compute_per_strategy_asr, compute_per_language_asr,
)
from medics.utils import load_jsonl, save_json


def main():
    parser = argparse.ArgumentParser(description="MediCS evaluation pipeline")
    parser.add_argument("--checkpoints", default="base,sft,dpo",
                        help="Comma-separated checkpoint names")
    parser.add_argument("--seeds", default="42,123,456",
                        help="Comma-separated random seeds")
    args = parser.parse_args()

    checkpoints = args.checkpoints.split(",")
    seeds = [int(s) for s in args.seeds.split(",")]

    all_results = {}
    for ckpt in checkpoints:
        attack_results = load_jsonl(f"results/eval/{ckpt}/held_out.jsonl")
        benign_results = load_jsonl(f"results/eval/{ckpt}/benign_results.jsonl")

        if not attack_results:
            print(f"WARNING: No attack results for {ckpt}")
            continue

        metrics = compute_all_metrics(attack_results, benign_results, label=ckpt)
        all_results[ckpt] = metrics

        print(f"\n{'='*50}")
        print(f"  {ckpt.upper()}")
        print(f"  ASR:  {metrics['asr']:.1%} "
              f"({metrics['asr_ci'][0]:.1%} - {metrics['asr_ci'][1]:.1%})")
        print(f"  HR:   {metrics['helpfulness_retention']:.1%}")
        print(f"  FRR:  {metrics['false_refusal_rate']:.1%}")
        print(f"  N:    {metrics['n_attacks']} attacks, {metrics['n_benign']} benign")

        # Per-category breakdown
        cat_asr = compute_per_category_asr(attack_results)
        print(f"  Per-category ASR:")
        for cat, asr in sorted(cat_asr.items()):
            print(f"    {cat}: {asr:.1%}")

        # Per-strategy breakdown
        strat_asr = compute_per_strategy_asr(attack_results)
        if strat_asr:
            print(f"  Per-strategy ASR:")
            for strat, asr in sorted(strat_asr.items()):
                print(f"    {strat}: {asr:.1%}")

        # Per-language breakdown
        lang_asr = compute_per_language_asr(attack_results)
        if lang_asr:
            print(f"  Per-language ASR:")
            for lang, asr in sorted(lang_asr.items()):
                print(f"    {lang}: {asr:.1%}")

    # McNemar's test: base vs dpo
    if "base" in all_results and "dpo" in all_results:
        base_results = load_jsonl("results/eval/base/held_out.jsonl")
        dpo_results = load_jsonl("results/eval/dpo/held_out.jsonl")
        base_correct = [r.get("judge_label") == "safe" for r in base_results]
        dpo_correct = [r.get("judge_label") == "safe" for r in dpo_results]

        if len(base_correct) == len(dpo_correct):
            p = mcnemar_test(base_correct, dpo_correct)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"\nMcNemar's p-value (base vs DPO): {p:.4f} {sig}")

    # Save summary
    Path("results/eval").mkdir(parents=True, exist_ok=True)
    save_json(all_results, "results/eval/summary.json")
    print(f"\nResults saved to results/eval/summary.json")


if __name__ == "__main__":
    main()
