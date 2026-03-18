#!/usr/bin/env python3
"""
Full evaluation pipeline. No GPU needed — reads saved inference results.

Multi-seed evaluation: expects results at results/eval/{ckpt}/seed_{seed}/held_out.jsonl
If multi-seed results not found, falls back to results/eval/{ckpt}/held_out.jsonl (single run).

Also supports --judge-helpfulness to run GPT-4o helpfulness judging on benign results.
Also supports --judge-transfer to run GPT-4o safety judging on transfer results.

Usage:
  python scripts/04_evaluate.py --checkpoints base,sft,dpo --seeds 42,123,456
  python scripts/04_evaluate.py --judge-helpfulness --checkpoint sft --input results/eval/sft/benign_results.jsonl
  python scripts/04_evaluate.py --judge-transfer --input results/transfer/mistral_results.jsonl
"""

import argparse
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from medics.metrics import (
    compute_all_metrics, mcnemar_test,
    compute_per_category_asr, compute_per_strategy_asr, compute_per_language_asr,
)
from medics.judge import judge_response_batch, judge_helpfulness_batch
from medics.utils import load_jsonl, save_jsonl, save_json


def _match_results_by_id(results_a, results_b):
    """
    Match results by seed_id + language for paired comparison (McNemar).
    Returns paired lists of (a_correct, b_correct) for entries present in both.
    """
    key_fn = lambda r: (r.get("seed_id", ""), r.get("language", ""))
    b_map = {}
    for r in results_b:
        k = key_fn(r)
        if k not in b_map:  # first occurrence wins
            b_map[k] = r

    a_correct = []
    b_correct = []
    matched = 0
    for r in results_a:
        k = key_fn(r)
        if k in b_map:
            a_correct.append(r.get("judge_label") == "safe")
            b_correct.append(b_map[k].get("judge_label") == "safe")
            matched += 1

    if matched < len(results_a) or matched < len(results_b):
        print(f"  McNemar: matched {matched} entries "
              f"(A={len(results_a)}, B={len(results_b)})")
    return a_correct, b_correct


def evaluate_checkpoints(checkpoints, seeds):
    """Main evaluation across checkpoints and seeds."""
    all_results = {}

    for ckpt in checkpoints:
        seed_metrics = []

        for seed in seeds:
            # Try multi-seed path first, fall back to single path
            seed_dir = Path(f"results/eval/{ckpt}/seed_{seed}")
            single_dir = Path(f"results/eval/{ckpt}")

            if seed_dir.exists():
                attack_path = seed_dir / "held_out.jsonl"
                benign_path = seed_dir / "benign_results.jsonl"
            else:
                attack_path = single_dir / "held_out.jsonl"
                benign_path = single_dir / "benign_results.jsonl"
                if len(seeds) > 1:
                    print(f"  NOTE: Using single-run results for {ckpt} "
                          f"(multi-seed dir seed_{seed} not found)")

            attack_results = load_jsonl(attack_path)
            benign_results = load_jsonl(benign_path)

            if not attack_results:
                print(f"WARNING: No attack results for {ckpt} seed={seed}")
                continue

            metrics = compute_all_metrics(
                attack_results, benign_results, label=f"{ckpt}_seed{seed}",
                bootstrap_seed=seed,
            )
            seed_metrics.append(metrics)

        if not seed_metrics:
            print(f"WARNING: No valid results for {ckpt}")
            continue

        # Aggregate across seeds
        asr_values = [m["asr"] for m in seed_metrics]
        hr_values = [m["helpfulness_retention"] for m in seed_metrics]
        frr_values = [m["false_refusal_rate"] for m in seed_metrics]

        aggregated = {
            "label": ckpt,
            "asr_mean": float(np.mean(asr_values)),
            "asr_std": float(np.std(asr_values)) if len(asr_values) > 1 else 0.0,
            "asr_per_seed": {str(s): m["asr"] for s, m in zip(seeds, seed_metrics)},
            # NOTE: CI is from first seed's within-seed bootstrap, not pooled across seeds.
            # Cross-seed variance is captured by asr_std. For pooled CI, aggregate raw
            # results across seeds and bootstrap once — not implemented (marginal gain).
            "asr_ci": seed_metrics[0]["asr_ci"],
            "hr_mean": float(np.mean(hr_values)),
            "hr_std": float(np.std(hr_values)) if len(hr_values) > 1 else 0.0,
            "hr_ci": seed_metrics[0].get("hr_ci", (0.0, 0.0)),
            "frr_mean": float(np.mean(frr_values)),
            "frr_std": float(np.std(frr_values)) if len(frr_values) > 1 else 0.0,
            "frr_ci": seed_metrics[0].get("frr_ci", (0.0, 0.0)),
            "attack_judge_error_rate": seed_metrics[0].get("attack_judge_error_rate", 0.0),
            "benign_judge_error_rate": seed_metrics[0].get("benign_judge_error_rate", 0.0),
            "n_seeds": len(seed_metrics),
            "n_attacks": seed_metrics[0]["n_attacks"],
            "n_benign": seed_metrics[0]["n_benign"],
        }
        all_results[ckpt] = aggregated

        # Print summary
        print(f"\n{'='*50}")
        print(f"  {ckpt.upper()} (across {len(seed_metrics)} seed(s))")
        print(f"  ASR:  {aggregated['asr_mean']:.1%} "
              f"± {aggregated['asr_std']:.1%} "
              f"({aggregated['asr_ci'][0]:.1%} - {aggregated['asr_ci'][1]:.1%})")
        print(f"  HR:   {aggregated['hr_mean']:.1%} "
              f"± {aggregated['hr_std']:.1%}")
        print(f"  FRR:  {aggregated['frr_mean']:.1%}")
        if aggregated['attack_judge_error_rate'] > 0:
            print(f"  Judge error rate: {aggregated['attack_judge_error_rate']:.1%}")
        print(f"  N:    {aggregated['n_attacks']} attacks, "
              f"{aggregated['n_benign']} benign")

        # Per-category/strategy/language breakdown (from first seed)
        attack_results_first = load_jsonl(
            Path(f"results/eval/{ckpt}/seed_{seeds[0]}/held_out.jsonl")
            if Path(f"results/eval/{ckpt}/seed_{seeds[0]}").exists()
            else Path(f"results/eval/{ckpt}/held_out.jsonl")
        )
        if attack_results_first:
            cat_asr = compute_per_category_asr(attack_results_first)
            print(f"  Per-category ASR:")
            for cat, asr in sorted(cat_asr.items()):
                print(f"    {cat}: {asr:.1%}")

            strat_asr = compute_per_strategy_asr(attack_results_first)
            if strat_asr:
                print(f"  Per-strategy ASR:")
                for strat, asr in sorted(strat_asr.items()):
                    print(f"    {strat}: {asr:.1%}")

            lang_asr = compute_per_language_asr(attack_results_first)
            if lang_asr:
                print(f"  Per-language ASR:")
                for lang, asr in sorted(lang_asr.items()):
                    print(f"    {lang}: {asr:.1%}")

    # McNemar's test: base vs dpo (matched by seed_id + language)
    if "base" in all_results and "dpo" in all_results:
        base_results = load_jsonl(
            _find_results_path("base", seeds[0], "held_out.jsonl")
        )
        dpo_results = load_jsonl(
            _find_results_path("dpo", seeds[0], "held_out.jsonl")
        )
        base_correct, dpo_correct = _match_results_by_id(base_results, dpo_results)
        if base_correct:
            p = mcnemar_test(base_correct, dpo_correct)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"\nMcNemar's p-value (base vs DPO): {p:.4f} {sig}")

    # Also test base vs sft
    if "base" in all_results and "sft" in all_results:
        base_results = load_jsonl(
            _find_results_path("base", seeds[0], "held_out.jsonl")
        )
        sft_results = load_jsonl(
            _find_results_path("sft", seeds[0], "held_out.jsonl")
        )
        base_correct, sft_correct = _match_results_by_id(base_results, sft_results)
        if base_correct:
            p = mcnemar_test(base_correct, sft_correct)
            sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'
            print(f"McNemar's p-value (base vs SFT): {p:.4f} {sig}")

    # Save summary
    Path("results/eval").mkdir(parents=True, exist_ok=True)
    save_json(all_results, "results/eval/summary.json")
    print(f"\nResults saved to results/eval/summary.json")


def _find_results_path(ckpt, seed, filename):
    """Find results file, preferring multi-seed path."""
    seed_path = Path(f"results/eval/{ckpt}/seed_{seed}/{filename}")
    single_path = Path(f"results/eval/{ckpt}/{filename}")
    return str(seed_path if seed_path.exists() else single_path)


def judge_helpfulness_cmd(args):
    """Judge helpfulness of benign evaluation results via GPT-4o."""
    results = load_jsonl(args.input)
    if not results:
        print(f"ERROR: No results at {args.input}")
        return

    print(f"Judging helpfulness for {len(results)} benign results...")
    judged = judge_helpfulness_batch(results)

    output = args.output or args.input.replace(".jsonl", "_judged.jsonl")
    save_jsonl(judged, output)
    print(f"Judged results saved: {output}")

    # Quick summary
    valid = [r for r in judged if not r.get("judge_error")]
    n_helpful = sum(1 for r in valid if not r.get("was_incorrectly_refused"))
    n_refused = sum(1 for r in valid if r.get("was_incorrectly_refused"))
    if valid:
        print(f"  Helpful: {n_helpful}/{len(valid)} ({n_helpful/len(valid):.1%})")
        print(f"  Over-refused: {n_refused}/{len(valid)} ({n_refused/len(valid):.1%})")
    else:
        print("  WARNING: All results had judge errors — no valid helpfulness data")


def judge_transfer_cmd(args):
    """Judge transfer evaluation results via GPT-4o safety judge."""
    results = load_jsonl(args.input)
    if not results:
        print(f"ERROR: No results at {args.input}")
        return

    print(f"Judging safety for {len(results)} transfer results...")
    judged = judge_response_batch(results)

    output = args.output or args.input.replace(".jsonl", "_judged.jsonl")
    save_jsonl(judged, output)

    # Quick ASR summary
    from medics.metrics import compute_asr
    asr = compute_asr(judged)
    print(f"Transfer ASR: {asr:.1%}")


def main():
    parser = argparse.ArgumentParser(description="MediCS evaluation pipeline")
    parser.add_argument("--checkpoints", default="base,sft,dpo",
                        help="Comma-separated checkpoint names")
    parser.add_argument("--seeds", default="42,123,456",
                        help="Comma-separated random seeds")
    # Helpfulness judging mode
    parser.add_argument("--judge-helpfulness", action="store_true",
                        help="Run GPT-4o helpfulness judging on benign results")
    # Transfer judging mode
    parser.add_argument("--judge-transfer", action="store_true",
                        help="Run GPT-4o safety judging on transfer results")
    parser.add_argument("--input", help="Input JSONL for judging modes")
    parser.add_argument("--output", help="Output JSONL for judging modes")
    args = parser.parse_args()

    if args.judge_helpfulness:
        if not args.input:
            print("ERROR: --input required for --judge-helpfulness")
            return
        judge_helpfulness_cmd(args)
    elif args.judge_transfer:
        if not args.input:
            print("ERROR: --input required for --judge-transfer")
            return
        judge_transfer_cmd(args)
    else:
        checkpoints = args.checkpoints.split(",")
        seeds = [int(s) for s in args.seeds.split(",")]
        evaluate_checkpoints(checkpoints, seeds)


if __name__ == "__main__":
    main()
