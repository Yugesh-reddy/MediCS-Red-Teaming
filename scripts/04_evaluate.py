#!/usr/bin/env python3
"""
Full evaluation pipeline. No GPU needed — reads saved inference results.

Multi-seed evaluation: expects results at results/eval/{ckpt}/seed_{seed}/held_out.jsonl
If multi-seed results not found, falls back to results/eval/{ckpt}/held_out.jsonl (single run).

Also supports --judge-helpfulness to run GPT-5 helpfulness judging on benign results.
Also supports --judge-transfer to run GPT-5 safety judging on transfer results.

Usage:
  python scripts/04_evaluate.py --checkpoints base,sft,dpo --seeds 42,123,456
  python scripts/04_evaluate.py --judge-helpfulness --checkpoint sft --input results/eval/sft/benign_results.jsonl
  python scripts/04_evaluate.py --judge-transfer --input results/transfer/mistral_results.jsonl
"""

import argparse
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from medics.metrics import (
    compute_all_metrics, mcnemar_test,
    compute_per_category_asr, compute_per_strategy_asr, compute_per_language_asr,
    compute_effect_sizes, compute_residual_failure_breakdown,
    paired_bootstrap_delta_ci, holm_bonferroni,
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


def _check_prompt_mode_consistency(all_loaded):
    """Warn if results being compared used different system prompt modes."""
    modes_by_ckpt = {}
    for ckpt, results in all_loaded.items():
        modes = {r.get("system_prompt_mode", "unknown") for r in results}
        modes_by_ckpt[ckpt] = modes

    all_modes = set()
    for modes in modes_by_ckpt.values():
        all_modes.update(modes)

    has_unknown = "unknown" in all_modes
    has_mixed = len(all_modes) > 1

    if has_unknown:
        unknown_ckpts = [c for c, m in modes_by_ckpt.items() if "unknown" in m]
        print(f"\n  WARNING: Results for {', '.join(unknown_ckpts)} have no "
              f"system_prompt_mode field (generated before prompt tracking).")
        print(f"  Cannot verify prompt consistency. Consider re-running inference.\n")
    if has_mixed:
        print(f"\n  WARNING: Comparing results across different system prompt modes:")
        for ckpt, modes in modes_by_ckpt.items():
            print(f"    {ckpt}: {', '.join(sorted(modes))}")
        print(f"  ASR deltas may reflect prompt choice, not model/attack behavior.")
        print(f"  Use --prompt-ablation for controlled comparison, or re-run with")
        print(f"  --system-prompt base|defense to hold prompt constant.\n")


def evaluate_checkpoints(checkpoints, seeds):
    """Main evaluation across checkpoints and seeds."""
    all_results = {}
    all_loaded = {}  # ckpt -> first-seed raw results, for prompt mode check

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

            # Stash first seed's results for prompt mode consistency check
            if ckpt not in all_loaded:
                all_loaded[ckpt] = attack_results

            metrics = compute_all_metrics(
                attack_results, benign_results, label=f"{ckpt}_seed{seed}",
                bootstrap_seed=seed,
            )
            metrics["_seed"] = seed
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
            "asr_per_seed": {str(m["_seed"]): m["asr"] for m in seed_metrics},
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

    # Check for prompt mode confound across checkpoints
    if len(all_loaded) > 1:
        _check_prompt_mode_consistency(all_loaded)

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

    # Effect sizes (Cohen's h) for all checkpoint pairs
    asr_dict = {ckpt: data["asr_mean"] for ckpt, data in all_results.items()}
    if len(asr_dict) >= 2:
        effect_sizes = compute_effect_sizes(asr_dict)
        print(f"\nEffect Sizes (Cohen's h):")
        for pair, es in effect_sizes.items():
            print(f"  {pair}: h={es['cohens_h']:.3f} ({es['interpretation']})")
        all_results["effect_sizes"] = effect_sizes

    # Paired bootstrap on ΔASR (base→SFT, base→DPO, SFT→DPO), aggregated
    # across all requested seed runs instead of only the first seed.
    paired = _paired_delta_asr_all_pairs(checkpoints, seeds)
    if paired:
        print("\nPaired bootstrap ΔASR (95% CI, base 0 = no change):")
        for pair, stats in paired.items():
            lo, hi = stats["ci"]
            sig = "*" if lo * hi > 0 else "ns"
            print(
                f"  {pair}: Δ={stats['delta_mean']:+.1%}  CI=[{lo:+.1%}, {hi:+.1%}]  "
                f"n={stats['n']}  {sig}"
            )
        all_results["paired_bootstrap_delta_asr"] = paired

    # Holm-Bonferroni correction over per-language McNemar tests, aggregated
    # across all requested seed runs instead of only the first seed.
    per_lang_mcnemar = _per_language_mcnemar(checkpoints, seeds)
    if per_lang_mcnemar:
        for contrast, pvals in per_lang_mcnemar.items():
            if not pvals:
                continue
            adj = holm_bonferroni(
                {lang: info["p_value"] for lang, info in pvals.items()},
                alpha=0.05,
            )
            print(f"\nPer-language McNemar, Holm-Bonferroni corrected ({contrast}):")
            for lang, info in sorted(adj["results"].items(),
                                     key=lambda x: x[1]["rank"]):
                support = pvals[lang]["n"]
                marker = "**" if info["reject_null"] else "ns"
                print(f"  {lang}: p_raw={info['p_raw']:.4f}  "
                      f"thresh={info['adj_threshold']:.4f}  n={support}  {marker}")
        all_results["per_language_mcnemar_holm"] = {
            contrast: {
                **holm_bonferroni(
                    {lang: info["p_value"] for lang, info in pvals.items()},
                    alpha=0.05,
                ),
                "support": {
                    lang: {
                        "n": info["n"],
                        "seeds_used": info["seeds_used"],
                    }
                    for lang, info in pvals.items()
                },
            }
            for contrast, pvals in per_lang_mcnemar.items() if pvals
        }

    # Save summary
    Path("results/eval").mkdir(parents=True, exist_ok=True)
    save_json(all_results, "results/eval/summary.json")
    print(f"\nResults saved to results/eval/summary.json")


def _load_results_across_seeds(ckpt, seeds, filename):
    """Load unique result files across requested seeds without double-counting fallbacks."""
    loaded = []
    seen_paths = set()
    for seed in seeds:
        path = _find_results_path(ckpt, seed, filename)
        if not Path(path).exists() or path in seen_paths:
            continue
        seen_paths.add(path)
        rows = load_jsonl(path)
        if rows:
            loaded.append((seed, rows, path))
    return loaded


def _paired_delta_asr_all_pairs(checkpoints, seeds):
    """Compute paired-bootstrap ΔASR CIs for every ordered checkpoint pair."""
    pairs = {}
    order = ["base", "sft", "dpo"]
    ckpt_order = [c for c in order if c in checkpoints]
    for i, a in enumerate(ckpt_order):
        for b in ckpt_order[i + 1:]:
            a_runs = {
                seed: rows
                for seed, rows, _ in _load_results_across_seeds(a, seeds, "held_out.jsonl")
            }
            b_runs = {
                seed: rows
                for seed, rows, _ in _load_results_across_seeds(b, seeds, "held_out.jsonl")
            }
            before_harm = []
            after_harm = []
            seeds_used = []

            for seed in seeds:
                a_rows = a_runs.get(seed)
                b_rows = b_runs.get(seed)
                if not (a_rows and b_rows):
                    continue
                before_safe, after_safe = _match_results_by_id(a_rows, b_rows)
                if not before_safe:
                    continue
                before_harm.extend(0 if correct else 1 for correct in before_safe)
                after_harm.extend(0 if correct else 1 for correct in after_safe)
                seeds_used.append(seed)

            if not before_harm:
                continue
            pair_stats = paired_bootstrap_delta_ci(before_harm, after_harm, seed=42)
            pair_stats["seeds_used"] = seeds_used
            pairs[f"{a}_to_{b}"] = pair_stats
    return pairs


def _per_language_mcnemar(checkpoints, seeds):
    """Compute per-language McNemar p-values for base→SFT and base→DPO."""
    contrasts = {}
    if "base" not in checkpoints:
        return contrasts

    base_runs = _load_results_across_seeds("base", seeds, "held_out.jsonl")
    if not base_runs:
        return contrasts

    for target in ("sft", "dpo"):
        if target not in checkpoints:
            continue
        target_runs = {
            seed: rows
            for seed, rows, _ in _load_results_across_seeds(target, seeds, "held_out.jsonl")
        }
        if not target_runs:
            continue

        per_lang = {}
        all_langs = sorted({
            row.get("language")
            for _, rows, _ in base_runs
            for row in rows
            if row.get("language")
        })

        for lang in all_langs:
            b_correct = []
            t_correct = []
            seeds_used = []

            for seed, base_rows, _ in base_runs:
                target_rows = target_runs.get(seed)
                if not target_rows:
                    continue
                base_subset = [row for row in base_rows if row.get("language") == lang]
                tgt_subset = [row for row in target_rows if row.get("language") == lang]
                seed_b_correct, seed_t_correct = _match_results_by_id(base_subset, tgt_subset)
                if not seed_b_correct:
                    continue
                b_correct.extend(seed_b_correct)
                t_correct.extend(seed_t_correct)
                seeds_used.append(seed)

            if len(b_correct) < 10:
                continue
            per_lang[lang] = {
                "p_value": mcnemar_test(b_correct, t_correct),
                "n": len(b_correct),
                "seeds_used": seeds_used,
            }
        contrasts[f"base_vs_{target}"] = per_lang
    return contrasts


def _find_results_path(ckpt, seed, filename):
    """Find results file, preferring multi-seed path."""
    seed_path = Path(f"results/eval/{ckpt}/seed_{seed}/{filename}")
    single_path = Path(f"results/eval/{ckpt}/{filename}")
    return str(seed_path if seed_path.exists() else single_path)


def judge_helpfulness_cmd(args):
    """Judge helpfulness of benign evaluation results via GPT-5."""
    results = load_jsonl(args.input)
    if not results:
        print(f"ERROR: No results at {args.input}")
        return

    print(f"Judging helpfulness for {len(results)} benign results...")
    judged = judge_helpfulness_batch(results)

    output = args.output or args.input
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


def judge_attacks_cmd(args):
    """Judge harmfulness of held-out attack evaluation results via GPT-5.

    Writes judge_label / judge_method / judge_rationale back to the same file
    (matches the --judge-helpfulness in-place pattern), so downstream metric
    evaluation (compute_asr, McNemar, Cohen's h) picks them up without any
    path changes.
    """
    results = load_jsonl(args.input)
    if not results:
        print(f"ERROR: No results at {args.input}")
        return

    # Skip rows that already have a valid attack judgment — lets the cell be
    # re-run safely after a partial failure without paying API cost twice.
    needs_judging = [
        r for r in results
        if r.get("judge_label") not in ("harmful", "safe")
    ]
    already_judged = len(results) - len(needs_judging)
    if already_judged:
        print(f"  {already_judged}/{len(results)} already judged — skipping those")

    if not needs_judging:
        print("  Nothing to judge.")
        return

    print(f"Judging harmfulness for {len(needs_judging)} attack responses...")
    judged_subset = judge_response_batch(needs_judging)

    # Merge judged subset back by id (fall back to attack_prompt if no id).
    def _row_key(row):
        return (
            row.get("seed_id", ""),
            row.get("strategy", ""),
            row.get("language", ""),
            row.get("attack_prompt", ""),
        )
    judged_by_key = {_row_key(r): r for r in judged_subset}
    merged = [judged_by_key.get(_row_key(r), r) for r in results]

    output = args.output or args.input
    save_jsonl(merged, output)
    print(f"Judged attack results saved: {output}")

    from medics.metrics import compute_asr, compute_judge_error_rate
    asr = compute_asr(merged)
    err = compute_judge_error_rate(merged)
    print(f"  ASR: {asr:.1%}  |  Judge error rate: {err:.1%}")


def judge_transfer_cmd(args):
    """Judge transfer evaluation results via GPT-5 safety judge."""
    results = load_jsonl(args.input)
    if not results:
        print(f"ERROR: No results at {args.input}")
        return

    print(f"Judging safety for {len(results)} transfer results...")
    judged = judge_response_batch(results)

    output = args.output or args.input.replace(".jsonl", "_judged.jsonl")
    save_jsonl(judged, output)

    # Quick ASR summary
    from medics.metrics import compute_asr, compute_judge_error_rate, compute_judge_fallback_rate
    asr = compute_asr(judged)
    print(f"Transfer ASR: {asr:.1%}")
    print(f"Judge error rate: {compute_judge_error_rate(judged):.1%}")
    fallback = compute_judge_fallback_rate(judged)
    if fallback > 0:
        n_fallback = sum(
            1 for r in judged if str(r.get("judge_method", "")).startswith("heuristic_")
        )
        if n_fallback == 0:
            n_fallback = sum(1 for r in judged if r.get("judge_fallback"))
        print(f"Judge fallback rate: {fallback:.1%} ({n_fallback}/{len(judged)})")
    method_counts = Counter(r.get("judge_method", "unknown") for r in judged)
    print(f"Judge method breakdown: {dict(method_counts)}")


def residual_analysis_cmd(args):
    """Analyze residual failures in the DPO (or specified) checkpoint."""
    checkpoint = args.checkpoint or "dpo"
    seed = int(args.seeds.split(",")[0]) if args.seeds else 42

    results_path = _find_results_path(checkpoint, seed, "held_out.jsonl")
    results = load_jsonl(results_path)
    if not results:
        print(f"ERROR: No results at {results_path}")
        return

    print(f"\nResidual Failure Analysis: {checkpoint} (seed={seed})")
    print(f"{'='*60}")

    breakdown = compute_residual_failure_breakdown(results)

    print(f"  Total valid: {breakdown['total_valid']}")
    print(f"  Total failures: {breakdown['total_failures']}")
    print(f"  Residual ASR: {breakdown['residual_asr']:.1%}")

    if breakdown.get("by_category"):
        print(f"\n  By Category:")
        for cat, info in sorted(breakdown["by_category"].items(),
                                key=lambda x: x[1]["asr"], reverse=True):
            print(f"    {cat}: ASR={info['asr']:.1%} "
                  f"({info['count']}/{info['total']})")

    if breakdown.get("by_language"):
        print(f"\n  By Language:")
        for lang, info in sorted(breakdown["by_language"].items(),
                                 key=lambda x: x[1]["asr"], reverse=True):
            print(f"    {lang}: ASR={info['asr']:.1%} "
                  f"({info['count']}/{info['total']})")

    if breakdown.get("by_strategy"):
        print(f"\n  By Strategy:")
        for strat, info in sorted(breakdown["by_strategy"].items(),
                                  key=lambda x: x[1]["asr"], reverse=True):
            print(f"    {strat}: ASR={info['asr']:.1%} "
                  f"({info['count']}/{info['total']})")

    if breakdown.get("hardest_pairs"):
        print(f"\n  Hardest (Category, Language) Pairs:")
        for pair in breakdown["hardest_pairs"][:10]:
            print(f"    ({pair['category']}, {pair['language']}): "
                  f"ASR={pair['asr']:.1%} ({pair['count']}/{pair['total']})")

    # Save
    Path("results/eval").mkdir(parents=True, exist_ok=True)
    save_json(breakdown, "results/eval/residual_analysis.json")
    print(f"\nSaved to results/eval/residual_analysis.json")


def prompt_ablation_cmd(args):
    """Print a 2x2 table: checkpoint × prompt mode, to separate prompt effect from training effect.

    Expects results at results/eval/{ckpt}_{prompt_mode}/seed_{seed}/held_out.jsonl
    e.g. results/eval/base_base/seed_42/held_out.jsonl
         results/eval/base_defense/seed_42/held_out.jsonl
         results/eval/sft_base/seed_42/held_out.jsonl
         results/eval/sft_defense/seed_42/held_out.jsonl
    """
    from medics.metrics import compute_asr

    checkpoints = args.checkpoints.split(",")
    prompt_modes = ["base", "defense"]
    seed = int(args.seeds.split(",")[0]) if args.seeds else 42

    # Collect ASR for each cell
    table = {}
    for ckpt in checkpoints:
        for pm in prompt_modes:
            label = f"{ckpt}_{pm}"
            path = _find_results_path(label, seed, "held_out.jsonl")
            results = load_jsonl(path)
            if results:
                asr = compute_asr(results)
                table[(ckpt, pm)] = asr
            else:
                table[(ckpt, pm)] = None

    # Print 2x2 table
    print(f"\n{'='*60}")
    print("  Prompt Ablation: Checkpoint × System Prompt")
    print(f"  (seed={seed})")
    print(f"{'='*60}")
    header = f"  {'Checkpoint':<20} {'BASE prompt':<15} {'DEFENSE prompt':<15} {'Δ (prompt effect)':<15}"
    print(header)
    print(f"  {'-'*65}")
    for ckpt in checkpoints:
        base_asr = table.get((ckpt, "base"))
        def_asr = table.get((ckpt, "defense"))
        base_str = f"{base_asr:.1%}" if base_asr is not None else "N/A"
        def_str = f"{def_asr:.1%}" if def_asr is not None else "N/A"
        if base_asr is not None and def_asr is not None:
            delta = base_asr - def_asr
            delta_str = f"{delta:+.1%}"
        else:
            delta_str = "N/A"
        print(f"  {ckpt:<20} {base_str:<15} {def_str:<15} {delta_str:<15}")

    # Training effect (column-wise delta)
    print(f"  {'-'*65}")
    if len(checkpoints) >= 2:
        for pm in prompt_modes:
            first = table.get((checkpoints[0], pm))
            last = table.get((checkpoints[-1], pm))
            if first is not None and last is not None:
                delta = first - last
                print(f"  Δ training effect ({pm} prompt): "
                      f"{checkpoints[0]} {first:.1%} → {checkpoints[-1]} {last:.1%} "
                      f"(reduction: {delta:+.1%})")

    print(f"\nWorkflow to generate these results:")
    print(f"  For each checkpoint C in {{{','.join(checkpoints)}}}:")
    print(f"    !python colab/run_inference.py --checkpoint $C --system-prompt base \\")
    print(f"        --output results/eval/${{C}}_base/seed_{seed}/held_out.jsonl ...")
    print(f"    !python colab/run_inference.py --checkpoint $C --system-prompt defense \\")
    print(f"        --output results/eval/${{C}}_defense/seed_{seed}/held_out.jsonl ...")


def main():
    parser = argparse.ArgumentParser(description="MediCS evaluation pipeline")
    parser.add_argument("--checkpoints", default="base,sft,dpo",
                        help="Comma-separated checkpoint names")
    parser.add_argument("--checkpoint", help="Single checkpoint for residual analysis")
    parser.add_argument("--seeds", default="42,123,456",
                        help="Comma-separated random seeds")
    # Helpfulness judging mode
    parser.add_argument("--judge-helpfulness", action="store_true",
                        help="Run GPT-5 helpfulness judging on benign results")
    # Attack judging mode (held-out harmfulness)
    parser.add_argument("--judge-attacks", action="store_true",
                        help="Run GPT-5 harmfulness judging on held-out attack results")
    # Transfer judging mode
    parser.add_argument("--judge-transfer", action="store_true",
                        help="Run GPT-5 safety judging on transfer results")
    # Residual failure analysis mode
    parser.add_argument("--residual-analysis", action="store_true",
                        help="Analyze residual failures after defense")
    # Prompt ablation mode
    parser.add_argument("--prompt-ablation", action="store_true",
                        help="Print 2x2 table: checkpoint × prompt mode (separate prompt vs training effect)")
    parser.add_argument("--input", help="Input JSONL for judging modes")
    parser.add_argument("--output", help="Output JSONL for judging modes")
    args = parser.parse_args()

    if args.prompt_ablation:
        prompt_ablation_cmd(args)
    elif args.judge_helpfulness:
        if not args.input:
            print("ERROR: --input required for --judge-helpfulness")
            return
        judge_helpfulness_cmd(args)
    elif args.judge_attacks:
        if not args.input:
            print("ERROR: --input required for --judge-attacks")
            return
        judge_attacks_cmd(args)
    elif args.judge_transfer:
        if not args.input:
            print("ERROR: --input required for --judge-transfer")
            return
        judge_transfer_cmd(args)
    elif args.residual_analysis:
        residual_analysis_cmd(args)
    else:
        checkpoints = args.checkpoints.split(",")
        seeds = [int(s) for s in args.seeds.split(",")]
        evaluate_checkpoints(checkpoints, seeds)


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("Evaluation"):
        main()
    save_timing_report()
