#!/usr/bin/env python3
"""
Algorithmic fairness analysis for MediCS defense training.

Three modes:
  # Mode 1: Generate code-switched benign queries ($0, CPU)
  python scripts/10_fairness_analysis.py --generate-benign-cs --n-samples 100

  # Mode 2: Judge helpfulness of CS benign results (~$0.15 API)
  python scripts/10_fairness_analysis.py --judge-multilingual \
      --input results/fairness/dpo/seed_42/benign_cs_results.jsonl

  # Mode 3: Full fairness analysis (CPU, $0)
  python scripts/10_fairness_analysis.py --checkpoints base,sft,dpo --seeds 42,123,456
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.utils import (
    load_jsonl, save_jsonl, load_json, save_json, load_config,
    code_switch_prompt, TARGET_LANGUAGES, LANGUAGE_NAMES,
)
from medics.fairness import defense_equity_report, full_fairness_report


def _cs_benign_path(checkpoint, seed, judged=False):
    """Checkpoint-specific CS-benign artifact path."""
    filename = "benign_cs_results_judged.jsonl" if judged else "benign_cs_results.jsonl"
    return Path("results/fairness") / checkpoint / f"seed_{seed}" / filename


def _generate_benign_cs(cfg, n_samples, seed):
    """Mode 1: Generate code-switched benign queries from benign twins."""
    benign_twins = load_jsonl("data/seeds/benign_twins.jsonl")
    held_out = load_jsonl("data/splits/held_out_eval.jsonl")

    # Build keyword lookup: seed_id -> keywords list (from held-out eval)
    kw_lookup = {}
    for entry in held_out:
        sid = entry.get("seed_id", "")
        kws = entry.get("keywords_translated", [])
        if sid and kws and sid not in kw_lookup:
            kw_lookup[sid] = kws

    # Stratified sample by category
    rng = random.Random(seed)
    by_cat = defaultdict(list)
    for bt in benign_twins:
        cat = bt.get("category", "")
        sid = bt.get("seed_id", "")
        if sid in kw_lookup:  # only include if we have keywords
            by_cat[cat].append(bt)

    per_cat = max(1, n_samples // len(by_cat)) if by_cat else 0
    sampled = []
    for cat in sorted(by_cat):
        pool = by_cat[cat]
        rng.shuffle(pool)
        sampled.extend(pool[:per_cat])

    print(f"Sampled {len(sampled)} benign twins across {len(by_cat)} categories")

    # Generate CS variants for each sample x language
    languages = cfg.get("dataset", {}).get("languages", [])
    lang_codes = [l["code"] for l in languages] if languages else TARGET_LANGUAGES

    prompts = []
    for bt in sampled:
        sid = bt["seed_id"]
        keywords = kw_lookup.get(sid, [])
        benign_text = bt.get("benign_question", "")

        for lang in lang_codes:
            # Build a seed-like dict for code_switch_prompt
            seed_dict = {
                "prompt": benign_text,
                "seed_id": sid,
                "category": bt.get("category", ""),
            }
            cs_result = code_switch_prompt(seed_dict, keywords, lang)
            prompts.append({
                "seed_id": sid,
                "category": bt.get("category", ""),
                "language": lang,
                "original_prompt": benign_text,
                "attack_prompt": cs_result["attack_prompt"],
                "keywords_translated": keywords,
                "strategy": "CS",
                "is_benign_cs": True,
            })

    out_dir = Path("data/fairness")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "benign_cs_prompts.jsonl"
    save_jsonl(prompts, str(out_path))
    print(f"Saved {len(prompts)} CS benign prompts to {out_path}")
    print(f"  ({len(sampled)} seeds x {len(lang_codes)} languages)")


def _judge_multilingual(input_path):
    """Mode 2: Judge helpfulness of CS benign inference results."""
    from medics.judge import judge_helpfulness_batch
    from medics.utils import load_dotenv

    load_dotenv()

    results = load_jsonl(input_path)
    print(f"Loaded {len(results)} results from {input_path}")

    judged = judge_helpfulness_batch(results)

    out_path = input_path.replace(".jsonl", "_judged.jsonl")
    save_jsonl(judged, out_path)
    print(f"Saved judged results to {out_path}")

    # Summary
    refused = sum(1 for r in judged if r.get("was_incorrectly_refused"))
    total = len(judged)
    print(f"FRR: {refused}/{total} = {refused/total:.1%}" if total else "No results")


_LEGACY_CS_BENIGN_PATH = Path("results/fairness/benign_cs_results_judged.jsonl")


def _warn_legacy_cs_benign_once(warned: set):
    """Flag the old single-file CS-benign layout so users regenerate it."""
    if "legacy" in warned or not _LEGACY_CS_BENIGN_PATH.exists():
        return
    warned.add("legacy")
    print(
        "  NOTE: Detected legacy shared CS-benign artifact at "
        f"{_LEGACY_CS_BENIGN_PATH}. This file is IGNORED — fairness now requires "
        "per-checkpoint/per-seed CS-benign inference under "
        "results/fairness/{ckpt}/seed_{seed}/. Re-run notebook cells 23-24 to "
        "regenerate checkpoint-specific artifacts."
    )


def _full_analysis(cfg, checkpoints, seeds):
    """Mode 3: Full fairness analysis across checkpoints and seeds."""
    fairness_cfg = cfg.get("analysis", {}).get("fairness", {})
    eval_cfg = cfg.get("evaluation", {})
    out_dir = Path(fairness_cfg.get("output_dir", "results/fairness"))
    out_dir.mkdir(parents=True, exist_ok=True)

    all_reports = {}
    warned = set()
    _warn_legacy_cs_benign_once(warned)

    for ckpt in checkpoints:
        print(f"\n{'='*50}")
        print(f"Checkpoint: {ckpt}")
        print(f"{'='*50}")

        # Aggregate attack results across seeds
        all_attack = []
        all_benign_en = []
        all_cs_benign = []
        missing_cs = []

        for seed in seeds:
            # Attack eval
            attack_path = f"results/eval/{ckpt}/seed_{seed}/held_out.jsonl"
            if Path(attack_path).exists():
                data = load_jsonl(attack_path)
                all_attack.extend(data)
            else:
                print(f"  WARNING: Missing {attack_path}")

            # English benign eval
            benign_path = f"results/eval/{ckpt}/seed_{seed}/benign_results.jsonl"
            if Path(benign_path).exists():
                all_benign_en.extend(load_jsonl(benign_path))

            cs_benign_path = _cs_benign_path(ckpt, seed, judged=True)
            if cs_benign_path.exists():
                all_cs_benign.extend(load_jsonl(str(cs_benign_path)))
            else:
                missing_cs.append(str(cs_benign_path))

        if not all_attack:
            print(f"  No attack data found for {ckpt}, skipping")
            continue

        if all_cs_benign and all_benign_en:
            report = full_fairness_report(
                all_attack,
                all_benign_en,
                all_cs_benign,
                label=ckpt,
                di_threshold=fairness_cfg.get("di_threshold", 0.8),
                eo_gap_threshold=fairness_cfg.get("eo_gap_threshold"),
                bootstrap_samples=eval_cfg.get("bootstrap_samples", 10000),
                confidence=1.0 - eval_cfg.get("significance_level", 0.05),
            )
        else:
            report = defense_equity_report(
                all_attack,
                label=ckpt,
                di_threshold=fairness_cfg.get("di_threshold", 0.8),
                eo_gap_threshold=fairness_cfg.get("eo_gap_threshold"),
                bootstrap_samples=eval_cfg.get("bootstrap_samples", 10000),
                confidence=1.0 - eval_cfg.get("significance_level", 0.05),
            )
            if missing_cs:
                print("  Missing checkpoint-specific CS-benign judged files:")
                for path in missing_cs[:3]:
                    print(f"    {path}")
                if len(missing_cs) > 3:
                    print(f"    ... and {len(missing_cs) - 3} more")
            if not all_cs_benign:
                print("  (No CS benign data for this checkpoint — using attack-only metrics)")

        all_reports[ckpt] = report
        _print_report(report)

    # Save combined report
    save_json(all_reports, str(out_dir / "fairness_report.json"))
    print(f"\nFull report saved to {out_dir / 'fairness_report.json'}")


def _print_report(report):
    """Print a formatted fairness report summary."""
    label = report.get("label", "?")

    # Per-language ASR
    per_lang = report.get("per_language_asr", {})
    if per_lang:
        print(f"\n  Per-language ASR ({label}):")
        for lang in sorted(per_lang):
            name = LANGUAGE_NAMES.get(lang, lang)
            print(f"    {name:12s} ({lang}): {per_lang[lang]:.1%}")

    # Disparate Impact
    di = report.get("disparate_impact", {})
    if di:
        status = "PASS" if di.get("four_fifths_pass") else "FAIL"
        print(f"\n  Disparate Impact: {di.get('ratio', 0):.4f}  [{status}]")
        if di.get("violations"):
            print(f"    Violations: {', '.join(di['violations'])}")

    # Equal Opportunity
    eo = report.get("equal_opportunity", {})
    if eo and eo.get("max_gap") is not None:
        status = ""
        if eo.get("passes_threshold") is not None:
            status = "  [PASS]" if eo["passes_threshold"] else "  [FAIL]"
        print(f"  Equal Opportunity Gap: {eo['max_gap']:.4f}{status}")
        if eo.get("worst_group"):
            print(f"    Worst: {eo['worst_group']}, Best: {eo['best_group']}")

    # Counterfactual Fairness
    cf = report.get("counterfactual_fairness", {})
    if cf and cf.get("n_matched_sets", 0) > 0:
        print(f"  Counterfactual Fairness: {cf['consistency_rate']:.1%} "
              f"({cf['n_consistent']}/{cf['n_matched_sets']} matched sets consistent)")

    # Inequality
    ineq = report.get("inequality", {})
    if ineq:
        print(f"  Gini (ASR): {ineq.get('gini_asr', 0):.4f}")
        print(f"  Theil (ASR): {ineq.get('theil_asr', 0):.4f}")
        print(f"  Max/Min Ratio: {ineq.get('max_min_ratio_asr', 0):.4f}")

    # FRR metrics (if available)
    if "english_frr" in report:
        print(f"\n  English FRR: {report['english_frr']:.1%}")
        per_frr = report.get("per_language_frr", {})
        for lang in sorted(per_frr):
            name = LANGUAGE_NAMES.get(lang, lang)
            print(f"    {name:12s} CS-benign FRR: {per_frr[lang]:.1%}")

        dp = report.get("cross_lingual_frr_parity", report.get("demographic_parity", {}))
        if dp:
            print(f"  Cross-Lingual FRR Parity Gap: {dp.get('max_gap', 0):.4f}")

        eod = report.get("equalized_odds", {})
        if eod:
            status = ""
            if eod.get("passes_threshold") is not None:
                status = "  [PASS]" if eod["passes_threshold"] else "  [FAIL]"
            print(f"  Equalized Odds Gap: {eod.get('eo_gap', 0):.4f}{status}")


def main():
    parser = argparse.ArgumentParser(description="MediCS fairness analysis")
    parser.add_argument("--config", default="config/experiment_config.yaml")

    # Mode 1
    parser.add_argument("--generate-benign-cs", action="store_true",
                        help="Generate code-switched benign queries")
    parser.add_argument("--n-samples", type=int, default=100,
                        help="Number of benign twins to sample per category")

    # Mode 2
    parser.add_argument("--judge-multilingual", action="store_true",
                        help="Judge helpfulness of CS benign results")
    parser.add_argument("--input", type=str, default=None,
                        help="Input JSONL for Mode 2")

    # Mode 3
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated checkpoint names (e.g., base,sft,dpo)")
    parser.add_argument("--seeds", type=str, default="42,123,456",
                        help="Comma-separated eval seeds")

    args = parser.parse_args()
    cfg = load_config(args.config)
    fairness_cfg = cfg.get("analysis", {}).get("fairness", {})

    if args.generate_benign_cs:
        n = args.n_samples or fairness_cfg.get("n_benign_cs_samples", 100)
        seed = fairness_cfg.get("benign_cs_seed", 42)
        _generate_benign_cs(cfg, n, seed)

    elif args.judge_multilingual:
        if not args.input:
            parser.error("--input required for --judge-multilingual")
        _judge_multilingual(args.input)

    elif args.checkpoints:
        ckpts = [c.strip() for c in args.checkpoints.split(",")]
        seeds = [int(s.strip()) for s in args.seeds.split(",")]
        _full_analysis(cfg, ckpts, seeds)

    else:
        parser.print_help()


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("Fairness Analysis"):
        main()
    save_timing_report()
