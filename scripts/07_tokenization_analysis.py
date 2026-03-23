#!/usr/bin/env python3
"""
Analyze token fragmentation across languages. Explains WHY code-switching
attacks bypass safety filters — low-resource text fragments into many more
tokens, disrupting the model's pattern recognition.

Runs AFTER 01_build_dataset.py (needs keywords). No GPU needed, $0 cost.

Usage:
    python scripts/07_tokenization_analysis.py --config config/experiment_config.yaml
    python scripts/07_tokenization_analysis.py --max-seeds 50  # quick test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.tokenization import analyze_tokenization, compute_fragmentation_summary
from medics.utils import load_jsonl, load_json, save_json, load_config


def main():
    parser = argparse.ArgumentParser(description="MediCS tokenization analysis")
    parser.add_argument("--config", default="config/experiment_config.yaml")
    parser.add_argument("--max-seeds", type=int, default=0,
                        help="Max seeds to analyze (0 = all)")
    parser.add_argument("--output-dir", default=None,
                        help="Override output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    model_id = config["target_model"]["model_id"]
    languages = [lang["code"] for lang in config["dataset"]["languages"]]

    analysis_cfg = config.get("analysis", {}).get("tokenization", {})
    max_seeds = args.max_seeds or analysis_cfg.get("max_seeds", 0)
    output_dir = args.output_dir or analysis_cfg.get("output_dir", "results/analysis")

    # Load seeds and keywords
    seeds_path = Path("data/medics_500/medics_500_full.jsonl")
    if not seeds_path.exists():
        seeds_path = Path("data/seeds/deduped_seeds.jsonl")
    if not seeds_path.exists():
        seeds_path = Path("data/seeds/raw_seeds.jsonl")
    if not seeds_path.exists():
        print("ERROR: No seed file found. Run 01_build_dataset.py first.")
        return

    seeds = load_jsonl(str(seeds_path))
    print(f"Loaded {len(seeds)} seeds from {seeds_path}")

    keywords_path = Path("data/seeds/keywords_checkpoint.json")
    keywords = load_json(str(keywords_path)) if keywords_path.exists() else {}
    print(f"Loaded keywords for {len(keywords)} seeds")

    print(f"\nAnalyzing tokenization for {model_id}")
    print(f"Languages: {languages}")
    if max_seeds > 0:
        print(f"Max seeds: {max_seeds}")

    results = analyze_tokenization(
        tokenizer_name_or_path=model_id,
        seeds=seeds,
        keywords=keywords,
        languages=languages,
        max_seeds=max_seeds,
    )

    # Save detailed results
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_path = f"{output_dir}/tokenization_analysis.json"
    save_json(results, output_path)
    print(f"\nDetailed results: {output_path} ({len(results)} entries)")

    # Print summary
    summary = compute_fragmentation_summary(results)
    save_json(summary, f"{output_dir}/tokenization_summary.json")

    print(f"\n{'='*60}")
    print(f"  TOKENIZATION FRAGMENTATION SUMMARY")
    print(f"{'='*60}")
    print(f"  {'Language':<12} {'Mean Ratio':>12} {'Median':>10} "
          f"{'Max':>8} {'OOV Proxy':>10} {'KW Ratio':>10}")
    print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*8} {'-'*10} {'-'*10}")
    for lang, stats in sorted(summary.items(),
                               key=lambda x: x[1]["mean_ratio"], reverse=True):
        print(f"  {lang:<12} {stats['mean_ratio']:>12.2f}x "
              f"{stats['median_ratio']:>9.2f}x {stats['max_ratio']:>7.2f}x "
              f"{stats['mean_oov_proxy']:>10.1%} {stats['mean_kw_ratio']:>9.2f}x")
    print(f"{'='*60}")
    print(f"\nInterpretation: Higher ratio = more token fragmentation = ")
    print(f"harder for model to recognize harmful patterns.")


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("Tokenization Analysis"):
        main()
    save_timing_report()
