#!/usr/bin/env python3
"""
Generate 10 publication-quality figures from evaluation results.

Usage:
  python scripts/05_generate_figures.py --results-dir results/eval/

Figures:
  1. ASR Across Defense Stages (bar chart with CI)
  2. Strategy Effectiveness Heatmap
  3. Cross-Language Vulnerability
  4. Thompson Sampling Convergence
  5. Failure Mode Distribution (enhanced with residual breakdown)
  6. Robustness Gain Summary
  7. DPO Over-Refusal Correction
  8. Semantic Preservation vs ASR
  9. Token Fragmentation by Language
  10. Perplexity Detection Baseline
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.figures import (
    fig1_asr_defense_stages,
    fig2_strategy_heatmap,
    fig3_cross_language,
    fig4_thompson_convergence,
    fig5_failure_modes,
    fig6_robustness_gain,
    fig7_overrefusal_correction,
    fig8_semantic_vs_asr,
    fig9_token_fragmentation,
    fig10_perplexity_detection,
)


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results-dir", default="results/eval")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating 10 Publication Figures ===\n")

    fig1_asr_defense_stages(args.results_dir, figures_dir)
    fig2_strategy_heatmap(args.results_dir, figures_dir)
    fig3_cross_language(args.results_dir, figures_dir)
    fig4_thompson_convergence(figures_dir)
    fig5_failure_modes(args.results_dir, figures_dir)
    fig6_robustness_gain(args.results_dir, figures_dir)
    fig7_overrefusal_correction(args.results_dir, figures_dir)
    fig8_semantic_vs_asr(figures_dir)
    fig9_token_fragmentation(figures_dir)
    fig10_perplexity_detection(figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
