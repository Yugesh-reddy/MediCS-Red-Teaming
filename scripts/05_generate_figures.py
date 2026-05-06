#!/usr/bin/env python3
"""
Generate 15 publication-quality figures from evaluation results.

Usage:
  python scripts/05_generate_figures.py --results-dir results/eval/

Figures:
  1. ASR Across Defense Stages (bar chart with CI)
  2. Strategy Effectiveness Heatmap
  3. Cross-Language Vulnerability
  4. Thompson Sampling Convergence
  5. Failure Mode Distribution (enhanced with residual breakdown)
  6. Robustness Gain Summary
  7. DPO Safety Regression (negative result)
  8. Semantic Preservation vs ASR
  9. Token Fragmentation by Language
  10. Perplexity Detection Baseline
  11. Fairness Dashboard (DI, Gini, counterfactual, intersectional)
  12. Safety-Fairness Tradeoff
  13. Thompson Sampling Entropy (exploration vs convergence)
  14. Response Length Analysis (refusals vs compliance)
  15. Strategy × Category Heatmap (attack-round data)
  16. Cross-Architecture Transfer Comparison (Llama-3 vs Mistral-7B vs Qwen-2.5-7B)
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
    fig11_fairness_dashboard,
    fig12_safety_fairness_tradeoff,
    fig13_thompson_entropy,
    fig14_response_length,
    fig15_strategy_by_category,
    fig16_transfer_comparison,
)


def main():
    parser = argparse.ArgumentParser(description="Generate publication figures")
    parser.add_argument("--results-dir", default="results/eval")
    parser.add_argument("--figures-dir", default="figures")
    args = parser.parse_args()

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=== Generating 15 Publication Figures ===\n")

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
    fig11_fairness_dashboard(args.results_dir, figures_dir)
    fig12_safety_fairness_tradeoff(args.results_dir, figures_dir)
    fig13_thompson_entropy(figures_dir)
    fig14_response_length(args.results_dir, figures_dir)
    fig15_strategy_by_category(figures_dir)
    fig16_transfer_comparison(args.results_dir, figures_dir)

    print(f"\nAll figures saved to {figures_dir}/")


if __name__ == "__main__":
    main()
