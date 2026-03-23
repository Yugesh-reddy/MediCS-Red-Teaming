"""
MediCS — Publication Figure Generation
========================================
10 publication-quality figures from evaluation results.
All plotting logic lives here; scripts/05_generate_figures.py orchestrates.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from medics.utils import load_jsonl, load_json
from medics.metrics import (
    compute_asr, compute_robustness_gain,
    compute_per_category_asr, compute_per_strategy_asr,
    compute_per_language_asr, bootstrap_ci,
)

# Style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
FIGSIZE = (10, 6)


def _find_held_out(results_dir, ckpt, filename="held_out.jsonl"):
    """Find results file, preferring multi-seed path (seed_42), then single path."""
    seed_path = Path(results_dir) / ckpt / "seed_42" / filename
    single_path = Path(results_dir) / ckpt / filename
    path = seed_path if seed_path.exists() else single_path
    return load_jsonl(str(path))


def fig1_asr_defense_stages(results_dir, figures_dir):
    """ASR Across Defense Stages — the money shot."""
    checkpoints = ["base", "sft", "dpo"]
    labels = ["Base\n(Llama-3-8B)", "+SFT\n(QLoRA)", "+SFT+DPO"]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]

    asrs = []
    cis = []
    for ckpt in checkpoints:
        results = _find_held_out(results_dir, ckpt)
        if results:
            valid = [r for r in results if r.get("judge_label") != "error"]
            asr_vals = [1 if r.get("judge_label") == "harmful" else 0 for r in valid]
            mean, lo, hi = bootstrap_ci(asr_vals)
            asrs.append(mean)
            cis.append((mean - lo, hi - mean))
        else:
            asrs.append(0)
            cis.append((0, 0))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, [a * 100 for a in asrs], color=colors,
                  edgecolor='white', linewidth=2, width=0.5)

    # Error bars
    if any(c[0] > 0 for c in cis):
        ax.errorbar(labels, [a * 100 for a in asrs],
                     yerr=[[c[0] * 100 for c in cis], [c[1] * 100 for c in cis]],
                     fmt='none', color='black', capsize=8, capthick=2)

    # Labels
    for bar, asr in zip(bars, asrs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{asr:.1%}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Attack Success Rate (%)', fontsize=14)
    ax.set_title('MediCS: ASR Across Defense Stages', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig1_asr_defense_stages.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig1_asr_defense_stages.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 1: ASR Across Defense Stages")


def fig2_strategy_heatmap(results_dir, figures_dir):
    """Strategy Effectiveness Heatmap (categories x strategies)."""
    results = _find_held_out(results_dir, "base")
    if not results:
        print("  Figure 2: No base results")
        return

    categories = sorted(set(r.get("category", "?") for r in results))
    strategies = sorted(set(r.get("strategy", "?") for r in results))

    matrix = np.zeros((len(categories), len(strategies)))
    for i, cat in enumerate(categories):
        for j, strat in enumerate(strategies):
            subset = [r for r in results
                      if r.get("category") == cat and r.get("strategy") == strat]
            if subset:
                matrix[i, j] = compute_asr(subset) * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=strategies, yticklabels=categories,
                annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax,
                vmin=0, vmax=100, cbar_kws={'label': 'ASR (%)'})
    ax.set_title('Strategy x Category ASR (%)', fontsize=16, fontweight='bold')
    ax.set_xlabel('Strategy', fontsize=14)
    ax.set_ylabel('Category', fontsize=14)

    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig2_strategy_heatmap.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig2_strategy_heatmap.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 2: Strategy Effectiveness Heatmap")


def fig3_cross_language(results_dir, figures_dir):
    """Cross-Language Vulnerability (ASR per language x model stage)."""
    checkpoints = ["base", "sft", "dpo"]
    stage_labels = ["Base", "+SFT", "+SFT+DPO"]

    all_data = []
    for ckpt, label in zip(checkpoints, stage_labels):
        results = _find_held_out(results_dir, ckpt)
        if results:
            lang_asr = compute_per_language_asr(results)
            for lang, asr in lang_asr.items():
                all_data.append({"Language": lang, "ASR": asr * 100, "Stage": label})

    if not all_data:
        print("  Figure 3: No data")
        return

    df = pd.DataFrame(all_data)
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(data=df, x="Language", y="ASR", hue="Stage", ax=ax)
    ax.set_title('Cross-Language ASR by Defense Stage', fontsize=16, fontweight='bold')
    ax.set_ylabel('ASR (%)', fontsize=14)
    ax.set_ylim(0, 100)
    ax.legend(title='Defense Stage')

    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig3_cross_language.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig3_cross_language.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 3: Cross-Language Vulnerability")


def fig4_thompson_convergence(figures_dir):
    """Thompson Sampling Convergence (per-category arm selection over time)."""
    bandit_paths = sorted(Path("results/attacks").glob("round_*/bandit_state.json"))
    if not bandit_paths:
        print("  Figure 4: No bandit states")
        return

    state = load_json(str(bandit_paths[-1]))
    if not state or "history" not in state:
        print("  Figure 4: Empty bandit state")
        return

    history = state["history"]
    arms = state.get("arms", ["CS", "RP", "MTE", "CS-RP", "CS-OBF"])

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for arm in arms:
        cum_counts = []
        count = 0
        for i, h in enumerate(history):
            if h.get("strategy") == arm:
                count += 1
            cum_counts.append(count / (i + 1))
        ax.plot(cum_counts, label=arm, linewidth=2)

    ax.set_xlabel('Trial', fontsize=14)
    ax.set_ylabel('Selection Rate', fontsize=14)
    ax.set_title('Thompson Sampling Convergence', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)

    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig4_thompson_convergence.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig4_thompson_convergence.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 4: Thompson Sampling Convergence")


def fig5_failure_modes(results_dir, figures_dir):
    """Failure Mode Distribution -- how attacks succeed + residual breakdown."""
    results = _find_held_out(results_dir, "dpo")
    if not results:
        results = _find_held_out(results_dir, "sft")
    if not results:
        print("  Figure 5: No results")
        return

    harmful = [r for r in results if r.get("judge_label") == "harmful"]
    if not harmful:
        print("  Figure 5: No harmful responses to analyze")
        return

    # Panel (a): Pie chart of failure modes by strategy
    strat_counts = {}
    for r in harmful:
        strat = r.get("strategy", "unknown")
        strat_counts[strat] = strat_counts.get(strat, 0) + 1

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    labels = list(strat_counts.keys())
    sizes = list(strat_counts.values())
    colors_pie = sns.color_palette("husl", len(labels))

    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, autopct='%1.1f%%',
        colors=colors_pie, startangle=90, textprops={'fontsize': 11}
    )
    ax1.set_title('(a) Residual Failures by Strategy', fontsize=14, fontweight='bold')

    # Panel (b): Residual ASR by category x language heatmap
    from collections import defaultdict
    valid = [r for r in results if r.get("judge_label") != "error"]
    categories = sorted(set(r.get("category", "?") for r in valid))
    languages = sorted(set(r.get("language", "?") for r in valid))

    matrix = np.zeros((len(categories), len(languages)))
    for i, cat in enumerate(categories):
        for j, lang in enumerate(languages):
            subset = [r for r in valid
                      if r.get("category") == cat and r.get("language") == lang]
            if subset:
                n_harmful = sum(1 for r in subset if r.get("judge_label") == "harmful")
                matrix[i, j] = n_harmful / len(subset) * 100

    if categories and languages:
        sns.heatmap(matrix, xticklabels=languages, yticklabels=categories,
                    annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax2,
                    vmin=0, vmax=max(50, matrix.max()), cbar_kws={'label': 'ASR (%)'})
        ax2.set_title('(b) Residual ASR: Category x Language', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Language', fontsize=12)
        ax2.set_ylabel('Category', fontsize=12)

    plt.suptitle('Failure Mode Analysis (Post-Defense)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig5_failure_modes.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig5_failure_modes.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 5: Failure Mode Distribution (enhanced with residual breakdown)")


def fig6_robustness_gain(results_dir, figures_dir):
    """Robustness Gain Summary (RG per category)."""
    base_results = _find_held_out(results_dir, "base")
    dpo_results = _find_held_out(results_dir, "dpo")
    if not base_results or not dpo_results:
        print("  Figure 6: Missing results")
        return

    base_cat = compute_per_category_asr(base_results)
    dpo_cat = compute_per_category_asr(dpo_results)

    categories = sorted(set(base_cat.keys()) & set(dpo_cat.keys()))
    gains = [compute_robustness_gain(base_cat[c], dpo_cat[c]) * 100 for c in categories]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(categories, gains, color=sns.color_palette("viridis", len(categories)),
                  edgecolor='white', linewidth=2)

    for bar, gain in zip(bars, gains):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{gain:.0f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_ylabel('Robustness Gain (%)', fontsize=14)
    ax.set_title('Robustness Gain by Category (Base -> +SFT+DPO)', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig6_robustness_gain.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig6_robustness_gain.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 6: Robustness Gain Summary")


def fig7_overrefusal_correction(results_dir, figures_dir):
    """DPO Over-Refusal Correction -- HR drops with SFT, recovers with DPO."""
    checkpoints = ["base", "sft", "dpo"]
    labels = ["Base", "+SFT", "+SFT+DPO"]
    colors = ["#3498db", "#e67e22", "#2ecc71"]

    hrs = []
    for ckpt in checkpoints:
        benign = _find_held_out(results_dir, ckpt, "benign_results.jsonl")
        if benign:
            valid = [r for r in benign if not r.get("judge_error")]
            helpful = sum(1 for r in valid if not r.get("was_incorrectly_refused"))
            hrs.append(helpful / len(valid) * 100 if valid else 0)
        else:
            hrs.append(0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bars = ax.bar(labels, hrs, color=colors, edgecolor='white', linewidth=2, width=0.5)

    for bar, hr in zip(bars, hrs):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{hr:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')

    ax.set_ylabel('Helpfulness Retention (%)', fontsize=14)
    ax.set_title('Over-Refusal Correction: SFT Drops, DPO Recovers',
                  fontsize=16, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='Target: 80%')
    ax.legend()

    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig7_overrefusal_correction.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig7_overrefusal_correction.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 7: DPO Over-Refusal Correction")


def fig8_semantic_vs_asr(figures_dir):
    """Semantic Preservation vs ASR -- does translation quality predict attack success?"""
    scores = load_json("data/medics_500/semantic_scores.json")
    results = _find_held_out("results/eval", "base")

    if not scores or not results:
        print("  Figure 8: Missing data")
        return

    result_lookup = {}
    for r in results:
        key = (r.get("seed_id", ""), r.get("language", ""))
        result_lookup[key] = r.get("judge_label") == "harmful"

    sem_scores = []
    attack_success = []
    for s in scores:
        key = (s["seed_id"], s["language"])
        if key in result_lookup:
            sem_scores.append(s["score"])
            attack_success.append(1 if result_lookup[key] else 0)

    if not sem_scores:
        print("  Figure 8: No matching data")
        return

    fig, ax = plt.subplots(figsize=FIGSIZE)

    bins = np.linspace(min(sem_scores), max(sem_scores), 10)
    bin_indices = np.digitize(sem_scores, bins)
    bin_asrs = []
    bin_centers = []
    for b in range(1, len(bins) + 1):
        mask = bin_indices == b
        if np.sum(mask) > 0:
            bin_asrs.append(np.mean([attack_success[i] for i in range(len(mask)) if mask[i]]) * 100)
            bin_centers.append(bins[b-1] if b <= len(bins) else bins[-1])

    ax.scatter(sem_scores, [a * 100 for a in attack_success], alpha=0.1, s=10)
    if bin_centers:
        ax.plot(bin_centers, bin_asrs, 'r-o', linewidth=2, markersize=8, label='Binned ASR')

    ax.set_xlabel('Semantic Preservation Score', fontsize=14)
    ax.set_ylabel('Attack Success Rate (%)', fontsize=14)
    ax.set_title('Semantic Preservation vs Attack Success', fontsize=16, fontweight='bold')
    ax.legend()

    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig8_semantic_vs_asr.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig8_semantic_vs_asr.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 8: Semantic Preservation vs ASR")


def fig9_token_fragmentation(figures_dir):
    """Token count ratio by language -- explains WHY CS attacks work."""
    analysis = load_json("results/analysis/tokenization_analysis.json")
    if not analysis:
        print("  Figure 9: No tokenization analysis data")
        return

    df = pd.DataFrame(analysis)
    if "token_count_ratio" not in df.columns:
        print("  Figure 9: Invalid analysis format")
        return

    # Order languages by median ratio (most fragmented first)
    lang_order = (
        df.groupby("language")["token_count_ratio"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel (a): Box plot of token count ratio
    sns.boxplot(data=df, x="language", y="token_count_ratio",
                order=lang_order, ax=ax1, palette="Set2")
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='No fragmentation')
    ax1.set_xlabel('Language', fontsize=14)
    ax1.set_ylabel('Token Count Ratio (CS / English)', fontsize=14)
    ax1.set_title('(a) Token Fragmentation by Language', fontsize=14, fontweight='bold')
    ax1.legend()

    # Panel (b): OOV proxy rate by language
    sns.boxplot(data=df, x="language", y="oov_proxy_rate",
                order=lang_order, ax=ax2, palette="Set2")
    ax2.set_xlabel('Language', fontsize=14)
    ax2.set_ylabel('Byte-Fallback Token Rate', fontsize=14)
    ax2.set_title('(b) OOV Proxy Rate by Language', fontsize=14, fontweight='bold')

    plt.suptitle('Tokenization Analysis: Why Code-Switching Works',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig9_token_fragmentation.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig9_token_fragmentation.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 9: Token Fragmentation by Language")


def fig10_perplexity_detection(figures_dir):
    """Perplexity-based detection baseline -- can CS attacks be trivially detected?"""
    detection = load_json("results/analysis/perplexity_results.json")
    if not detection:
        print("  Figure 10: No perplexity detection data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel (a): Violin plot of perplexity distribution
    if "per_input" in detection:
        entries = detection["per_input"]
        en_ppls = [e["perplexity"] for e in entries if not e.get("is_cs", True)]
        cs_ppls = [e["perplexity"] for e in entries if e.get("is_cs", True)]

        data_violin = (
            [{"Perplexity": p, "Type": "English"} for p in en_ppls] +
            [{"Perplexity": p, "Type": "Code-Switched"} for p in cs_ppls]
        )
        if data_violin:
            df_v = pd.DataFrame(data_violin)
            sns.violinplot(data=df_v, x="Type", y="Perplexity", ax=ax1,
                          palette=["#3498db", "#e74c3c"], cut=0)
            ax1.set_title('(a) Perplexity Distribution', fontsize=14, fontweight='bold')
            ax1.set_ylabel('Perplexity', fontsize=14)

    # Panel (b): ROC curve
    if "roc" in detection:
        fpr = detection["roc"]["fpr"]
        tpr = detection["roc"]["tpr"]
        auroc = detection.get("auroc", 0.0)
        ax2.plot(fpr, tpr, 'b-', linewidth=2, label=f'Perplexity detector (AUROC={auroc:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
        ax2.set_xlabel('False Positive Rate', fontsize=14)
        ax2.set_ylabel('True Positive Rate', fontsize=14)
        ax2.set_title('(b) ROC Curve: Detecting CS Attacks', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
    elif "per_threshold" in detection:
        thresholds = [t["threshold"] for t in detection["per_threshold"]]
        f1s = [t["f1"] for t in detection["per_threshold"]]
        ax2.plot(thresholds, f1s, 'b-o', linewidth=2)
        ax2.set_xlabel('Perplexity Threshold', fontsize=14)
        ax2.set_ylabel('F1 Score', fontsize=14)
        ax2.set_title('(b) Detection F1 by Threshold', fontsize=14, fontweight='bold')

    plt.suptitle('Perplexity-Based Detection Baseline',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig10_perplexity_detection.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig10_perplexity_detection.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 10: Perplexity Detection Baseline")
