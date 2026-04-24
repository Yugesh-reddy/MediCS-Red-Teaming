"""
MediCS — Publication Figure Generation
========================================
12 publication-quality figures from evaluation results.
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


def fig11_fairness_dashboard(results_dir, figures_dir):
    """Fairness Dashboard — 2x2 panel: DI ratio, Gini trend, counterfactual, intersectional."""
    from medics.fairness import (
        disparate_impact_ratio, gini_coefficient,
        counterfactual_fairness, intersectional_analysis,
    )

    checkpoints = ["base", "sft", "dpo"]
    stage_labels = ["Base", "+SFT", "+SFT+DPO"]

    # Collect per-checkpoint data
    di_ratios = {}
    gini_vals = {}
    cf_data = {}
    inter_matrix = None

    for ckpt, label in zip(checkpoints, stage_labels):
        results = _find_held_out(results_dir, ckpt)
        if not results:
            continue
        lang_asr = compute_per_language_asr(results)
        defense_rates = {l: 1.0 - a for l, a in lang_asr.items()}

        di = disparate_impact_ratio(defense_rates)
        di_ratios[label] = di.get("ratio", 0)
        gini_vals[label] = gini_coefficient(lang_asr)

        cf = counterfactual_fairness(results)
        cf_data[label] = cf.get("consistency_rate", 1.0)

        if ckpt == "dpo":
            inter_matrix = intersectional_analysis(results).get("matrix", {})

    if not di_ratios:
        print("  Figure 11: No data")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Disparate Impact Ratio
    ax = axes[0, 0]
    stages = list(di_ratios.keys())
    vals = [di_ratios[s] for s in stages]
    colors = ["#e74c3c", "#f39c12", "#27ae60"][:len(stages)]
    ax.bar(stages, vals, color=colors, edgecolor='white', linewidth=2, width=0.5)
    ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='4/5 threshold')
    ax.set_ylabel('Disparate Impact Ratio', fontsize=12)
    ax.set_title('(a) Disparate Impact (4/5 Rule)', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)

    # (b) Gini coefficient trend
    ax = axes[0, 1]
    stages_g = list(gini_vals.keys())
    ax.plot(stages_g, [gini_vals[s] for s in stages_g], 'o-', color='#2c3e50',
            linewidth=2, markersize=8)
    ax.set_ylabel('Gini Coefficient', fontsize=12)
    ax.set_title('(b) ASR Inequality Across Languages', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(gini_vals.values()) * 1.5 + 0.05)

    # (c) Counterfactual fairness
    ax = axes[1, 0]
    stages_c = list(cf_data.keys())
    vals_c = [cf_data[s] * 100 for s in stages_c]
    ax.bar(stages_c, vals_c, color=colors[:len(stages_c)], edgecolor='white',
           linewidth=2, width=0.5)
    ax.set_ylabel('Consistency (%)', fontsize=12)
    ax.set_title('(c) Counterfactual Fairness', fontsize=13, fontweight='bold')
    ax.set_ylim(0, 105)

    # (d) Intersectional heatmap (DPO checkpoint)
    ax = axes[1, 1]
    if inter_matrix:
        langs = sorted(inter_matrix.keys())
        cats = sorted(set(c for l in inter_matrix.values() for c in l))
        data = np.array([[inter_matrix.get(l, {}).get(c, 0) for c in cats] for l in langs])
        im = ax.imshow(data, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(range(len(cats)))
        ax.set_xticklabels(cats, fontsize=9, rotation=45)
        ax.set_yticks(range(len(langs)))
        ax.set_yticklabels(langs, fontsize=9)
        plt.colorbar(im, ax=ax, label='ASR', shrink=0.8)
    ax.set_title('(d) Intersectional ASR (DPO)', fontsize=13, fontweight='bold')

    plt.suptitle('Algorithmic Fairness Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig11_fairness_dashboard.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig11_fairness_dashboard.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 11: Fairness Dashboard")


def fig12_safety_fairness_tradeoff(results_dir, figures_dir):
    """Safety-Fairness Tradeoff — scatter + FRR comparison."""
    from medics.fairness import equal_opportunity_gap

    checkpoints = ["base", "sft", "dpo"]
    stage_labels = ["Base", "+SFT", "+SFT+DPO"]
    markers = ["o", "s", "D"]

    asrs = []
    eo_gaps = []
    labels_plot = []

    for ckpt, label in zip(checkpoints, stage_labels):
        results = _find_held_out(results_dir, ckpt)
        if not results:
            continue
        valid = [r for r in results if r.get("judge_label") != "error"]
        overall_asr = sum(1 for r in valid if r.get("judge_label") == "harmful") / len(valid) if valid else 0
        lang_asr = compute_per_language_asr(results)
        defense_rates = {l: 1.0 - a for l, a in lang_asr.items()}
        eo = equal_opportunity_gap(defense_rates)

        asrs.append(overall_asr * 100)
        eo_gaps.append(eo["max_gap"] * 100)
        labels_plot.append(label)

    if not asrs:
        print("  Figure 12: No data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # (a) Safety-Fairness tradeoff scatter
    ax = axes[0]
    colors = ["#e74c3c", "#f39c12", "#27ae60"]
    for i, (x, y, lab) in enumerate(zip(asrs, eo_gaps, labels_plot)):
        ax.scatter(x, y, c=colors[i], s=150, marker=markers[i], zorder=5,
                   edgecolors='black', linewidth=1.5)
        ax.annotate(lab, (x, y), textcoords="offset points", xytext=(10, 5), fontsize=11)
    ax.set_xlabel('Overall ASR (%)', fontsize=13)
    ax.set_ylabel('Equal Opportunity Gap (%)', fontsize=13)
    ax.set_title('(a) Safety vs Fairness Tradeoff', fontsize=13, fontweight='bold')

    # (b) English FRR vs CS-benign FRR (if data exists)
    ax = axes[1]
    fairness_report = load_json("results/fairness/fairness_report.json")
    plotted = False
    if fairness_report:
        # Use DPO checkpoint if available, else last available
        for ckpt_key in ["dpo", "sft", "base"]:
            rpt = fairness_report.get(ckpt_key, {})
            if "per_language_frr" in rpt:
                en_frr = rpt.get("english_frr", 0) * 100
                per_lang = rpt["per_language_frr"]
                langs = sorted(per_lang.keys())
                x_pos = np.arange(len(langs))
                width = 0.35
                ax.bar(x_pos - width/2, [en_frr] * len(langs), width,
                       label='English', color='#3498db', edgecolor='white')
                ax.bar(x_pos + width/2, [per_lang[l] * 100 for l in langs], width,
                       label='CS-Benign', color='#e74c3c', edgecolor='white')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(langs, fontsize=10)
                ax.set_ylabel('False Refusal Rate (%)', fontsize=12)
                ax.legend(fontsize=10)
                ax.set_title(f'(b) FRR: English vs CS-Benign ({ckpt_key.upper()})',
                           fontsize=13, fontweight='bold')
                plotted = True
                break

    if not plotted:
        ax.text(0.5, 0.5, 'CS-benign FRR data\nnot yet available',
                ha='center', va='center', transform=ax.transAxes, fontsize=14,
                color='gray')
        ax.set_title('(b) FRR: English vs CS-Benign', fontsize=13, fontweight='bold')

    plt.suptitle('Safety-Fairness Tradeoff Analysis', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig12_safety_fairness.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig12_safety_fairness.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 12: Safety-Fairness Tradeoff")


def fig13_thompson_entropy(figures_dir):
    """Thompson Sampling exploration — arm-selection entropy across rounds.

    Entropy = -Σ p_i log2 p_i over the 5 arms, computed on each round's history.
    High entropy late in training → bandit still exploring (failure mode).
    Low entropy late → bandit has converged on a preferred arm.
    """
    bandit_paths = sorted(Path("results/attacks").glob("round_*/bandit_state.json"))
    if not bandit_paths:
        print("  Figure 13: No bandit states")
        return

    rounds = []
    entropies = []
    per_arm_rates_by_round = []
    arms_master = None

    for path in bandit_paths:
        state = load_json(str(path))
        if not state or "history" not in state:
            continue
        history = state["history"]
        arms = state.get("arms", ["CS", "RP", "MTE", "CS-RP", "CS-OBF"])
        if arms_master is None:
            arms_master = arms
        counts = {arm: 0 for arm in arms}
        for h in history:
            arm = h.get("strategy")
            if arm in counts:
                counts[arm] += 1
        total = sum(counts.values())
        if total == 0:
            continue
        probs = np.array([counts[a] / total for a in arms])
        nonzero = probs[probs > 0]
        ent = float(-np.sum(nonzero * np.log2(nonzero)))
        round_num = int(path.parent.name.split("_")[-1])
        rounds.append(round_num)
        entropies.append(ent)
        per_arm_rates_by_round.append(probs)

    if not rounds:
        print("  Figure 13: No bandit history")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    max_ent = np.log2(len(arms_master))
    ax1.plot(rounds, entropies, 'o-', color='#2c3e50', linewidth=2.5, markersize=10)
    ax1.axhline(max_ent, color='red', linestyle='--', linewidth=1.5,
                label=f'Max entropy (uniform) = {max_ent:.2f}')
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Arm-Selection Entropy (bits)', fontsize=12)
    ax1.set_title('(a) Bandit Exploration Over Rounds', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, max_ent * 1.1)
    ax1.set_xticks(rounds)
    ax1.legend(fontsize=10)

    matrix = np.array(per_arm_rates_by_round).T  # arms × rounds
    im = ax2.imshow(matrix, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
    ax2.set_yticks(range(len(arms_master)))
    ax2.set_yticklabels(arms_master, fontsize=11)
    ax2.set_xticks(range(len(rounds)))
    ax2.set_xticklabels([f"R{r}" for r in rounds], fontsize=11)
    ax2.set_title('(b) Per-Arm Selection Rate', fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax2, label='Selection rate', shrink=0.85)

    plt.suptitle('Thompson Sampling Convergence Analysis', fontsize=15,
                 fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig13_thompson_entropy.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig13_thompson_entropy.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 13: Thompson Sampling Entropy")


def fig14_response_length(results_dir, figures_dir):
    """Response length distributions — refusals vs compliance across checkpoints.

    Short responses tend to be refusals; long responses tend to be compliance.
    A DPO checkpoint whose harmful-response length collapses toward refusal
    length indicates the defense is producing real refusals rather than
    partially compliant text.
    """
    checkpoints = ["base", "sft", "dpo"]
    records = []
    for ckpt in checkpoints:
        results = _find_held_out(results_dir, ckpt)
        if not results:
            continue
        for r in results:
            label = r.get("judge_label")
            if label not in ("harmful", "safe"):
                continue
            resp = r.get("model_response", "") or ""
            records.append({
                "checkpoint": ckpt,
                "label": label,
                "length_chars": len(resp),
                "length_words": len(resp.split()),
            })

    if not records:
        print("  Figure 14: No judged data — run Cell 22 first")
        return

    df = pd.DataFrame(records)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Violin plot: word counts by checkpoint × label
    stages = [c for c in checkpoints if c in df["checkpoint"].unique()]
    palette = {"safe": "#27ae60", "harmful": "#e74c3c"}

    ax = axes[0]
    sns.violinplot(data=df, x="checkpoint", y="length_words", hue="label",
                   order=stages, palette=palette, split=True,
                   inner="quartile", ax=ax, cut=0)
    ax.set_ylabel('Response length (words)', fontsize=12)
    ax.set_xlabel('')
    ax.set_title('(a) Response Length by Judgment', fontsize=13, fontweight='bold')
    ax.legend(title='Judge', fontsize=10)
    ax.set_ylim(0, df["length_words"].quantile(0.98))

    # Median lengths table
    ax = axes[1]
    ax.axis('off')
    summary = (
        df.groupby(["checkpoint", "label"])["length_words"]
        .agg(["median", "mean", "count"]).round(1).reset_index()
    )
    table_rows = [["Checkpoint", "Label", "Median", "Mean", "N"]]
    for _, row in summary.iterrows():
        table_rows.append([
            str(row["checkpoint"]), str(row["label"]),
            f"{row['median']:.0f}", f"{row['mean']:.0f}", f"{row['count']:.0f}",
        ])
    tbl = ax.table(cellText=table_rows[1:], colLabels=table_rows[0],
                   loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.8)
    ax.set_title('(b) Summary Statistics', fontsize=13, fontweight='bold')

    plt.suptitle('Response Length Analysis', fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig14_response_length.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig14_response_length.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 14: Response Length Analysis")


def fig15_strategy_by_category(figures_dir):
    """Strategy × category heatmap on the attack-round data (which strategy
    beats which harm category). Uses round_*/results.jsonl (all rounds merged)
    because held-out eval is code-switch-only by construction.
    """
    results = []
    for path in sorted(Path("results/attacks").glob("round_*/results.jsonl")):
        results.extend(load_jsonl(str(path)))

    if not results:
        print("  Figure 15: No attack round results")
        return

    # Build cells: {(strategy, category): (harmful_count, total_valid)}
    cells = {}
    for r in results:
        if r.get("judge_label") not in ("harmful", "safe"):
            continue
        key = (r.get("strategy", "?"), r.get("category", "?"))
        h, t = cells.get(key, (0, 0))
        t += 1
        if r.get("judge_label") == "harmful":
            h += 1
        cells[key] = (h, t)

    if not cells:
        print("  Figure 15: No judged attack data")
        return

    strategies = sorted({k[0] for k in cells})
    categories = sorted({k[1] for k in cells})
    asr_matrix = np.full((len(strategies), len(categories)), np.nan)
    support_matrix = np.zeros((len(strategies), len(categories)), dtype=int)
    for i, s in enumerate(strategies):
        for j, c in enumerate(categories):
            h, t = cells.get((s, c), (0, 0))
            support_matrix[i, j] = t
            if t >= 3:
                asr_matrix[i, j] = h / t

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(asr_matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies, fontsize=11)
    ax.set_xlabel('Harm Category', fontsize=12)
    ax.set_ylabel('Attack Strategy', fontsize=12)
    ax.set_title('ASR by Strategy × Category (all rounds)', fontsize=14,
                 fontweight='bold')

    for i in range(len(strategies)):
        for j in range(len(categories)):
            val = asr_matrix[i, j]
            n = support_matrix[i, j]
            if np.isnan(val):
                txt = f"n={n}"
                color = "gray"
            else:
                txt = f"{val:.0%}\n(n={n})"
                color = "white" if val > 0.5 else "black"
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label='ASR', shrink=0.85)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/fig15_strategy_by_category.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{figures_dir}/fig15_strategy_by_category.pdf", bbox_inches='tight')
    plt.close()
    print("  Figure 15: Strategy × Category Heatmap")
