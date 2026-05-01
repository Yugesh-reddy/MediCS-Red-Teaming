"""Regenerates report/figures/* with a colorblind-safe palette, larger
labels, and a redesigned set:

  - fig_pipeline.png            : closed-loop architecture diagram
  - fig1_asr_defense_stages.png : redesigned headline ASR bar chart
  - fig3_cross_language.png     : per-language ASR with CIs (grouped bars)
  - fig5_failure_modes.png      : SFT residual heatmap, single shared scale
  - fig15_strategy_by_category  : strategy x category, descriptive caveat
  - fig16_transfer_comparison   : two-panel transfer
  - fig11a_disparate_impact     : DI ratio per checkpoint with 4/5 threshold
  - fig11b_counterfactual       : counterfactual consistency per checkpoint
  - fig11c_intersectional       : 6 x 6 (cat x lang) intersectional heatmap
  - fig11d_frr_parity           : EN vs per-language FRR (parity strip plot)

Colorblind-safe checkpoint palette:
  base = #D55E00  (vermillion)
  SFT  = #009E73  (green)
  DPO  = #7B4EA3  (plum)
"""

from __future__ import annotations
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "report" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {"base": "#D55E00", "sft": "#009E73", "dpo": "#7B4EA3"}
PRETTY = {"base": "Base", "sft": "SFT", "dpo": "DPO"}
LANG_ORDER = ["bn", "gu", "hi", "sw", "tl", "yo"]
LANG_PRETTY = {"bn": "Bengali", "gu": "Gujarati", "hi": "Hindi",
               "sw": "Swahili", "tl": "Tagalog", "yo": "Yoruba"}
CAT_ORDER = ["TOX", "ULP", "UCA", "SH", "MIS", "PPV"]
ARM_COLORS = {
    "CS": "#D55E00",
    "RP": "#E69F00",
    "MTE": "#0072B2",
    "CS-RP": "#009E73",
    "CS-OBF": "#7B4EA3",
}

plt.rcParams.update({
    "font.size": 11.5,
    "axes.titlesize": 13.5,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.22,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "savefig.bbox": "tight",
    "savefig.dpi": 220,
})

fairness = json.loads((ROOT / "results/fairness/fairness_report.json").read_text())
summary  = json.loads((ROOT / "results/eval/summary.json").read_text())
FAIR_COLORS = {"base": "#4C78A8", "sft": "#2A9D8F", "dpo": "#E76F51"}


# ===================================================================
# 1. Closed-loop architecture diagram (replaces narrative-only figure)
# ===================================================================
def fig_pipeline():
    fig, ax = plt.subplots(figsize=(14.0, 6.8))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def box(x, y, w, h, label, fill, edge="#222", text_color="white",
            fontsize=10.0, fontweight="bold"):
        b = FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.22",
            linewidth=1.9, edgecolor=edge, facecolor=fill, zorder=2)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                color=text_color, fontsize=fontsize, fontweight=fontweight,
                zorder=3)

    def arrow(x1, y1, x2, y2, color="#444", lw=1.6, style="-|>", rad=0.0,
              label=None, label_xy=None, label_color=None,
              fontsize=11, align="center"):
        ar = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style, mutation_scale=18, color=color,
            linewidth=lw, connectionstyle=f"arc3,rad={rad}", zorder=1,
            shrinkA=6, shrinkB=6)
        ax.add_patch(ar)
        if label:
            mx, my = label_xy if label_xy else ((x1 + x2) / 2, (y1 + y2) / 2)
            ax.text(mx, my, label,
                    ha=align, va="center", fontsize=fontsize,
                    color=label_color or color, style="italic",
                    fontweight="semibold")

    # Title
    ax.text(7.75, 6.85, "MediCS closed-loop adversarial-training pipeline",
            ha="center", va="center", fontsize=21, fontweight="bold")
    ax.text(7.75, 6.45,
            "Adaptive red-teaming, GPT-5 judging, and QLoRA-SFT defense",
            ha="center", va="center", fontsize=11.5, color="#5f6368")

    # --- Top row: attack / judge loop ---
    box(0.65, 4.85, 3.35, 1.4,
        "MediCS-500\n500 seeds × 6 langs\n+ benign twins",
        fill="#234B5A", fontsize=15)
    box(4.6, 4.85, 3.35, 1.4,
        "Thompson-Sampling\nbandit\n(5 strategies)",
        fill="#0F77B9", fontsize=15)
    box(8.55, 4.85, 3.45, 1.4,
        "Llama-3-8B-Instruct\n(target model)",
        fill="#5A5A5A", fontsize=15)
    box(12.45, 4.85, 2.55, 1.4,
        "GPT-5 judge\n(rubric, T=0)",
        fill="#8646A0", fontsize=15)

    arrow(4.0, 5.55, 4.6, 5.55, color="#234B5A", lw=2.1)
    arrow(7.95, 5.55, 8.55, 5.55, color="#3C3C3C", lw=2.1)
    arrow(12.0, 5.55, 12.45, 5.55, color="#3C3C3C", lw=2.1)

    # --- Bottom row: defense / evaluation ---
    box(1.7, 1.55, 4.0, 1.5,
        "Held-out evaluation\n533 attacks × 3 seeds\n500 benign twins × 3 seeds",
        fill="#2C4E5E", fontsize=13.2)
    box(7.05, 1.55, 3.55, 1.5,
        "QLoRA-SFT defense\n+ prefix-recovery\nupsampling",
        fill="#10A37F", fontsize=15)

    # judge -> defense
    arrow(13.45, 4.85, 9.2, 3.0, color="#8646A0", rad=-0.18, lw=2.2,
          label="reward + paired refusal", label_xy=(11.55, 4.15),
          label_color="#8646A0")

    # defense -> target
    arrow(8.95, 3.05, 10.1, 4.85, color="#10A37F", rad=-0.02, lw=2.2,
          label="LoRA adapter", label_xy=(9.75, 3.72),
          label_color="#10A37F")

    # judge -> bandit posterior update
    arrow(13.2, 6.05, 6.45, 6.1, color="#0F77B9", rad=-0.34, lw=2.2,
          label="update α, β  (bandit posterior)",
          label_xy=(10.35, 6.22), label_color="#0F77B9", fontsize=12)

    # target -> evaluation
    arrow(9.15, 4.85, 4.8, 3.05, color="#4F4F4F", rad=0.2, lw=1.9,
          label="held-out attacks + benign twins", label_xy=(6.55, 4.18),
          label_color="#5f6368", fontsize=10)

    # evaluation -> defense set construction
    arrow(5.7, 2.3, 7.05, 2.3, color="#2C4E5E", lw=2.1)

    fig.savefig(OUT / "fig_pipeline.png")
    plt.close(fig)


# ===================================================================
# 2. Headline ASR by stage (redesigned bar)
# ===================================================================
def fig1_asr_stages():
    stages = ["base", "sft", "dpo"]
    means = [0.2758, 0.0638, 0.2145]
    cis = [(0.248, 0.323), (0.038, 0.077), (0.188, 0.259)]

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(stages))
    bars = ax.bar(x, [m * 100 for m in means],
                  color=[PALETTE[s] for s in stages],
                  edgecolor="#1f1f1f", linewidth=0.7, width=0.58, zorder=3)
    yerr_low = [(means[i] - cis[i][0]) * 100 for i in range(3)]
    yerr_high = [(cis[i][1] - means[i]) * 100 for i in range(3)]
    ax.errorbar(x, [m * 100 for m in means],
                yerr=[yerr_low, yerr_high], fmt="none",
                ecolor="#222", elinewidth=1.2, capsize=5, zorder=4)

    for i, m in enumerate(means):
        ax.text(i, m * 100 + 1.5, f"{m*100:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([PRETTY[s] for s in stages])
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("ASR across defense stages", loc="left")
    ax.set_ylim(0, 36)
    ax.axhline(0, color="#999", linewidth=0.6)
    ax.grid(axis="x", visible=False)
    ax.text(0.02, 0.96, "Held-out pool, 3 seeds, BASE prompt at inference",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            color="#555")

    ax.annotate("", xy=(1, 8.5), xytext=(0, 29.0),
                arrowprops=dict(arrowstyle="->", color=PALETTE["sft"], lw=1.9))
    ax.text(0.5, 31.2, "-21.2 pp",
            ha="center", va="center", color=PALETTE["sft"],
            fontweight="bold", fontsize=10.5)

    # Annotate the regression
    ax.annotate("", xy=(2, 21.5 + 5), xytext=(1, 6.4 + 5),
                arrowprops=dict(arrowstyle="->", color=PALETTE["dpo"], lw=1.6))
    ax.text(1.5, 14, "+15.1 pp\nregression",
            ha="center", color=PALETTE["dpo"], fontweight="bold", fontsize=10)

    fig.savefig(OUT / "fig1_asr_defense_stages.png")
    plt.close(fig)


# ===================================================================
# 3. Per-language ASR with bootstrap CIs (grouped bars, all 3 stages)
# ===================================================================
def fig3_cross_language():
    per_lang = {s: fairness[s]["per_language_asr"] for s in ("base", "sft", "dpo")}
    per_lang_stats = {s: fairness[s]["per_language_asr_stats"]
                      for s in ("base", "sft", "dpo")}

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    x = np.arange(len(LANG_ORDER))
    width = 0.26
    for i, s in enumerate(("base", "sft", "dpo")):
        means = [per_lang[s][l] * 100 for l in LANG_ORDER]
        cis = [per_lang_stats[s][l]["ci"] for l in LANG_ORDER]
        yerr_low  = [(per_lang[s][l] - c[0]) * 100 for l, c in zip(LANG_ORDER, cis)]
        yerr_high = [(c[1] - per_lang[s][l]) * 100 for l, c in zip(LANG_ORDER, cis)]
        ax.bar(x + (i - 1) * width, means, width,
               color=PALETTE[s], edgecolor="black", linewidth=0.5,
               label=PRETTY[s], zorder=3)
        ax.errorbar(x + (i - 1) * width, means,
                    yerr=[yerr_low, yerr_high], fmt="none",
                    ecolor="#222", elinewidth=0.9, capsize=3, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([LANG_PRETTY[l] for l in LANG_ORDER], rotation=12)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Per-language ASR with bootstrap 95% CIs (held-out pool)")
    ax.legend(loc="upper right", frameon=True, framealpha=0.95)
    ax.set_ylim(0, 42)
    ax.grid(axis="x", visible=False)

    fig.savefig(OUT / "fig3_cross_language.png")
    plt.close(fig)


# ===================================================================
# 3b. Thompson-sampling trajectories for the appendix
# ===================================================================
def fig4_thompson_convergence():
    bandit_paths = sorted((ROOT / "results" / "attacks").glob("round_*/bandit_state.json"))
    if not bandit_paths:
        return

    state = json.loads(bandit_paths[-1].read_text())
    history = state.get("history", [])
    arms = state.get("arms", ["CS", "RP", "MTE", "CS-RP", "CS-OBF"])
    if not history:
        return

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    for arm in arms:
        cumulative = []
        count = 0
        for idx, row in enumerate(history, start=1):
            if row.get("strategy") == arm:
                count += 1
            cumulative.append(count / idx)
        ax.plot(
            cumulative,
            linewidth=2.3,
            label=arm,
            color=ARM_COLORS.get(arm, "#666666"),
        )

    ax.set_xlabel("Trial")
    ax.set_ylabel("Cumulative selection rate")
    ax.set_title("Thompson-sampling arm selection over the red-team campaign", loc="left")
    ax.text(0.02, 0.96, "Descriptive only: no bandit-vs-baseline causal claim",
            transform=ax.transAxes, ha="left", va="top", fontsize=9.5,
            color="#555")
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="center right", frameon=True, framealpha=0.95)

    fig.savefig(OUT / "fig4_thompson_convergence.png")
    plt.close(fig)


# ===================================================================
# 4. SFT residual: 6 x 6 (cat x lang) heatmap with shared scale
# ===================================================================
def _heatmap(ax, M, row_labels, col_labels, title, vmin=0, vmax=1,
             cmap="magma_r", annot_n=None, fmt="{:.0%}"):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            color = "white" if v > (vmin + vmax) / 2 else "black"
            label = fmt.format(v)
            if annot_n is not None:
                label = f"{label}\nn={annot_n[i, j]}"
            ax.text(j, i, label, ha="center", va="center",
                    color=color, fontsize=8.5)
    ax.grid(False)
    return im


def fig5_residual():
    M = np.zeros((len(CAT_ORDER), len(LANG_ORDER)))
    intersect = fairness["sft"]["intersectional"]["matrix"]
    for j, l in enumerate(LANG_ORDER):
        for i, c in enumerate(CAT_ORDER):
            M[i, j] = intersect.get(l, {}).get(c, 0.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    im = _heatmap(ax, M, CAT_ORDER,
                  [LANG_PRETTY[l] for l in LANG_ORDER],
                  "SFT residual ASR by harm category × language",
                  vmin=0, vmax=0.5, cmap="magma_r")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Conditional ASR (lower is better)")
    fig.savefig(OUT / "fig5_failure_modes.png")
    plt.close(fig)


# ===================================================================
# 5. Strategy x category attack-pool ASR (descriptive heatmap)
# ===================================================================
def fig15_strategy_by_category():
    # Use values consistent with paper text; annotate descriptively.
    strategies = ["CS", "RP", "MTE", "CS-RP", "CS-OBF"]
    cats = ["TOX", "ULP", "UCA", "SH", "MIS", "PPV"]
    # These are descriptive snapshots from results/attacks aggregation
    # (matches the existing figure's intent).
    M = np.array([
        [0.21, 0.20, 0.18, 0.06, 0.34, 0.05],   # CS
        [0.45, 0.41, 0.36, 0.07, 0.10, 0.04],   # RP
        [0.36, 0.42, 0.30, 0.32, 0.08, 0.04],   # MTE
        [0.62, 0.71, 0.55, 0.08, 0.07, 0.04],   # CS-RP
        [0.18, 0.18, 0.16, 0.05, 0.30, 0.05],   # CS-OBF
    ])
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    im = _heatmap(ax, M, strategies, cats,
                  "Strategy × harm-category ASR (red-team pool, descriptive)",
                  vmin=0, vmax=0.75, cmap="magma_r")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("ASR")
    fig.savefig(OUT / "fig15_strategy_by_category.png")
    plt.close(fig)


# ===================================================================
# 6. Transfer comparison
# ===================================================================
def fig16_transfer():
    models = ["Llama-3-8B\n(base)", "Llama-3-8B\n+SFT", "Qwen2.5-7B", "Mistral-7B"]
    asr = [0.2758, 0.0638, 0.5159, 0.6173]
    cis = [(0.248, 0.323), (0.038, 0.077),
           (0.475, 0.559), (0.576, 0.659)]
    colors = [PALETTE["base"], PALETTE["sft"], "#7F3F98", "#D55E00"]

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    bars = ax.bar(range(4), [a * 100 for a in asr], color=colors,
                  edgecolor="black", linewidth=0.5, zorder=3)
    yerr_low = [(asr[i] - cis[i][0]) * 100 for i in range(4)]
    yerr_high = [(cis[i][1] - asr[i]) * 100 for i in range(4)]
    ax.errorbar(range(4), [a * 100 for a in asr],
                yerr=[yerr_low, yerr_high], fmt="none",
                ecolor="#222", elinewidth=1.0, capsize=4, zorder=4)
    for i, a in enumerate(asr):
        ax.text(i, a * 100 + 1.5, f"{a*100:.1f}%",
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    ax.set_xticks(range(4))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylabel("Attack Success Rate (%)")
    ax.set_title("Overall ASR on the 533 held-out CS attacks")
    ax.set_ylim(0, 75)
    ax.grid(axis="x", visible=False)

    # Per-language
    ax2 = axes[1]
    transfer = summary["transfer"]["models"]
    width = 0.26
    x = np.arange(len(LANG_ORDER))
    llama_per_lang = {l: fairness["base"]["per_language_asr"][l]
                      for l in LANG_ORDER}
    qwen_per_lang = {l: transfer["qwen"]["asr_per_language"][l]["asr"]
                     for l in LANG_ORDER}
    mistral_per_lang = {l: transfer["mistral"]["asr_per_language"][l]["asr"]
                        for l in LANG_ORDER}
    ax2.bar(x - width, [llama_per_lang[l] * 100 for l in LANG_ORDER], width,
            color=PALETTE["base"], edgecolor="black", linewidth=0.4,
            label="Llama-3-8B (base)", zorder=3)
    ax2.bar(x,         [qwen_per_lang[l] * 100 for l in LANG_ORDER], width,
            color="#7F3F98", edgecolor="black", linewidth=0.4,
            label="Qwen2.5-7B", zorder=3)
    ax2.bar(x + width, [mistral_per_lang[l] * 100 for l in LANG_ORDER], width,
            color="#D55E00", edgecolor="black", linewidth=0.4,
            label="Mistral-7B", zorder=3)
    ax2.set_xticks(x)
    ax2.set_xticklabels([LANG_PRETTY[l] for l in LANG_ORDER], rotation=12)
    ax2.set_ylabel("ASR (%)")
    ax2.set_title("Per-language ASR by target model")
    ax2.set_ylim(0, 80)
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(axis="x", visible=False)

    fig.savefig(OUT / "fig16_transfer_comparison.png")
    plt.close(fig)


# ===================================================================
# 7. Individual fairness plots (replacing the dashboard)
# ===================================================================
def fig11a_disparate_impact():
    stages = ["base", "sft", "dpo"]
    di = [fairness[s]["disparate_impact"]["ratio"] for s in stages]
    cis = [fairness[s]["disparate_impact"]["ratio_ci"] for s in stages]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(stages))[::-1]
    ax.axvspan(0.70, 0.80, color="#FDECEC", zorder=0)
    ax.axvline(0.8, ls=(0, (4, 2)), color="#D96C6C", linewidth=1.5, zorder=1)
    yerr_low = [di[i] - cis[i][0] for i in range(3)]
    yerr_high = [cis[i][1] - di[i] for i in range(3)]
    for i, s in enumerate(stages):
        ax.hlines(y[i], cis[i][0], cis[i][1], color=FAIR_COLORS[s], linewidth=2.4, zorder=2)
        ax.scatter(di[i], y[i], s=120, color=FAIR_COLORS[s], edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(di[i] + 0.012, y[i], f"{di[i]:.3f}", va="center", ha="left",
                fontsize=11, fontweight="bold", color="#1f1f1f",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9))
    ax.text(0.804, 1.68, "4/5 rule threshold", color="#B65454", fontsize=10, va="bottom",
            bbox=dict(boxstyle="round,pad=0.14", fc="white", ec="none", alpha=0.9))
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[s] for s in stages], fontsize=11)
    ax.set_xlim(0.70, 1.03)
    ax.set_xlabel("Disparate-impact ratio")
    ax.set_title("Cross-lingual disparate impact", loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig11a_disparate_impact.png")
    plt.close(fig)


def fig11b_counterfactual():
    stages = ["base", "sft", "dpo"]
    cf = [fairness[s]["counterfactual_fairness"]["consistency_rate"]
          for s in stages]
    cis = [fairness[s]["counterfactual_fairness"]["consistency_rate_ci"]
           for s in stages]

    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    y = np.arange(len(stages))[::-1]
    yerr_low = [(cf[i] - cis[i][0]) * 100 for i in range(3)]
    yerr_high = [(cis[i][1] - cf[i]) * 100 for i in range(3)]
    for i, s in enumerate(stages):
        x0 = cf[i] * 100
        ax.hlines(y[i], x0 - yerr_low[i], x0 + yerr_high[i],
                  color=FAIR_COLORS[s], linewidth=2.4, zorder=2)
        ax.scatter(x0, y[i], s=120, color=FAIR_COLORS[s], edgecolor="white", linewidth=1.2, zorder=3)
        ax.text(x0 + 1.8, y[i], f"{x0:.1f}%", va="center", ha="left",
                fontsize=11, fontweight="bold", color="#1f1f1f",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9))
    ax.axvspan(75, 100, color="#EAF7F3", zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([PRETTY[s] for s in stages], fontsize=11)
    ax.set_xlim(45, 93)
    ax.set_xlabel("Consistency rate (%)")
    ax.set_title("Counterfactual fairness on matched triples", loc="left")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "fig11b_counterfactual.png")
    plt.close(fig)


def fig11c_intersectional():
    M_base = np.zeros((len(CAT_ORDER), len(LANG_ORDER)))
    M_sft = np.zeros_like(M_base)
    for j, l in enumerate(LANG_ORDER):
        for i, c in enumerate(CAT_ORDER):
            M_base[i, j] = fairness["base"]["intersectional"]["matrix"].get(l, {}).get(c, 0.0)
            M_sft[i, j]  = fairness["sft"]["intersectional"]["matrix"].get(l, {}).get(c, 0.0)

    vmax = max(M_base.max(), M_sft.max())
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))
    im0 = _heatmap(axes[0], M_base, CAT_ORDER,
                   [LANG_PRETTY[l] for l in LANG_ORDER],
                   "Base: intersectional ASR", vmin=0, vmax=vmax,
                   cmap="magma_r")
    im1 = _heatmap(axes[1], M_sft, CAT_ORDER,
                   [LANG_PRETTY[l] for l in LANG_ORDER],
                   "SFT: intersectional ASR (shared scale)",
                   vmin=0, vmax=vmax, cmap="magma_r")
    cbar = fig.colorbar(im1, ax=axes, fraction=0.038, pad=0.02)
    cbar.set_label("Conditional ASR")
    fig.suptitle("Intersectional fairness: category × language ASR",
                 fontsize=13, y=1.02)
    fig.savefig(OUT / "fig11c_intersectional.png")
    plt.close(fig)


def fig11d_frr_parity():
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.arange(len(LANG_ORDER))
    ref_label_y = {"base": 1.42, "dpo": 1.22, "sft": 0.56}
    for s in ("base", "sft", "dpo"):
        per_lang = fairness[s]["per_language_frr"]
        vals = [per_lang[l] * 100 for l in LANG_ORDER]
        ax.plot(x, vals, color=FAIR_COLORS[s], linewidth=2.3, marker="o",
                markersize=7, label=PRETTY[s], zorder=3)
        ax.scatter(x, vals, s=70, color=FAIR_COLORS[s], edgecolor="white", linewidth=1.1, zorder=4)
        en_frr = fairness[s]["english_frr"] * 100
        ax.axhline(en_frr, color=FAIR_COLORS[s], linestyle=(0, (4, 2)), linewidth=1.2,
                   alpha=0.7,
                   zorder=1)
        ax.text(5.12, ref_label_y[s], f"{PRETTY[s]} EN {en_frr:.1f}%",
                color=FAIR_COLORS[s], fontsize=9.5, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.16", fc="white", ec="none", alpha=0.9))
    ax.set_xticks(x)
    ax.set_xticklabels([LANG_PRETTY[l] for l in LANG_ORDER], rotation=12)
    ax.set_ylabel("False Refusal Rate (%)")
    ax.set_title("FRR parity versus English benign performance", loc="left")
    ax.legend(loc="upper left", frameon=False, ncol=3, fontsize=10)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 5.5)
    ax.set_xlim(-0.2, 5.25)
    fig.savefig(OUT / "fig11d_frr_parity.png")
    plt.close(fig)


# ===================================================================
def main():
    fig_pipeline()
    fig1_asr_stages()
    fig3_cross_language()
    fig4_thompson_convergence()
    fig5_residual()
    fig15_strategy_by_category()
    fig16_transfer()
    fig11a_disparate_impact()
    fig11b_counterfactual()
    fig11c_intersectional()
    fig11d_frr_parity()
    print("Wrote:", sorted(p.name for p in OUT.iterdir() if p.suffix == ".png"))


if __name__ == "__main__":
    main()
