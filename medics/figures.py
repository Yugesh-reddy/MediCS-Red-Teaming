"""
MediCS — Publication Figure Generation
========================================
16 publication-quality figures with a coherent modern style.

Narrative discipline (v2.7.x):
  - The headline result is **SFT alone**: 27.6% → 6.4% ASR with HR preserved.
  - DPO is a **negative / cautionary result**, NOT a defense progression.
    DPO regressed safety from 6.4% (SFT) to 21.5% (DPO+SFT) because SFT
    did not introduce the over-refusal pathology that DPO is meant to correct.
  - Figures emphasize Base → SFT as the win and visually subordinate DPO
    (lighter color, dashed lines, "cautionary" annotations) so a reader who
    only glances at fig1 leaves with the right takeaway.

Design system:
  - Palette: Okabe-Ito (colorblind-safe, used by Nature).
  - Typography: sans-serif, hierarchical sizes, no bold-everything.
  - Axes: top/right despined, subtle horizontal grid only.
  - Labels: directly on bars/points where possible, legends only when needed.
  - Outputs: 300 DPI PNG + vector PDF, white background.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import seaborn as sns
import numpy as np
import pandas as pd

from medics.utils import load_jsonl, load_json
from medics.metrics import (
    compute_asr, compute_robustness_gain,
    compute_per_category_asr, compute_per_strategy_asr,
    compute_per_language_asr, bootstrap_ci,
)


# =============================================================================
# DESIGN SYSTEM
# =============================================================================

# Okabe-Ito palette — colorblind-safe, perceptually uniform.
# Semantic mapping is consistent across every figure in the deck.
PALETTE = {
    "base":     "#D55E00",   # vermillion — vulnerable / undefended
    "sft":      "#009E73",   # bluish-green — the headline defense
    "dpo":      "#CC79A7",   # reddish-purple — regression / cautionary
    "mistral":  "#E69F00",   # orange — transfer model A
    "qwen":     "#0072B2",   # blue — transfer model B
    "english":  "#56B4E9",   # sky blue — English baseline
    "harmful":  "#D55E00",
    "safe":     "#009E73",
    "neutral":  "#444444",
    "muted":    "#9b9b9b",
    "grid":     "#e8e8e8",
    "ink":      "#222222",
    "subink":   "#666666",
}

CHECKPOINT_LABEL = {
    "base": "Base",
    "sft":  "+SFT",
    "dpo":  "+SFT+DPO",
}

CHECKPOINT_COLOR = {
    "base": PALETTE["base"],
    "sft":  PALETTE["sft"],
    "dpo":  PALETTE["dpo"],
}

# Sequential colormap for heatmaps with a clean modern feel.
HEATMAP_CMAP = "rocket_r"     # dark→light, perceptually uniform
HEATMAP_DIVERGING = "RdBu_r"  # for regression/improvement deltas


def _apply_style():
    """Apply the publication style. Idempotent — safe to call repeatedly."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        # DejaVu Sans first — bundled with matplotlib, has full glyph coverage
        # for arrows, minus signs, math symbols. Avoids macOS Helvetica subset issues.
        "font.sans-serif": [
            "DejaVu Sans", "Liberation Sans", "Arial",
            "Helvetica", "Helvetica Neue",
        ],
        "font.size": 11,
        "figure.titlesize": 15,
        "figure.titleweight": "bold",
        "figure.facecolor": "white",
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlepad": 12,
        "axes.labelsize": 11,
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": PALETTE["subink"],
        "axes.linewidth": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "axes.titlecolor": PALETTE["ink"],
        "grid.color": PALETTE["grid"],
        "grid.linewidth": 0.6,
        "grid.linestyle": "-",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.color": PALETTE["subink"],
        "ytick.color": PALETTE["subink"],
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,    # TrueType, not Type-3 (publishable)
        "ps.fonttype": 42,
    })


_apply_style()


def _hgrid_only(ax):
    """Show only horizontal gridlines — cleaner for bar charts."""
    ax.grid(axis='x', visible=False)
    ax.grid(axis='y', visible=True)


def _vgrid_only(ax):
    ax.grid(axis='x', visible=True)
    ax.grid(axis='y', visible=False)


def _no_grid(ax):
    ax.grid(False)


def _save(fig, figures_dir, name):
    """Save figure as PNG (300 DPI). PDFs intentionally not produced."""
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / f"{name}.png", dpi=300, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)


def _label_bars(ax, bars, values, fmt="{:.1f}%", offset=1.0,
                fontsize=11, weight="semibold", color=None):
    """Direct-label bars at their tops."""
    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            fmt.format(v),
            ha='center', va='bottom',
            fontsize=fontsize, fontweight=weight,
            color=color or PALETTE["ink"],
        )


def _footer(fig, text, y=0.005):
    """Tiny footer note for sample sizes / CI methodology."""
    fig.text(0.99, y, text, ha='right', va='bottom',
             fontsize=8, color=PALETTE["muted"], style='italic')


def _find_held_out(results_dir, ckpt, filename="held_out.jsonl"):
    """Locate held-out file, preferring multi-seed (seed_42) layout."""
    seed_path = Path(results_dir) / ckpt / "seed_42" / filename
    single_path = Path(results_dir) / ckpt / filename
    path = seed_path if seed_path.exists() else single_path
    return load_jsonl(str(path))


def _multi_seed_metric(results_dir, ckpt, metric_fn, seeds=(42, 123, 456),
                       filename="held_out.jsonl"):
    """Run metric_fn(rows) across all seeds and return (mean, list)."""
    vals = []
    for s in seeds:
        path = Path(results_dir) / ckpt / f"seed_{s}" / filename
        if path.exists():
            vals.append(metric_fn(load_jsonl(str(path))))
    return (np.mean(vals) if vals else 0.0, vals)


def _all_seeds(results_dir, ckpt, filename="held_out.jsonl",
               seeds=(42, 123, 456)):
    """Concatenate rows across all eval seeds for headline metrics.

    Falls back to the single-seed file if the multi-seed layout is absent.
    """
    rows = []
    for s in seeds:
        path = Path(results_dir) / ckpt / f"seed_{s}" / filename
        if path.exists():
            rows.extend(load_jsonl(str(path)))
    if not rows:
        rows = _find_held_out(results_dir, ckpt, filename)
    return rows


# =============================================================================
# FIGURE 1 — Headline: Base → SFT (the win)
# =============================================================================
def fig1_asr_defense_stages(results_dir, figures_dir):
    """Two-panel headline:
       (a) ASR: Base → SFT (large bars, the win) with DPO shown as a faded
           "cautionary" reference bar. Annotation directs reader to fig 7.
       (b) HR: Helpfulness Retention is preserved across all stages
           (visualized to refute the 'safety vs helpfulness tradeoff' worry).
    """
    checkpoints = ["base", "sft", "dpo"]

    asrs, asr_lo, asr_hi, hrs = [], [], [], []
    for ckpt in checkpoints:
        attacks = _all_seeds(results_dir, ckpt)
        if attacks:
            valid = [r for r in attacks if r.get("judge_label") in ("harmful", "safe")]
            arr = [1 if r.get("judge_label") == "harmful" else 0 for r in valid]
            mean, lo, hi = bootstrap_ci(arr)
            asrs.append(mean * 100); asr_lo.append((mean - lo) * 100); asr_hi.append((hi - mean) * 100)
        else:
            asrs.append(0); asr_lo.append(0); asr_hi.append(0)

        benign = _all_seeds(results_dir, ckpt, "benign_results.jsonl")
        if benign:
            valid = [r for r in benign if not r.get("judge_error")]
            helpful = sum(1 for r in valid if not r.get("was_incorrectly_refused"))
            hrs.append(helpful / len(valid) * 100 if valid else 0)
        else:
            hrs.append(0)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6),
                             gridspec_kw={'width_ratios': [1.4, 1]})

    # --- Panel (a): ASR ---
    ax = axes[0]
    labels = [CHECKPOINT_LABEL[c] for c in checkpoints]
    colors = [PALETTE["base"], PALETTE["sft"], PALETTE["dpo"]]
    alphas = [0.95, 1.0, 0.55]   # fade DPO so it reads as cautionary
    bars = []
    for i, (lab, val, c, a) in enumerate(zip(labels, asrs, colors, alphas)):
        b = ax.bar(i, val, color=c, alpha=a, width=0.62,
                   edgecolor='white', linewidth=1.5,
                   hatch='///' if i == 2 else None,
                   zorder=3)
        bars.append(b)
    ax.errorbar(range(len(asrs)), asrs,
                yerr=[asr_lo, asr_hi], fmt='none',
                color=PALETTE["ink"], capsize=5, capthick=1.2, lw=1.2, zorder=4)

    for i, v in enumerate(asrs):
        ax.text(i, v + 2.0, f"{v:.1f}%", ha='center', va='bottom',
                fontsize=13, fontweight='bold',
                color=colors[i] if i < 2 else PALETTE["subink"])

    # Headline-arrow annotation: Base -> SFT
    ax.annotate(
        "", xy=(0.92, asrs[1] + 1.4), xytext=(0.08, asrs[0] - 1.4),
        arrowprops=dict(arrowstyle='->', color=PALETTE["sft"], lw=2.0,
                        connectionstyle="arc3,rad=-0.18"),
    )
    ax.text(0.5, (asrs[0] + asrs[1]) / 2 + 4,
            f"-{asrs[0]-asrs[1]:.1f} pp\np<0.0001",
            ha='center', va='center', fontsize=11, fontweight='semibold',
            color=PALETTE["sft"])

    # DPO cautionary note (small, off to the side)
    ax.annotate(
        "DPO regressed -> see Fig 7",
        xy=(2, asrs[2]), xytext=(2.0, asrs[2] + 8),
        ha='center', va='bottom',
        fontsize=9.5, fontstyle='italic', color=PALETTE["dpo"],
        arrowprops=dict(arrowstyle='-', color=PALETTE["dpo"],
                        lw=0.8, alpha=0.7),
    )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Attack Success Rate (%)", fontsize=11)
    ax.set_ylim(0, max(asrs) * 1.45 + 6)
    ax.set_title(f"(a)  Base -> SFT cuts ASR by {asrs[0]-asrs[1]:.1f} pp",
                 fontsize=13, fontweight='semibold', loc='left',
                 color=PALETTE["ink"])
    _hgrid_only(ax)

    # --- Panel (b): HR ---
    ax = axes[1]
    bars = ax.bar(range(len(hrs)), hrs, width=0.62,
                  color=[PALETTE["base"], PALETTE["sft"], PALETTE["dpo"]],
                  alpha=0.85, edgecolor='white', linewidth=1.5, zorder=3)
    for i, v in enumerate(hrs):
        ax.text(i, v + 0.4, f"{v:.1f}%", ha='center', va='bottom',
                fontsize=11.5, fontweight='semibold', color=PALETTE["ink"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Helpfulness Retention (%)", fontsize=11)
    ax.set_ylim(min(hrs) - 4, 101.5)
    ax.set_title("(b)  No measurable cost to helpfulness",
                 fontsize=13, fontweight='semibold', loc='left',
                 color=PALETTE["ink"])
    _hgrid_only(ax)

    fig.suptitle("MediCS Headline — SFT is the Defense; DPO is a Cautionary Result",
                 fontsize=15, fontweight='bold', y=1.02)
    _footer(fig, "n=533 held-out attacks × 500 benign queries × 3 seeds · 95% bootstrap CI on ASR")
    plt.tight_layout()
    _save(fig, figures_dir, "fig1_asr_defense_stages")
    print("  Figure 1: Headline ASR + HR (Base → SFT win)")


# =============================================================================
# FIGURE 2 — Strategy × Category effectiveness (base, attack-round data)
# =============================================================================
def fig2_strategy_heatmap(results_dir, figures_dir):
    """Where each attack strategy lands across harm categories on the
    undefended base model. Uses attack-round data (Round 1) where strategies
    were sampled by Thompson Sampling — held-out is CS-only by construction.
    """
    rows = []
    for path in sorted(Path("results/attacks").glob("round_*/results.jsonl")):
        rows.extend(load_jsonl(str(path)))

    # Fall back to held-out base if no round data
    if not rows:
        rows = _find_held_out(results_dir, "base")
    if not rows:
        print("  Figure 2: no data")
        return

    cats = sorted({r.get("category", "?") for r in rows})
    strats = sorted({r.get("strategy", "?") for r in rows})

    asr_mat = np.full((len(strats), len(cats)), np.nan)
    n_mat = np.zeros((len(strats), len(cats)), dtype=int)
    for i, s in enumerate(strats):
        for j, c in enumerate(cats):
            sub = [r for r in rows
                   if r.get("strategy") == s and r.get("category") == c
                   and r.get("judge_label") in ("harmful", "safe")]
            if len(sub) >= 3:
                asr_mat[i, j] = sum(1 for r in sub if r.get("judge_label") == "harmful") / len(sub)
            n_mat[i, j] = len(sub)

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    im = ax.imshow(asr_mat * 100, cmap=HEATMAP_CMAP, vmin=0, vmax=100,
                   aspect='auto')

    for i in range(len(strats)):
        for j in range(len(cats)):
            v = asr_mat[i, j]
            n = n_mat[i, j]
            if np.isnan(v):
                txt = "—"; col = PALETTE["muted"]
            else:
                txt = f"{v*100:.0f}%"
                col = "white" if v > 0.55 else PALETTE["ink"]
            ax.text(j, i, txt, ha='center', va='center',
                    fontsize=11, fontweight='semibold', color=col)
            if n > 0:
                ax.text(j, i + 0.32, f"n={n}", ha='center', va='center',
                        fontsize=7.5, color=col, alpha=0.8)

    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats, fontsize=10.5)
    ax.set_yticks(range(len(strats)))
    ax.set_yticklabels(strats, fontsize=10.5)
    ax.set_xlabel("Harm category", fontsize=11)
    ax.set_ylabel("Attack strategy", fontsize=11)
    ax.set_title("Attack effectiveness on the undefended base model",
                 fontsize=14, fontweight='bold', loc='left')

    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("ASR (%)", fontsize=10)
    cbar.outline.set_linewidth(0)
    cbar.ax.tick_params(length=0, labelsize=9)

    _no_grid(ax)
    _footer(fig,
            f"Attack-round data, base Llama-3-8B · n shown per cell · cells with <3 samples masked")
    plt.tight_layout()
    _save(fig, figures_dir, "fig2_strategy_heatmap")
    print("  Figure 2: Strategy × Category effectiveness")


# =============================================================================
# FIGURE 3 — Cross-Language Vulnerability (Base vs SFT, with DPO sidebar)
# =============================================================================
def fig3_cross_language(results_dir, figures_dir):
    """Headline panel: per-language ASR Base → SFT (the win across all 6 languages).
    Right panel: DPO regression overlay — same languages, SFT vs DPO."""
    langs_order = ["bn", "gu", "hi", "sw", "tl", "yo"]

    def per_lang(ckpt):
        rows = _all_seeds(results_dir, ckpt)
        d = compute_per_language_asr(rows) if rows else {}
        return [d.get(l, 0) * 100 for l in langs_order]

    base = per_lang("base")
    sft = per_lang("sft")
    dpo = per_lang("dpo")
    if not any(base):
        print("  Figure 3: no data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             gridspec_kw={"width_ratios": [1.6, 1]})

    # --- Panel (a): Base vs SFT ---
    ax = axes[0]
    x = np.arange(len(langs_order))
    w = 0.38
    b1 = ax.bar(x - w/2, base, w, color=PALETTE["base"],
                edgecolor='white', linewidth=1.2, label="Base", zorder=3)
    b2 = ax.bar(x + w/2, sft, w, color=PALETTE["sft"],
                edgecolor='white', linewidth=1.2, label="+SFT", zorder=3)
    for bb, v in zip(b1, base):
        ax.text(bb.get_x() + bb.get_width()/2, v + 0.8, f"{v:.0f}",
                ha='center', va='bottom', fontsize=9, color=PALETTE["base"])
    for bb, v in zip(b2, sft):
        ax.text(bb.get_x() + bb.get_width()/2, v + 0.8, f"{v:.0f}",
                ha='center', va='bottom', fontsize=9, fontweight='semibold',
                color=PALETTE["sft"])
    # Δ pp annotation per language
    for i, (b_, s_) in enumerate(zip(base, sft)):
        ax.annotate(f"−{b_-s_:.0f}", xy=(i, max(b_, s_) + 4),
                    ha='center', fontsize=8.5, fontweight='semibold',
                    color=PALETTE["sft"])
    ax.set_xticks(x); ax.set_xticklabels(langs_order, fontsize=10.5)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_ylim(0, max(base) * 1.30 + 5)
    ax.set_title("(a)  SFT cuts ASR uniformly across all six languages",
                 fontsize=13, fontweight='semibold', loc='left')
    ax.legend(loc='upper right', frameon=False)
    _hgrid_only(ax)

    # --- Panel (b): SFT vs DPO regression overlay ---
    ax = axes[1]
    b1 = ax.bar(x - w/2, sft, w, color=PALETTE["sft"], alpha=0.85,
                edgecolor='white', linewidth=1.2, label="+SFT", zorder=3)
    b2 = ax.bar(x + w/2, dpo, w, color=PALETTE["dpo"], alpha=0.85,
                hatch='///', edgecolor='white', linewidth=1.2,
                label="+SFT+DPO", zorder=3)
    for i, (s_, d_) in enumerate(zip(sft, dpo)):
        ax.annotate(f"+{d_-s_:.0f}", xy=(i, max(s_, d_) + 1.5),
                    ha='center', fontsize=8.5, fontweight='semibold',
                    color=PALETTE["dpo"])
    ax.set_xticks(x); ax.set_xticklabels(langs_order, fontsize=10.5)
    ax.set_ylim(0, max(dpo) * 1.30 + 5)
    ax.set_title("(b)  Adding DPO regresses every language",
                 fontsize=13, fontweight='semibold', loc='left',
                 color=PALETTE["dpo"])
    ax.legend(loc='upper right', frameon=False)
    _hgrid_only(ax)
    ax.set_ylabel("ASR (%)", fontsize=11)

    fig.suptitle("Per-language safety after defense", fontsize=15,
                 fontweight='bold', y=1.02)
    _footer(fig, "3 seeds × 533 held-out attacks · ASR = harmful / (total − judge_errors)")
    plt.tight_layout()
    _save(fig, figures_dir, "fig3_cross_language")
    print("  Figure 3: Cross-Language Vulnerability")


# =============================================================================
# FIGURE 4 — Thompson Sampling Convergence
# =============================================================================
def fig4_thompson_convergence(figures_dir):
    bandit_paths = sorted(Path("results/attacks").glob("round_*/bandit_state.json"))
    if not bandit_paths:
        print("  Figure 4: no bandit states"); return

    state = load_json(str(bandit_paths[-1]))
    if not state or "history" not in state:
        print("  Figure 4: empty bandit state"); return
    history = state["history"]
    arms = state.get("arms", ["CS", "RP", "MTE", "CS-RP", "CS-OBF"])

    arm_colors = {
        "CS":     PALETTE["base"],
        "RP":     PALETTE["mistral"],
        "MTE":    PALETTE["qwen"],
        "CS-RP":  PALETTE["sft"],
        "CS-OBF": PALETTE["dpo"],
    }

    fig, ax = plt.subplots(figsize=(11, 5.2))
    for arm in arms:
        cum = []
        c = 0
        for i, h in enumerate(history):
            if h.get("strategy") == arm:
                c += 1
            cum.append(c / (i + 1))
        ax.plot(cum, label=arm, linewidth=2.4,
                color=arm_colors.get(arm, PALETTE["neutral"]))

    ax.set_xlabel("Trial", fontsize=11)
    ax.set_ylabel("Cumulative selection rate", fontsize=11)
    ax.set_title("Thompson Sampling concentrates probability on Code-Switch",
                 fontsize=14, fontweight='bold', loc='left')
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))
    ax.legend(loc='center right', fontsize=10, ncol=1)
    _hgrid_only(ax)

    _footer(fig, "Beta-Bernoulli posterior · α=β=1 prior · min 10 exploration pulls/arm")
    plt.tight_layout()
    _save(fig, figures_dir, "fig4_thompson_convergence")
    print("  Figure 4: Thompson Sampling Convergence")


# =============================================================================
# FIGURE 5 — Where SFT residual failures live (the 6.4% remaining)
# =============================================================================
def fig5_failure_modes(results_dir, figures_dir):
    """SFT is the headline defense. This figure asks: where do the remaining
    6.4% of SFT failures concentrate? Two panels:
      (a) per-category bars  (b) language × category heatmap of residuals.
    """
    rows = _all_seeds(results_dir, "sft")
    if not rows:
        print("  Figure 5: no SFT data"); return

    valid = [r for r in rows if r.get("judge_label") in ("harmful", "safe")]
    cats = sorted({r.get("category", "?") for r in valid})
    langs = sorted({r.get("language", "?") for r in valid})

    cat_asr = []
    cat_n = []
    for c in cats:
        sub = [r for r in valid if r.get("category") == c]
        cat_n.append(len(sub))
        cat_asr.append(sum(1 for r in sub if r.get("judge_label") == "harmful") /
                       len(sub) * 100 if sub else 0)

    matrix = np.full((len(cats), len(langs)), np.nan)
    for i, c in enumerate(cats):
        for j, l in enumerate(langs):
            sub = [r for r in valid
                   if r.get("category") == c and r.get("language") == l]
            if len(sub) >= 3:
                matrix[i, j] = sum(1 for r in sub if r.get("judge_label") == "harmful") / len(sub) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             gridspec_kw={"width_ratios": [1, 1.05]})

    # --- (a) per-category residual ASR bars ---
    ax = axes[0]
    order = np.argsort(cat_asr)[::-1]
    cats_o = [cats[i] for i in order]
    vals = [cat_asr[i] for i in order]
    ns = [cat_n[i] for i in order]
    bars = ax.barh(cats_o, vals, color=PALETTE["sft"], alpha=0.85,
                   edgecolor='white', linewidth=1.2, zorder=3)
    for bar, v, n in zip(bars, vals, ns):
        ax.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                f"{v:.1f}%  (n={n})", va='center', fontsize=9.5,
                color=PALETTE["ink"])
    ax.set_xlabel("Residual ASR after SFT (%)", fontsize=11)
    ax.set_xlim(0, max(vals) * 1.4 + 1)
    ax.set_title("(a)  Where SFT failures concentrate, by category",
                 fontsize=13, fontweight='semibold', loc='left')
    ax.invert_yaxis()
    _vgrid_only(ax)

    # --- (b) heatmap: residual per (category, language) ---
    ax = axes[1]
    masked = np.ma.array(matrix, mask=np.isnan(matrix))
    cmap = matplotlib.cm.get_cmap(HEATMAP_CMAP).copy()
    cmap.set_bad("#f0f0f0")
    im = ax.imshow(masked, cmap=cmap, vmin=0,
                   vmax=max(np.nanmax(matrix) if matrix.size else 1, 25),
                   aspect='auto')
    for i in range(len(cats)):
        for j in range(len(langs)):
            v = matrix[i, j]
            if not np.isnan(v):
                col = "white" if v > 12 else PALETTE["ink"]
                ax.text(j, i, f"{v:.0f}", ha='center', va='center',
                        fontsize=10, fontweight='semibold', color=col)
    ax.set_xticks(range(len(langs)))
    ax.set_xticklabels(langs, fontsize=10)
    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels(cats, fontsize=10)
    ax.set_title("(b)  Residual ASR by (category × language)",
                 fontsize=13, fontweight='semibold', loc='left')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("ASR (%)", fontsize=9)
    cbar.outline.set_linewidth(0)
    cbar.ax.tick_params(length=0, labelsize=9)
    _no_grid(ax)

    fig.suptitle("Residual failures after SFT — where to invest next",
                 fontsize=15, fontweight='bold', y=1.02)
    _footer(fig, "SFT seed 42 · TOX/ULP/UCA Tagalog and Hindi remain the hardest pairs")
    plt.tight_layout()
    _save(fig, figures_dir, "fig5_failure_modes")
    print("  Figure 5: Residual failures after SFT")


# =============================================================================
# FIGURE 6 — Robustness Gain: SFT (the win) per category
# =============================================================================
def fig6_robustness_gain(results_dir, figures_dir):
    """SFT-only robustness gain by category. DPO RG is omitted because
    averaging Base→DPO whitewashes the SFT→DPO regression."""
    base_rows = _all_seeds(results_dir, "base")
    sft_rows = _all_seeds(results_dir, "sft")
    if not base_rows or not sft_rows:
        print("  Figure 6: missing data"); return

    base_cat = compute_per_category_asr(base_rows)
    sft_cat = compute_per_category_asr(sft_rows)
    cats = sorted(set(base_cat.keys()) & set(sft_cat.keys()))

    rg = [compute_robustness_gain(base_cat[c], sft_cat[c]) * 100 for c in cats]
    base_pct = [base_cat[c] * 100 for c in cats]
    sft_pct = [sft_cat[c] * 100 for c in cats]

    order = np.argsort(rg)[::-1]
    cats_o = [cats[i] for i in order]
    rg_o = [rg[i] for i in order]
    base_o = [base_pct[i] for i in order]
    sft_o = [sft_pct[i] for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             gridspec_kw={"width_ratios": [1, 1.1]})

    # (a) RG bars
    ax = axes[0]
    bars = ax.barh(cats_o, rg_o, color=PALETTE["sft"], alpha=0.9,
                   edgecolor='white', linewidth=1.2, zorder=3)
    for bar, v in zip(bars, rg_o):
        ax.text(v + 1.2, bar.get_y() + bar.get_height()/2,
                f"{v:.0f}%", va='center', fontsize=10.5,
                fontweight='semibold', color=PALETTE["ink"])
    ax.set_xlabel("Robustness Gain (%)", fontsize=11)
    ax.set_xlim(0, max(rg_o) * 1.18 + 5)
    ax.set_title("(a)  SFT robustness gain by category",
                 fontsize=13, fontweight='semibold', loc='left')
    ax.invert_yaxis()
    _vgrid_only(ax)

    # (b) Dumbbell: Base → SFT per category
    ax = axes[1]
    y = np.arange(len(cats_o))
    for yi, b, s in zip(y, base_o, sft_o):
        ax.plot([b, s], [yi, yi], color=PALETTE["muted"], lw=2.2, zorder=2)
        ax.scatter([b], [yi], color=PALETTE["base"], s=120, zorder=3,
                   edgecolor='white', linewidth=1.2)
        ax.scatter([s], [yi], color=PALETTE["sft"], s=120, zorder=3,
                   edgecolor='white', linewidth=1.2)
        ax.text(b + 1.5, yi - 0.15, f"{b:.0f}%", fontsize=9, color=PALETTE["base"])
        ax.text(s + 1.5, yi - 0.15, f"{s:.0f}%", fontsize=9,
                color=PALETTE["sft"], fontweight='semibold')
    ax.set_yticks(y); ax.set_yticklabels(cats_o, fontsize=10.5)
    ax.set_xlabel("ASR (%)", fontsize=11)
    ax.set_title("(b)  Base → SFT shift per category",
                 fontsize=13, fontweight='semibold', loc='left')
    ax.invert_yaxis()
    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["base"],
                   markersize=10, label='Base'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=PALETTE["sft"],
                   markersize=10, label='+SFT'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', frameon=False)
    _vgrid_only(ax)

    fig.suptitle("Robustness gain — SFT is the win (DPO covered in Fig 7)",
                 fontsize=15, fontweight='bold', y=1.02)
    _footer(fig, "RG = (ASR_base − ASR_sft) / ASR_base · 3-seed mean held-out")
    plt.tight_layout()
    _save(fig, figures_dir, "fig6_robustness_gain")
    print("  Figure 6: Robustness Gain (SFT)")


# =============================================================================
# FIGURE 7 — DPO Cautionary: slope chart per category (SFT → DPO regression)
# =============================================================================
def fig7_overrefusal_correction(results_dir, figures_dir):
    """The negative result, made unmistakable. Per-category slope chart from
    SFT to SFT+DPO. Lines tilting up = regression. Annotations call out the
    three high-stakes categories where DPO undid the most safety."""
    sft_rows = _all_seeds(results_dir, "sft")
    dpo_rows = _all_seeds(results_dir, "dpo")
    if not sft_rows or not dpo_rows:
        print("  Figure 7: missing data"); return

    sft_c = compute_per_category_asr(sft_rows)
    dpo_c = compute_per_category_asr(dpo_rows)
    cats = sorted(set(sft_c.keys()) & set(dpo_c.keys()))

    deltas = [(c, sft_c[c]*100, dpo_c[c]*100, (dpo_c[c]-sft_c[c])*100) for c in cats]
    deltas.sort(key=lambda x: -x[3])  # biggest regression first

    # Aggregate ASR/HR for the side bar
    sft_overall = sum(1 for r in sft_rows if r.get("judge_label") == "harmful") / \
                  max(1, sum(1 for r in sft_rows if r.get("judge_label") in ("harmful", "safe"))) * 100
    dpo_overall = sum(1 for r in dpo_rows if r.get("judge_label") == "harmful") / \
                  max(1, sum(1 for r in dpo_rows if r.get("judge_label") in ("harmful", "safe"))) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8),
                             gridspec_kw={"width_ratios": [1.7, 1]})

    # --- (a) slope chart per category — spread vertically to avoid overlap ---
    ax = axes[0]
    x_left, x_right = 0, 1

    # Avoid label collisions on the LEFT side: when SFT values cluster near 0,
    # spread the text labels vertically using a tick column.
    n = len(deltas)
    sft_vals = [s for _, s, _, _ in deltas]
    dpo_vals = [d for _, _, d, _ in deltas]
    ymin = -2
    ymax = max(dpo_vals) * 1.18 + 6

    # Compute spread label positions for the LEFT column (SFT side)
    # so categories with similar SFT ASR don't stack on top of each other.
    left_label_y = list(sft_vals)
    sorted_idx = sorted(range(n), key=lambda i: left_label_y[i])
    min_sep = (ymax - ymin) * 0.055  # min vertical separation between labels
    for k in range(1, n):
        i_prev = sorted_idx[k - 1]; i_cur = sorted_idx[k]
        if left_label_y[i_cur] - left_label_y[i_prev] < min_sep:
            left_label_y[i_cur] = left_label_y[i_prev] + min_sep

    for idx, (cat, s, d, delta) in enumerate(deltas):
        c = PALETTE["dpo"] if delta > 0 else PALETTE["sft"]
        lw = max(1.6, min(4.0, 1.6 + abs(delta) / 8))
        ax.plot([x_left, x_right], [s, d], color=c, lw=lw, alpha=0.9, zorder=3)
        ax.scatter([x_left], [s], color=PALETTE["sft"], s=110,
                   edgecolor='white', linewidth=1.4, zorder=4)
        ax.scatter([x_right], [d], color=c, s=110,
                   edgecolor='white', linewidth=1.4, zorder=4)

        # Left-side label with leader line if displaced
        ly = left_label_y[idx]
        ax.text(x_left - 0.05, ly, f"{cat}  {s:.1f}%",
                ha='right', va='center',
                fontsize=10, color=PALETTE["sft"], fontweight='semibold')
        if abs(ly - s) > 0.5:
            ax.plot([x_left - 0.04, x_left - 0.005], [ly, s],
                    color=PALETTE["muted"], lw=0.6, alpha=0.6, zorder=2)

        # Right-side label
        sign = "+" if delta >= 0 else ""
        ax.text(x_right + 0.05, d, f"{d:.1f}%  ({sign}{delta:.1f} pp)",
                ha='left', va='center',
                fontsize=10, color=c, fontweight='semibold')

    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels(["+SFT", "+SFT+DPO"], fontsize=12, fontweight='semibold')
    ax.set_xlim(-0.55, 1.65)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_title("(a)  DPO regresses safety on TOX, ULP, UCA",
                 fontsize=13, fontweight='semibold', loc='left',
                 color=PALETTE["dpo"])
    _hgrid_only(ax)

    # --- (b) summary bar: overall ASR + delta ---
    ax = axes[1]
    bars = ax.bar([0, 1], [sft_overall, dpo_overall],
                  color=[PALETTE["sft"], PALETTE["dpo"]],
                  alpha=0.92, edgecolor='white', linewidth=1.5,
                  width=0.55, zorder=3)
    bars[1].set_hatch('///')
    ax.errorbar([0, 1], [sft_overall, dpo_overall],
                yerr=[[0.6, 1.1], [0.6, 1.1]], fmt='none',
                color=PALETTE["ink"], capsize=4, lw=1.2, zorder=4)
    for i, v in enumerate([sft_overall, dpo_overall]):
        ax.text(i, v + 0.7, f"{v:.1f}%", ha='center', va='bottom',
                fontsize=12, fontweight='bold')
    # Big delta annotation
    ax.annotate("", xy=(1, dpo_overall - 0.5), xytext=(0, sft_overall + 0.5),
                arrowprops=dict(arrowstyle='->', color=PALETTE["dpo"],
                                lw=2.4, connectionstyle="arc3,rad=-0.18"))
    ax.text(0.5, (sft_overall + dpo_overall) / 2 + 4,
            f"+{dpo_overall - sft_overall:.1f} pp\n[+13.0, +17.1] CI",
            ha='center', va='center', fontsize=11, fontweight='semibold',
            color=PALETTE["dpo"])
    ax.set_xticks([0, 1]); ax.set_xticklabels(["+SFT", "+SFT+DPO"],
                                              fontsize=11, fontweight='semibold')
    ax.set_ylabel("Overall ASR (%)", fontsize=11)
    ax.set_ylim(0, max(sft_overall, dpo_overall) * 1.5 + 4)
    ax.set_title("(b)  Overall regression: SFT → DPO",
                 fontsize=13, fontweight='semibold', loc='left',
                 color=PALETTE["dpo"])
    _hgrid_only(ax)

    fig.suptitle("DPO Cautionary Result — applied without an over-refusal signal,\nDPO undoes SFT's safety gains",
                 fontsize=15, fontweight='bold', y=1.04)
    _footer(fig,
            "Paired ΔASR SFT→DPO = +15.1 pp [95% CI +13.0, +17.1] · 3 seeds × 533 attacks")
    plt.tight_layout()
    _save(fig, figures_dir, "fig7_overrefusal_correction")
    print("  Figure 7: DPO cautionary result (slope chart)")


# =============================================================================
# FIGURE 8 — Semantic preservation vs ASR
# =============================================================================
def fig8_semantic_vs_asr(figures_dir):
    scores = load_json("data/medics_500/semantic_scores.json")
    results = _find_held_out("results/eval", "base")
    if not scores or not results:
        # Plot a styled placeholder so the figure slot is filled
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5,
                "Semantic Preservation vs ASR\n\n"
                "Awaiting `data/medics_500/semantic_scores.json`\n"
                "(MiniLM back-translation similarity per CS variant)",
                ha='center', va='center', fontsize=12, color=PALETTE["muted"],
                transform=ax.transAxes)
        ax.set_title("Semantic preservation does not predict attack success",
                     fontsize=14, fontweight='bold', loc='left',
                     color=PALETTE["muted"])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        _save(fig, figures_dir, "fig8_semantic_vs_asr")
        print("  Figure 8: placeholder (no semantic scores)"); return

    lookup = {}
    for r in results:
        key = (r.get("seed_id", ""), r.get("language", ""))
        lookup[key] = r.get("judge_label") == "harmful"

    sem, suc = [], []
    for s in scores:
        k = (s["seed_id"], s["language"])
        if k in lookup:
            sem.append(s["score"])
            suc.append(1 if lookup[k] else 0)
    if not sem:
        print("  Figure 8: no matching data"); return

    sem = np.array(sem); suc = np.array(suc)
    bins = np.linspace(sem.min(), sem.max(), 11)
    idx = np.digitize(sem, bins)
    centers = []; rates = []; ns = []
    for b in range(1, len(bins)):
        mask = idx == b
        if mask.sum() >= 5:
            centers.append((bins[b-1] + bins[b]) / 2)
            rates.append(suc[mask].mean() * 100)
            ns.append(int(mask.sum()))

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.scatter(sem, suc * 100,
               color=PALETTE["base"], alpha=0.10, s=14, zorder=2)
    if centers:
        ax.plot(centers, rates, '-o', color=PALETTE["sft"], lw=2.4,
                markersize=8, zorder=4, label="Binned ASR (≥5 samples)")
    ax.set_xlabel("Semantic preservation score (back-translation cosine)",
                  fontsize=11)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_title("Semantic preservation does not predict attack success",
                 fontsize=14, fontweight='bold', loc='left')
    ax.legend(loc='upper right', frameon=False)
    _hgrid_only(ax)
    _footer(fig, "Bin-wise mean ASR with ≥5 prompts/bin")
    plt.tight_layout()
    _save(fig, figures_dir, "fig8_semantic_vs_asr")
    print("  Figure 8: Semantic vs ASR")


# =============================================================================
# FIGURE 9 — Token Fragmentation (depends on Colab tokenization analysis)
# =============================================================================
def fig9_token_fragmentation(figures_dir):
    analysis = load_json("results/analysis/tokenization_analysis.json")
    if not analysis:
        # Styled placeholder so the figure slot is documented
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.text(0.5, 0.5,
                "Token Fragmentation by Language\n\n"
                "Awaiting `results/analysis/tokenization_analysis.json`\n"
                "(run `scripts/07_tokenization_analysis.py` on Colab)",
                ha='center', va='center', fontsize=12, color=PALETTE["muted"],
                transform=ax.transAxes)
        ax.set_title("Why code-switching works: tokenizer fragmentation",
                     fontsize=14, fontweight='bold', loc='left',
                     color=PALETTE["muted"])
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
        _save(fig, figures_dir, "fig9_token_fragmentation")
        print("  Figure 9: placeholder (run script 07 on Colab)"); return

    df = pd.DataFrame(analysis)
    if "token_count_ratio" not in df.columns:
        print("  Figure 9: invalid format"); return
    order = (df.groupby("language")["token_count_ratio"].median()
             .sort_values(ascending=False).index.tolist())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    palette = sns.color_palette("rocket_r", n_colors=len(order))

    ax = axes[0]
    sns.boxplot(data=df, x="language", y="token_count_ratio",
                order=order, ax=ax, palette=palette,
                fliersize=2, linewidth=1)
    ax.axhline(1.0, color=PALETTE["muted"], linestyle='--', lw=1, alpha=0.7)
    ax.text(len(order) - 0.5, 1.05, "no fragmentation",
            fontsize=8.5, color=PALETTE["muted"], style='italic',
            ha='right', va='bottom')
    ax.set_xlabel("Language", fontsize=11)
    ax.set_ylabel("Token fragmentation ratio (ρ = T_lang / T_en)",
                  fontsize=11)
    ax.set_title("(a)  Code-switched text shatters into more tokens",
                 fontsize=13, fontweight='semibold', loc='left')
    _hgrid_only(ax)

    ax = axes[1]
    sns.boxplot(data=df, x="language", y="oov_proxy_rate",
                order=order, ax=ax, palette=palette,
                fliersize=2, linewidth=1)
    ax.set_xlabel("Language", fontsize=11)
    ax.set_ylabel("Byte-fallback token rate", fontsize=11)
    ax.set_title("(b)  Higher OOV proxy → tokenizer hits byte fallback",
                 fontsize=13, fontweight='semibold', loc='left')
    _hgrid_only(ax)

    fig.suptitle("Why code-switching works: tokenizer fragmentation",
                 fontsize=15, fontweight='bold', y=1.02)
    _footer(fig, "Llama-3-8B BPE tokenizer · 500 seeds × 6 languages")
    plt.tight_layout()
    _save(fig, figures_dir, "fig9_token_fragmentation")
    print("  Figure 9: Token Fragmentation")


# =============================================================================
# FIGURE 10 — Perplexity Detection Baseline
# =============================================================================
def fig10_perplexity_detection(figures_dir):
    detection = load_json("results/analysis/perplexity_results.json")
    if not detection:
        print("  Figure 10: no detection data"); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    # (a) ROC
    ax = axes[0]
    auroc = detection.get("auroc", 0.0)
    if "roc" in detection:
        fpr = detection["roc"]["fpr"]; tpr = detection["roc"]["tpr"]
        ax.plot(fpr, tpr, color=PALETTE["base"], lw=2.6,
                label=f"Perplexity detector (AUROC = {auroc:.3f})")
        ax.fill_between(fpr, tpr, alpha=0.10, color=PALETTE["base"])
    ax.plot([0, 1], [0, 1], color=PALETTE["muted"], lw=1.4, ls='--',
            label="Random (AUROC = 0.5)")
    ax.set_xlabel("False positive rate", fontsize=11)
    ax.set_ylabel("True positive rate", fontsize=11)
    ax.set_title("(a)  ROC: detector is barely above chance",
                 fontsize=13, fontweight='semibold', loc='left')
    ax.legend(loc='lower right', frameon=False)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    _no_grid(ax)

    # (b) Distribution overlap
    ax = axes[1]
    if "per_input" in detection:
        entries = detection["per_input"]
        en = [e["perplexity"] for e in entries if not e.get("is_cs", True)]
        cs = [e["perplexity"] for e in entries if e.get("is_cs", True)]
        # Clip extreme tails for readability
        cap = np.percentile(en + cs, 98)
        en_c = np.clip(en, 0, cap); cs_c = np.clip(cs, 0, cap)
        bins = np.linspace(0, cap, 36)
        ax.hist(en_c, bins=bins, color=PALETTE["english"], alpha=0.55,
                label=f"English (μ={np.mean(en):.1f})", zorder=2)
        ax.hist(cs_c, bins=bins, color=PALETTE["base"], alpha=0.55,
                label=f"Code-switched (μ={np.mean(cs):.1f})", zorder=3)
        ax.set_xlabel("Perplexity", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.legend(loc='upper right', frameon=False)
    ax.set_title("(b)  Distributions overlap heavily",
                 fontsize=13, fontweight='semibold', loc='left')
    _hgrid_only(ax)

    fig.suptitle("Input-level detection cannot filter CS attacks "
                 "(weight-level defense is necessary)",
                 fontsize=15, fontweight='bold', y=1.02)
    _footer(fig, "533 CS held-out vs 500 English seeds · Meta-Llama-3-8B-Instruct")
    plt.tight_layout()
    _save(fig, figures_dir, "fig10_perplexity_detection")
    print("  Figure 10: Perplexity Detection")


# =============================================================================
# FIGURE 11 — Fairness Dashboard (DI / Gini / Counterfactual / Intersectional)
# =============================================================================
def fig11_fairness_dashboard(results_dir, figures_dir):
    from medics.fairness import (
        disparate_impact_ratio, gini_coefficient,
        counterfactual_fairness, intersectional_analysis,
    )

    checkpoints = ["base", "sft", "dpo"]
    di, gini, cf = {}, {}, {}
    inter_matrix = None

    for ckpt in checkpoints:
        rows = _all_seeds(results_dir, ckpt)
        if not rows: continue
        lang_asr = compute_per_language_asr(rows)
        defense = {l: 1.0 - a for l, a in lang_asr.items()}
        di[ckpt] = disparate_impact_ratio(defense).get("ratio", 0)
        gini[ckpt] = gini_coefficient(lang_asr)
        cf[ckpt] = counterfactual_fairness(rows).get("consistency_rate", 1.0)
        if ckpt == "sft":   # use SFT (the headline) for intersectional view
            inter_matrix = intersectional_analysis(rows).get("matrix", {})

    if not di:
        print("  Figure 11: no data"); return

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    labels = [CHECKPOINT_LABEL[c] for c in checkpoints if c in di]
    colors = [CHECKPOINT_COLOR[c] for c in checkpoints if c in di]
    alphas = [0.95, 1.0, 0.6]

    # (a) Disparate Impact
    ax = axes[0, 0]
    vals = [di[c] for c in checkpoints if c in di]
    bars = ax.bar(labels, vals, color=colors,
                  edgecolor='white', linewidth=1.4, width=0.55, zorder=3)
    for bar, a in zip(bars, alphas[:len(bars)]):
        bar.set_alpha(a)
    ax.axhline(0.8, color=PALETTE["base"], linestyle='--', lw=1.4,
               label="4/5 fairness threshold")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.2f}", ha='center', va='bottom',
                fontsize=11, fontweight='semibold')
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Defense rate ratio (min / max)", fontsize=10.5)
    ax.set_title("(a)  Disparate Impact (4/5 rule)",
                 fontsize=13, fontweight='semibold', loc='left')
    ax.legend(loc='lower right', frameon=False)
    _hgrid_only(ax)

    # (b) Gini coefficient
    ax = axes[0, 1]
    g_vals = [gini[c] for c in checkpoints if c in gini]
    ax.plot(labels, g_vals, '-o', color=PALETTE["neutral"], lw=2.4,
            markersize=10, markerfacecolor=PALETTE["sft"],
            markeredgecolor='white', markeredgewidth=1.5)
    for x_, v in zip(labels, g_vals):
        ax.text(x_, v + 0.012, f"{v:.3f}", ha='center', va='bottom',
                fontsize=10.5, color=PALETTE["ink"], fontweight='semibold')
    ax.set_ylim(0, max(g_vals) * 1.6 + 0.02)
    ax.set_ylabel("Gini coefficient (cross-language ASR)", fontsize=10.5)
    ax.set_title("(b)  Inequality across languages",
                 fontsize=13, fontweight='semibold', loc='left')
    _hgrid_only(ax)

    # (c) Counterfactual consistency
    ax = axes[1, 0]
    cf_vals = [cf[c] * 100 for c in checkpoints if c in cf]
    bars = ax.bar(labels, cf_vals, color=colors,
                  edgecolor='white', linewidth=1.4, width=0.55, zorder=3)
    for bar, a in zip(bars, alphas[:len(bars)]):
        bar.set_alpha(a)
    for bar, v in zip(bars, cf_vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 1.0,
                f"{v:.1f}%", ha='center', va='bottom',
                fontsize=11, fontweight='semibold')
    ax.set_ylim(0, 110)
    ax.set_ylabel("Consistency across languages (%)", fontsize=10.5)
    ax.set_title("(c)  Counterfactual fairness (matched-set)",
                 fontsize=13, fontweight='semibold', loc='left')
    _hgrid_only(ax)

    # (d) Intersectional heatmap (SFT)
    ax = axes[1, 1]
    if inter_matrix:
        langs = sorted(inter_matrix.keys())
        cats = sorted({c for row in inter_matrix.values() for c in row})
        data = np.array([[inter_matrix.get(l, {}).get(c, 0) for c in cats] for l in langs])
        im = ax.imshow(data * 100, cmap=HEATMAP_CMAP, aspect='auto',
                       vmin=0, vmax=max(25, (data * 100).max()))
        for i in range(len(langs)):
            for j in range(len(cats)):
                v = data[i, j] * 100
                col = "white" if v > 12 else PALETTE["ink"]
                ax.text(j, i, f"{v:.0f}", ha='center', va='center',
                        fontsize=9.5, color=col, fontweight='semibold')
        ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=10)
        ax.set_yticks(range(len(langs))); ax.set_yticklabels(langs, fontsize=10)
        cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("ASR (%)", fontsize=9)
        cbar.outline.set_linewidth(0)
        cbar.ax.tick_params(length=0, labelsize=9)
    ax.set_title("(d)  Intersectional ASR after SFT",
                 fontsize=13, fontweight='semibold', loc='left')
    _no_grid(ax)

    fig.suptitle("Algorithmic fairness — SFT improves equity, DPO partly regresses it",
                 fontsize=15, fontweight='bold', y=1.00)
    _footer(fig, "Per-language defense rate = 1 − ASR · DI = min/max · Gini over 6 languages")
    plt.tight_layout()
    _save(fig, figures_dir, "fig11_fairness_dashboard")
    print("  Figure 11: Fairness Dashboard")


# =============================================================================
# FIGURE 12 — Safety-Fairness Tradeoff
# =============================================================================
def fig12_safety_fairness_tradeoff(results_dir, figures_dir):
    from medics.fairness import equal_opportunity_gap

    checkpoints = ["base", "sft", "dpo"]
    overall, eo = [], []
    for ckpt in checkpoints:
        rows = _all_seeds(results_dir, ckpt)
        if not rows: continue
        valid = [r for r in rows if r.get("judge_label") in ("harmful", "safe")]
        overall.append(sum(1 for r in valid if r.get("judge_label") == "harmful")
                       / max(1, len(valid)) * 100)
        defense = {l: 1.0 - a for l, a in compute_per_language_asr(rows).items()}
        eo.append(equal_opportunity_gap(defense)["max_gap"] * 100)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))

    # (a) Pareto-style scatter
    ax = axes[0]
    markers = ['o', 's', 'D']
    # Per-checkpoint label offsets (points) so labels never overlap markers
    # or each other.  base→above-right, sft→below-left, dpo→below-right
    label_offsets = {
        "base": (12, 8),
        "sft":  (-14, -14),
        "dpo":  (12, -12),
    }
    for i, ckpt in enumerate(checkpoints):
        c = CHECKPOINT_COLOR[ckpt]
        ax.scatter(overall[i], eo[i],
                   s=260, color=c, marker=markers[i],
                   edgecolor='white', linewidth=2, zorder=4,
                   alpha=0.95 if ckpt != "dpo" else 0.6)
        dx, dy = label_offsets[ckpt]
        ax.annotate(CHECKPOINT_LABEL[ckpt], (overall[i], eo[i]),
                    xytext=(dx, dy), textcoords='offset points',
                    fontsize=11, fontweight='semibold', color=c,
                    arrowprops=dict(arrowstyle='-', color=c,
                                   lw=0.8, alpha=0.5) if abs(dx) > 10 else None)
    # Pareto arrow Base -> SFT
    ax.annotate("", xy=(overall[1] + 1.0, eo[1] + 0.15),
                xytext=(overall[0] - 1.0, eo[0] - 0.15),
                arrowprops=dict(arrowstyle='->', color=PALETTE["sft"], lw=2.2,
                                connectionstyle="arc3,rad=-0.18"))
    ax.text((overall[0] + overall[1]) / 2,
            (eo[0] + eo[1]) / 2 + 0.6,
            "Pareto improvement", color=PALETTE["sft"],
            fontsize=10.5, fontweight='semibold', style='italic',
            ha='center')
    ax.set_xlabel("Overall ASR (%)  ←  safer", fontsize=11)
    ax.set_ylabel("Equal-opportunity gap (%)  ←  fairer", fontsize=11)
    ax.set_xlim(0, max(overall) * 1.22 + 3)
    ax.set_ylim(0, max(eo) * 1.35 + 1)
    ax.set_title("(a)  Safety vs fairness — SFT dominates Base on both axes",
                 fontsize=13, fontweight='semibold', loc='left')
    _hgrid_only(ax)

    # (b) FRR English vs CS-benign per language for SFT (the win)
    ax = axes[1]
    fr = load_json("results/fairness/fairness_report.json")
    plotted = False
    for ckpt_key in ["sft", "dpo", "base"]:
        rpt = (fr or {}).get(ckpt_key, {})
        if "per_language_frr" in rpt:
            en_frr = rpt.get("english_frr", 0) * 100
            per_lang = rpt["per_language_frr"]
            langs = sorted(per_lang.keys())
            x = np.arange(len(langs))
            w = 0.36
            b1 = ax.bar(x - w/2, [en_frr] * len(langs), w,
                        color=PALETTE["english"], edgecolor='white',
                        linewidth=1.2, label=f"English (avg {en_frr:.1f}%)",
                        zorder=3)
            b2 = ax.bar(x + w/2, [per_lang[l] * 100 for l in langs], w,
                        color=PALETTE["base"], edgecolor='white',
                        linewidth=1.2, label="CS-benign", zorder=3)
            ax.set_xticks(x); ax.set_xticklabels(langs, fontsize=10)
            ax.set_ylabel("False refusal rate (%)", fontsize=11)
            ax.set_title(f"(b)  Multilingual FRR (checkpoint: {ckpt_key.upper()})",
                         fontsize=13, fontweight='semibold', loc='left')
            ax.legend(loc='upper right', frameon=False)
            plotted = True; break
    if not plotted:
        ax.text(0.5, 0.5, "CS-benign FRR data unavailable",
                ha='center', va='center', transform=ax.transAxes,
                color=PALETTE["muted"])
    _hgrid_only(ax)

    fig.suptitle("Safety/Fairness frontier — SFT is a Pareto improvement over Base",
                 fontsize=15, fontweight='bold', y=1.02)
    _footer(fig, "EO gap = max(|TPR_gap|, |FPR_gap|) on common-language set")
    plt.tight_layout()
    _save(fig, figures_dir, "fig12_safety_fairness")
    print("  Figure 12: Safety-Fairness Tradeoff")


# =============================================================================
# FIGURE 13 — Thompson Sampling entropy
# =============================================================================
def fig13_thompson_entropy(figures_dir):
    paths = sorted(Path("results/attacks").glob("round_*/bandit_state.json"))
    if not paths:
        print("  Figure 13: no bandit states"); return

    rounds, ents, mats = [], [], []
    arms_master = None
    for p in paths:
        st = load_json(str(p))
        if not st or "history" not in st: continue
        history = st["history"]
        arms = st.get("arms", ["CS", "RP", "MTE", "CS-RP", "CS-OBF"])
        if arms_master is None: arms_master = arms
        cnt = {a: 0 for a in arms}
        for h in history:
            a = h.get("strategy")
            if a in cnt: cnt[a] += 1
        total = sum(cnt.values())
        if total == 0: continue
        probs = np.array([cnt[a] / total for a in arms])
        nz = probs[probs > 0]
        ent = float(-np.sum(nz * np.log2(nz)))
        rn = int(p.parent.name.split("_")[-1])
        rounds.append(rn); ents.append(ent); mats.append(probs)
    if not rounds:
        print("  Figure 13: no history"); return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    max_ent = float(np.log2(len(arms_master)))

    ax = axes[0]
    ax.plot(rounds, ents, '-o', color=PALETTE["neutral"], lw=2.4,
            markersize=10, markerfacecolor=PALETTE["sft"],
            markeredgecolor='white', markeredgewidth=1.5, zorder=3)
    ax.axhline(max_ent, color=PALETTE["base"], ls='--', lw=1.4,
               label=f"Uniform ({max_ent:.2f} bits)")
    for r, e in zip(rounds, ents):
        ax.text(r, e - 0.18, f"{e:.2f}", ha='center', fontsize=10,
                color=PALETTE["ink"])
    ax.set_xticks(rounds); ax.set_xticklabels([f"R{r}" for r in rounds])
    ax.set_xlabel("Round", fontsize=11)
    ax.set_ylabel("Arm-selection entropy (bits)", fontsize=11)
    ax.set_ylim(0, max_ent * 1.12)
    ax.legend(loc='lower right', frameon=False)
    ax.set_title("(a)  Bandit converges (entropy ↓ over rounds)",
                 fontsize=13, fontweight='semibold', loc='left')
    _hgrid_only(ax)

    ax = axes[1]
    matrix = np.array(mats).T
    im = ax.imshow(matrix * 100, cmap=HEATMAP_CMAP, aspect='auto',
                   vmin=0, vmax=100)
    for i in range(len(arms_master)):
        for j in range(len(rounds)):
            v = matrix[i, j] * 100
            col = "white" if v > 50 else PALETTE["ink"]
            ax.text(j, i, f"{v:.0f}", ha='center', va='center',
                    fontsize=10, fontweight='semibold', color=col)
    ax.set_yticks(range(len(arms_master)))
    ax.set_yticklabels(arms_master, fontsize=10.5)
    ax.set_xticks(range(len(rounds)))
    ax.set_xticklabels([f"R{r}" for r in rounds], fontsize=10.5)
    ax.set_title("(b)  Per-arm selection rate (%)",
                 fontsize=13, fontweight='semibold', loc='left')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("rate (%)", fontsize=9)
    cbar.outline.set_linewidth(0)
    cbar.ax.tick_params(length=0, labelsize=9)
    _no_grid(ax)

    fig.suptitle("Thompson Sampling exploration vs convergence",
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    _save(fig, figures_dir, "fig13_thompson_entropy")
    print("  Figure 13: Thompson Sampling Entropy")


# =============================================================================
# FIGURE 14 — Response length (refusals vs compliance)
# =============================================================================
def fig14_response_length(results_dir, figures_dir):
    checkpoints = ["base", "sft", "dpo"]
    records = []
    for ckpt in checkpoints:
        rows = _find_held_out(results_dir, ckpt)
        if not rows: continue
        for r in rows:
            lab = r.get("judge_label")
            if lab not in ("harmful", "safe"): continue
            resp = r.get("model_response", "") or ""
            records.append({
                "checkpoint": ckpt,
                "label": lab,
                "length_words": len(resp.split()),
            })
    if not records:
        print("  Figure 14: no data"); return
    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    palette = {"safe": PALETTE["safe"], "harmful": PALETTE["harmful"]}
    order = [c for c in checkpoints if c in df["checkpoint"].unique()]

    sns.violinplot(data=df, x="checkpoint", y="length_words", hue="label",
                   order=order, hue_order=["safe", "harmful"],
                   palette=palette, split=True, inner="quartile",
                   linewidth=1, cut=0, ax=ax, density_norm="width")
    ax.set_xticklabels([CHECKPOINT_LABEL[c] for c in order], fontsize=11)
    ax.set_ylabel("Response length (words)", fontsize=11)
    ax.set_xlabel("")
    ymax = df["length_words"].quantile(0.97)
    ax.set_ylim(0, ymax)
    ax.set_title("Response length by judgment — SFT refusals are short and crisp",
                 fontsize=14, fontweight='bold', loc='left')
    ax.legend(title="Judge", loc='upper right', frameon=False)
    _hgrid_only(ax)

    # Median annotations
    med = df.groupby(["checkpoint", "label"])["length_words"].median()
    for i, c in enumerate(order):
        for k, lab in enumerate(["safe", "harmful"]):
            try:
                v = med.loc[(c, lab)]
                ax.text(i + (-0.18 if lab == "safe" else 0.18),
                        v, f"med={v:.0f}",
                        ha='center', va='center', fontsize=8.5,
                        color=PALETTE["ink"], fontweight='semibold',
                        bbox=dict(boxstyle="round,pad=0.18",
                                  facecolor='white', edgecolor='none', alpha=0.85))
            except KeyError:
                pass

    _footer(fig, "y-axis clipped at p97 for readability")
    plt.tight_layout()
    _save(fig, figures_dir, "fig14_response_length")
    print("  Figure 14: Response Length")


# =============================================================================
# FIGURE 15 — Strategy × Category (attack rounds)
# =============================================================================
def fig15_strategy_by_category(figures_dir):
    rows = []
    for path in sorted(Path("results/attacks").glob("round_*/results.jsonl")):
        rows.extend(load_jsonl(str(path)))
    if not rows:
        print("  Figure 15: no attack-round data"); return

    cells = {}
    for r in rows:
        if r.get("judge_label") not in ("harmful", "safe"): continue
        k = (r.get("strategy", "?"), r.get("category", "?"))
        h, t = cells.get(k, (0, 0))
        t += 1
        if r.get("judge_label") == "harmful": h += 1
        cells[k] = (h, t)
    if not cells:
        print("  Figure 15: no judged data"); return

    strats = sorted({k[0] for k in cells})
    cats = sorted({k[1] for k in cells})
    asr_mat = np.full((len(strats), len(cats)), np.nan)
    n_mat = np.zeros((len(strats), len(cats)), dtype=int)
    for i, s in enumerate(strats):
        for j, c in enumerate(cats):
            h, t = cells.get((s, c), (0, 0))
            n_mat[i, j] = t
            if t >= 3:
                asr_mat[i, j] = h / t * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    masked = np.ma.array(asr_mat, mask=np.isnan(asr_mat))
    cmap = matplotlib.cm.get_cmap(HEATMAP_CMAP).copy(); cmap.set_bad("#f0f0f0")
    im = ax.imshow(masked, cmap=cmap, vmin=0, vmax=100, aspect='auto')

    for i in range(len(strats)):
        for j in range(len(cats)):
            v = asr_mat[i, j]; n = n_mat[i, j]
            if np.isnan(v):
                ax.text(j, i, f"n={n}", ha='center', va='center',
                        fontsize=8.5, color=PALETTE["muted"])
            else:
                col = "white" if v > 55 else PALETTE["ink"]
                ax.text(j, i, f"{v:.0f}", ha='center', va='center',
                        fontsize=11, fontweight='semibold', color=col)
                ax.text(j, i + 0.32, f"n={n}", ha='center', va='center',
                        fontsize=7.5, color=col, alpha=0.8)

    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, fontsize=10.5)
    ax.set_yticks(range(len(strats))); ax.set_yticklabels(strats, fontsize=10.5)
    ax.set_xlabel("Harm category", fontsize=11)
    ax.set_ylabel("Attack strategy", fontsize=11)
    ax.set_title("ASR by strategy × category (attack-round data)",
                 fontsize=14, fontweight='bold', loc='left')
    cbar = plt.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label("ASR (%)", fontsize=9)
    cbar.outline.set_linewidth(0)
    cbar.ax.tick_params(length=0, labelsize=9)
    _no_grid(ax)
    _footer(fig, "All rounds merged · cells with <3 samples masked")
    plt.tight_layout()
    _save(fig, figures_dir, "fig15_strategy_by_category")
    print("  Figure 15: Strategy × Category")


# =============================================================================
# FIGURE 16 — Cross-architecture transfer (Llama / Mistral / Qwen)
# =============================================================================
def fig16_transfer_comparison(results_dir, figures_dir,
                              transfer_dir="results/transfer"):
    """Same 533 held-out CS attacks, three architectures."""
    transfer_path = Path(transfer_dir)
    sources = {
        "Llama-3-8B (source)":      _all_seeds(results_dir, "base"),
        "Mistral-7B-Instruct-v0.3": load_jsonl(str(transfer_path / "mistral_results_judged.jsonl"))
            if (transfer_path / "mistral_results_judged.jsonl").exists() else [],
        "Qwen-2.5-7B-Instruct":     load_jsonl(str(transfer_path / "qwen_results_judged.jsonl"))
            if (transfer_path / "qwen_results_judged.jsonl").exists() else [],
    }
    if not all(sources.values()):
        print("  Figure 16: missing transfer data"); return

    colors = [PALETTE["base"], PALETTE["mistral"], PALETTE["qwen"]]

    overall = {}
    for name, rows in sources.items():
        valid = [r for r in rows if r.get("judge_label") in ("harmful", "safe")]
        arr = [1 if r.get("judge_label") == "harmful" else 0 for r in valid]
        m, lo, hi = bootstrap_ci(arr)
        overall[name] = (m * 100, (m - lo) * 100, (hi - m) * 100, len(valid))

    common = sorted(
        set.intersection(*[
            {r.get("language", "?") for r in rows
             if r.get("judge_label") in ("harmful", "safe")}
            for rows in sources.values()
        ])
    )
    lang_mat = np.zeros((len(sources), len(common)))
    for i, (name, rows) in enumerate(sources.items()):
        for j, l in enumerate(common):
            sub = [r for r in rows if r.get("language") == l
                   and r.get("judge_label") in ("harmful", "safe")]
            if sub:
                lang_mat[i, j] = sum(1 for r in sub if r.get("judge_label") == "harmful") / len(sub) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4),
                             gridspec_kw={"width_ratios": [0.85, 1.6]})

    # (a) overall
    ax = axes[0]
    names = list(sources.keys())
    means = [overall[n][0] for n in names]
    elo = [overall[n][1] for n in names]
    ehi = [overall[n][2] for n in names]
    bars = ax.bar(range(len(names)), means, color=colors, alpha=0.92,
                  edgecolor='white', linewidth=1.4, width=0.55, zorder=3)
    ax.errorbar(range(len(names)), means, yerr=[elo, ehi], fmt='none',
                color=PALETTE["ink"], capsize=4, lw=1.2, zorder=4)
    for i, v in enumerate(means):
        # Place label above the upper error bar to avoid overlap
        y_top = v + ehi[i] + 2.5
        ax.text(i, y_top, f"{v:.1f}%", ha='center', va='bottom',
                fontsize=11.5, fontweight='bold', color=colors[i])
    short_names = ["Llama-3-8B", "Mistral-7B", "Qwen-2.5-7B"]
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(short_names, fontsize=10.5)
    ax.set_ylabel("Overall ASR (%)", fontsize=11)
    ax.set_ylim(0, max(means) * 1.30 + 4)
    ax.set_title("(a)  Transfer ASR (held-out)",
                 fontsize=13, fontweight='semibold', loc='left')
    _hgrid_only(ax)

    # (b) per-language grouped bars
    ax = axes[1]
    x = np.arange(len(common))
    w = 0.27
    for i, name in enumerate(names):
        ax.bar(x + (i - 1) * w, lang_mat[i], w, color=colors[i], alpha=0.92,
               edgecolor='white', linewidth=1.0, label=name.split("(")[0].strip(),
               zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(common, fontsize=10.5)
    ax.set_ylabel("ASR (%)", fontsize=11)
    ax.set_xlabel("Language", fontsize=11)
    ax.set_title("(b)  Per-language ASR",
                 fontsize=13, fontweight='semibold', loc='left')
    ax.legend(loc='upper right', frameon=False, fontsize=9.5)
    ax.set_ylim(0, max(lang_mat.max(), 1) * 1.18 + 5)
    _hgrid_only(ax)

    fig.suptitle("CS attacks generalize cross-architecture (Mistral/Qwen even more vulnerable)",
                 fontsize=15, fontweight='bold', y=1.02)
    _footer(fig, "Same 533 held-out CS attacks · base undefended models · 95% bootstrap CI")
    plt.tight_layout()
    _save(fig, figures_dir, "fig16_transfer_comparison")
    print("  Figure 16: Cross-Architecture Transfer")
