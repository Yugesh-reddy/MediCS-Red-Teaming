<div align="center">

# MediCS — Medical Code-Switching

### A Closed-Loop Agentic Red-Teaming & Defense Framework for Medical LLM Safety

*Cross-lingual adversarial attacks · Thompson-Sampling red team · QLoRA-SFT/DPO defense · Algorithmic fairness audit*

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers%20%7C%20PEFT%20%7C%20TRL-yellow)](https://huggingface.co/)
[![Llama 3](https://img.shields.io/badge/Model-Llama--3--8B-FF6F00)](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
[![Tests](https://img.shields.io/badge/tests-213%20passing-brightgreen)]()
[![License](https://img.shields.io/badge/License-Research-purple.svg)]()

</div>

---

## TL;DR

> **MediCS** is the first end-to-end framework that combines **medical-domain code-switching attacks**, **adaptive agentic red-teaming via Thompson Sampling**, and **iterative LoRA-based defense** in a fully closed loop — and rigorously evaluates the resulting model with **algorithmic fairness, cross-architecture transfer, and statistical significance testing** at the standard of an ACM conference paper.

**Headline result:** A single round of QLoRA-SFT reduces Attack Success Rate on a hardened Llama-3-8B-Instruct from **27.6% → 6.4%** (−21.2 pp, *p* < 0.0001, paired bootstrap 95% CI **[−23.5, −18.9]**) while preserving helpfulness at **99.6%** (over 1,599 held-out adversarial prompts and 1,500 benign twins, evaluated across 3 random seeds and judged by GPT-5).

We also publish a **negative result**: stacking DPO on top of SFT *regresses* safety to 21.5% ASR — a cautionary finding when DPO's premise (correcting over-refusal) is not satisfied.

---

## Why this project matters

Medical AI is a high-stakes deployment surface. The same model that helps a patient understand a prescription can, under the right adversarial pressure, leak procedural detail for self-harm, toxic dosing, or weaponized misinformation. Existing safety benchmarks are almost entirely **English-centric** and **single-turn** — yet the real internet, and real attackers, are multilingual and adaptive.

MediCS asks a simple but uncomfortable question:

> *What happens to a state-of-the-art medical assistant when an attacker mixes English with Hindi, Bengali, Tagalog, Yoruba, Swahili, or Gujarati — and adapts strategy on the fly?*

The answer, on Llama-3-8B-Instruct: a **27.6% ASR baseline** that further **transfers cross-architecture** to Mistral-7B (**61.7%**) and Qwen-2.5-7B (**51.6%**). The same framework then closes the loop and *defends* the model down to **6.4%** with no measurable cost to helpfulness.

---

## What's novel

| Contribution | First of its kind in this combination |
|---|---|
| **Medical-domain code-switching attack suite** | 5 strategies × 6 low-resource languages × 6 harm categories, sentence-level (not keyword-level) translation |
| **Adaptive agentic red team** | Thompson-Sampling Beta-Bernoulli bandit per harm category — explores then exploits the most effective strategy |
| **Closed-loop defense** | GPT-5 judge → SFT/DPO data construction → QLoRA fine-tune → re-attack — fully automated |
| **Prefix-recovery training** | Novel SFT examples that teach the model to *pivot and refuse* even when generation is primed with compliant text |
| **MediCS-500 dataset** | 500 expert-curated harmful seeds + 500 benign twins, code-switched to 6 languages with back-translation verification |
| **Publication-grade evaluation** | Paired bootstrap 95% CIs, McNemar with Holm-Bonferroni FWER, Cohen's *h*, residual-failure breakdown, transfer to 2 architectures, full algorithmic fairness audit |

---

## Headline numbers

### Defense effectiveness (Llama-3-8B-Instruct, 3 seeds × 533 attacks × 500 benign twins)

| Checkpoint | ASR ↓ | Helpfulness Retention ↑ | False Refusal Rate ↓ | McNemar vs Base |
|---|---|---|---|---|
| **Base**            | 27.6% ± 0.9% | 98.7% | 1.3% | — |
| **+ QLoRA-SFT**     | **6.4% ± 0.6%** | **99.6%** | **0.4%** | *p* < 0.0001 *** |
| + DPO (regression) | 21.5% ± 1.1% | 98.9% | 1.1% | *p* = 0.0001 *** |

Paired ΔASR (95% CI): base→SFT = **−21.2 [−23.5, −18.9]**, SFT→DPO = **+15.1 [+13.0, +17.1]**. Cohen's *h*: base↔SFT = 0.595 (medium).

### Cross-architecture transfer (held-out CS attacks, undefended models)

| Model | Overall ASR | 95% CI |
|---|---|---|
| Llama-3-8B-Instruct (source) | 27.6% | — |
| **Mistral-7B-Instruct-v0.3** | **61.7%** | [57.6, 65.9] |
| **Qwen-2.5-7B-Instruct**     | **51.6%** | [47.5, 55.9] |

Attacks crafted against Llama-3 generalize *and amplify* across architectures.

---

## System architecture

```
┌────────────────────────────────────────────────────────────────────┐
│  MediCS-500 Dataset    500 seeds × 6 categories × 6 languages      │
│                       + 500 benign twins (back-translation verified)│
└──────────────────────────────┬─────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
       ┌──────────────┐                  ┌──────────────┐
       │   ATTACK     │                  │   DEFENSE    │
       │              │                  │              │
       │ 5 strategies │                  │ Phase 1: SFT │
       │  ├─ CS       │                  │  (QLoRA r=16)│
       │  ├─ RP       │   Thompson       │              │
       │  ├─ MTE      │  Sampling        │ Phase 2: DPO │
       │  ├─ CS-RP    │  (Beta-Bern)     │  (QLoRA r=8) │
       │  └─ CS-OBF   │                  │              │
       └──────┬───────┘                  └──────┬───────┘
              │                                 │
              ▼                                 ▼
       ┌──────────────────────────────────────────────┐
       │   GPT-5 JUDGE  (gpt-5-chat, T=0.0)           │
       │   harmful / refusal / off-topic / partial    │
       └──────┬───────────────────────────────────────┘
              │
              ▼
       ┌──────────────────────────────────────────────┐
       │   EVALUATION                                 │
       │   • 3 checkpoints × 3 seeds × 1599 attacks   │
       │   • Bootstrap 95% CI, McNemar, Cohen's h     │
       │   • Holm-Bonferroni across 6 languages       │
       │   • Residual failure analysis                │
       │   • Transfer eval (Mistral-7B, Qwen-2.5-7B)  │
       │   • Algorithmic fairness (DI, EO, Gini, CF)  │
       └──────────────────────────────────────────────┘

       ↻  CLOSED LOOP: attack → infer → judge → defend → re-attack
```

---

## The 5 attack strategies

| Code | Name | What it does |
|---|---|---|
| **CS** | Code-Switching | Sentence-level translation — entire sentences containing harmful keywords are translated into one of 6 low-resource languages, fragmenting tokenization and bypassing English-centric safety classifiers |
| **RP** | Roleplay | 8 medical-professional personas (ER attending, toxicology fellow, OSCE station, forensic pathologist…) with directive suffixes and **response priming** (compliant assistant prefix injected after generation prompt) |
| **MTE** | Multi-Turn Escalation | Topic-aware 3–5 turn scaffolds per harm category (TOX, SH, MIS, ULP, PPV, UCA) that build genuine clinical context before pivoting to the harmful ask |
| **CS-RP** | CS + Roleplay | Sentence-level CS layered with medical persona + response priming. Strongest hybrid. |
| **CS-OBF** | CS + Obfuscation | Sentence-level CS + light leetspeak (deliberately readable — earlier dot-join/base64 destroyed comprehension) |

A **Beta-Bernoulli Thompson-Sampling bandit per harm category** chooses which strategy to fire. Min 10 exploration pulls per arm; full RNG-state persistence per round for reproducibility.

---

## Repository layout

```
MediCS/
├── medics/                  # Importable Python package (v2.7.2)
│   ├── attacks.py           #  → 5 attack strategies, prefix injection
│   ├── bandit.py            #  → Thompson Sampling (seeded)
│   ├── judge.py             #  → GPT-5 judge, retries, cost tracking
│   ├── defense.py           #  → SFT/DPO data, prefix-recovery examples
│   ├── metrics.py           #  → ASR / RG / HR / FRR + paired bootstrap, McNemar, Cohen's h, Holm-Bonferroni
│   ├── fairness.py          #  → DI, EO, Gini, Theil, counterfactual, intersectional (with bootstrap CIs)
│   ├── figures.py           #  → 16 publication-quality figure generators
│   ├── tokenization.py      #  → Token-fragmentation analysis (why CS works)
│   ├── detection.py         #  → Perplexity-based detection baseline
│   ├── timing.py            #  → Wall-clock & GPU-hour tracking
│   └── ethics.py            #  → Dual-use disclosure framework
├── scripts/                 # CPU-side pipeline (numbered, run in order)
│   ├── 01_build_dataset.py
│   ├── 02_run_attack_round.py
│   ├── 03_build_defense_data.py
│   ├── 04_evaluate.py
│   ├── 05_generate_figures.py
│   ├── 07_tokenization_analysis.py
│   ├── 08_detection_analysis.py
│   ├── 09_cost_report.py
│   ├── 10_fairness_analysis.py
│   └── 11_extract_qualitative_examples.py
├── colab/                   # GPU-side scripts (Llama-3 inference + training)
│   ├── train_sft.py         #  → QLoRA-SFT, multi-round chaining
│   ├── train_dpo.py         #  → DPO preference optimization
│   ├── run_inference.py     #  → Batch inference, prefix injection, prompt-mode logging
│   ├── run_transfer.py      #  → Mistral / Qwen cross-architecture eval
│   └── run_perplexity.py    #  → Detection baseline
├── notebooks/colab_runner.ipynb   # Single thin launcher for Colab
├── tests/                   # 13 test files, 213 tests, all passing
├── config/experiment_config.yaml  # Single source of truth for hyperparameters
├── data/                    # MediCS-500 + splits + cached refusals/helpful
├── results/                 # attacks/, eval/, transfer/, analysis/, figures
├── docs/                    # attack_strategy_overhaul.md, project explanation
└── reproduce.sh             # One-shot replay of the entire pipeline
```

---

## Engineering principles I applied

This is a research project, but it was engineered with production discipline because reproducibility *is* the product:

- **Library-first architecture** — every piece of logic lives in `medics/` as importable, testable functions. Scripts only orchestrate.
- **Single source of truth for hyperparameters** — everything in `config/experiment_config.yaml`; nothing hard-coded.
- **End-to-end seeding** — `random.Random(seed + round_num)`, `np.random.RandomState`, `torch.manual_seed`, full bandit RNG state persisted per round, seeded vectorized bootstrap, 3 evaluation seeds (42, 123, 456).
- **213 tests across 13 files**, all passing. Includes counterfactual matched-set validation, common-group equalized-odds alignment, and CI-presence checks on every fairness metric.
- **Cost discipline** — $10 OpenAI budget, tracked per-call in `cost_tracking.md`. DPO build is **$0 API** (`generate_missing_helpful=False`).
- **Compute discipline** — `timing.py` tracks every phase; final cost report: **66.31 h wall-clock, 51.89 h GPU**, all on a Colab L4.
- **Idempotent pipelines** — `--rebuild-from-cache` rebuilds SFT data with $0 API; `--judge-attacks` skips already-labeled rows.
- **Statistical honesty** — paired bootstrap CIs (tighter than independent), McNemar's exact test, FWER correction across 6 languages × 3 checkpoints, Cohen's *h* for effect size, and a published **negative result** when the DPO phase regressed.

---

## What an evaluator can verify in 5 minutes

1. **Numbers reproduce:** `bash reproduce.sh eval` reruns evaluation from cached responses.
2. **Tests pass:** `pytest -q` → 213 passing.
3. **Figures regenerate:** `python scripts/05_generate_figures.py --results-dir results/eval/` → all 16 publication figures.
4. **Statistical claims are auditable:** `results/eval/summary.json` contains every CI, *p*-value, and effect size cited in the paper.
5. **Attacks work:** open `results/attacks/round_1/results.jsonl` — first-round ASR 56.5% on undefended Llama-3.

---

## Selected figures

The framework produces **16 publication-quality figures**, including:

- **fig01** — ASR across defense stages (base → SFT → DPO) with bootstrap CIs
- **fig02** — Strategy effectiveness heatmap (5 strategies × 6 categories)
- **fig03** — Cross-language vulnerability with Holm-Bonferroni significance
- **fig04** — Thompson-Sampling convergence (per-arm Beta posteriors over rounds)
- **fig07** — DPO over-refusal **regression** (the negative result, visualized)
- **fig11** — Fairness dashboard (DI ratio, Gini, counterfactual, intersectional heatmap)
- **fig12** — Safety–fairness tradeoff scatter
- **fig16** — Cross-architecture transfer (Llama-3 vs Mistral-7B vs Qwen-2.5-7B)

All figures use colorblind-safe palettes and are designed for ACM-sigconf two-column rendering.

---

## Tech stack

**Modeling:** PyTorch · 🤗 Transformers · PEFT · TRL (DPOTrainer) · bitsandbytes (4-bit NF4 QLoRA) · Llama-3-8B-Instruct · Mistral-7B-Instruct-v0.3 · Qwen-2.5-7B-Instruct

**Judging & translation:** OpenAI Azure (GPT-5-chat, GPT-5-mini) · Google Translate / NLLB fallback · back-translation verification

**Statistics & analysis:** NumPy · SciPy (`mcnemar`, paired bootstrap) · scikit-learn (ROC/AUROC for detection) · custom Holm-Bonferroni & Cohen's *h*

**Visualization:** Matplotlib · Seaborn · colorblind-safe palettes (Wong, ColorBrewer)

**Infra:** Google Colab Pro (L4 GPU) · gated HuggingFace Hub (artifact upload via `06_upload_hf.py`) · pytest · YAML config

---

## Quickstart

```bash
git clone <this-repo> && cd MediCS
pip install -r requirements.txt
pytest -q                                    # 213 tests should pass

# Build the MediCS-500 dataset (one-time, no GPU)
python scripts/01_build_dataset.py --config config/experiment_config.yaml

# A full attack-defense round (Colab cells handle GPU steps)
python scripts/02_run_attack_round.py --round 1 --phase generate
# → run colab/run_inference.py on Colab
python scripts/02_run_attack_round.py --round 1 --phase judge
python scripts/03_build_defense_data.py --rounds 1 --type sft
# → run colab/train_sft.py on Colab

# Evaluate across 3 checkpoints × 3 seeds with full statistics
python scripts/04_evaluate.py --checkpoints base,sft,dpo --seeds 42,123,456

# Algorithmic fairness audit
python scripts/10_fairness_analysis.py --mode full

# 16 figures
python scripts/05_generate_figures.py --results-dir results/eval/

# One-shot replay of the entire pipeline
bash reproduce.sh all
```

---

## Responsible disclosure & ethics

This project studies adversarial attacks on medical AI **to defend against them**, not to enable harm. Specifically:

- The MediCS-500 seeds were curated against the WHO harm taxonomy and never include working operational detail for real-world misuse.
- All judge outputs and harmful generations are stored locally with restricted access — none are released publicly.
- The framework is designed to ship a **defended** checkpoint and a **detection baseline**, not the offensive prompts themselves.
- A dual-use disclosure framework lives in `medics/ethics.py` and is reproduced verbatim in the paper appendix.
- The reported attack research aligns with the practice of published red-teaming work (e.g., RoleBreaker, Multilingual Jailbreak Challenges, CLINIC) and follows the responsible-disclosure spirit of the OpenAI / Anthropic safety evaluation guidelines.

---

## What I learned building this

- **Statistical rigor is a design choice, not a postscript.** Pairing every claim with a CI and a McNemar *p*-value forced me to redesign the evaluation harness three times — and caught a bug where held-out attacks were never being judged (silently giving 0% ASR).
- **Negative results are first-class.** The DPO regression is the most interesting finding in the paper; reporting it honestly is more valuable than burying it.
- **Reproducibility is plumbing.** Seeded RNG-state persistence, idempotent pipelines, and a single config file are unglamorous — they are also the reason this work can be verified.
- **Cross-lingual safety is unsolved.** A 27.6% baseline ASR on a frontier-class instruction-tuned model, against attacks costing pennies to generate, is a result the field has not yet internalized.

---

## Author

**Yugesh Reddy Sappidi** — graduate research, CS 516 (Adversarial AI Safety)

Co-author: Yash Yeole

If you'd like to talk about LLM safety, adversarial robustness, or RLHF/DPO behavior in low-resource regimes, I'd love to hear from you.

📧 yugeshreddysappidi@gmail.com

---

<div align="center">

*MediCS — because safety that only works in English isn't safety.*

</div>
