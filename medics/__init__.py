"""
MediCS — Agentic Adversarial Training for Medical AI Safety
============================================================

Core Python package. All logic lives here as importable, testable modules.

Modules:
    utils          — File I/O, config, translation, caching, code-switching
    bandit         — Thompson Sampling (Beta-Bernoulli multi-armed bandit)
    judge          — GPT-5 judge, helpfulness eval, API cost tracking
    attacks        — 5 attack strategy functions (CS, RP, MTE, CS-RP, CS-OBF)
    defense        — SFT/DPO data construction
    metrics        — ASR, RG, HR, FRR, bootstrap CI, McNemar, Cohen's h, residual analysis
    figures        — 10 publication-quality figure generators
    tokenization   — Token fragmentation analysis (WHY code-switching works)
    detection      — Perplexity-based detection baseline
    timing         — Computational cost tracking (wall-clock, GPU hours)
    ethics         — Ethical framework template (dual-use, disclosure, impact)

Usage:
    from medics.utils import load_jsonl, save_jsonl, load_config
    from medics.bandit import ThompsonBandit
    from medics.judge import (call_judge, judge_response_batch,
                              judge_helpfulness_batch, print_session_cost,
                              reset_session_usage, track_external_usage)
    from medics.attacks import apply_strategy, get_available_strategies
    from medics.defense import build_sft_data, build_dpo_pairs
    from medics.metrics import (compute_asr, compute_all_metrics, bootstrap_ci,
                                compute_cohens_h, compute_effect_sizes,
                                compute_residual_failure_breakdown)
    from medics.tokenization import analyze_tokenization, compute_fragmentation_summary
    from medics.detection import perplexity_detector, detection_by_language
    from medics.timing import timed_phase, save_timing_report, compute_gpu_hours
    from medics.ethics import generate_ethics_statement, format_ethics_section
"""

__version__ = "2.4.0"
