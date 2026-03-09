"""
MediCS — Agentic Adversarial Training for Medical AI Safety
============================================================

Core Python package. All logic lives here as importable, testable modules.

Modules:
    utils       — File I/O, config, translation, caching, code-switching
    bandit      — Thompson Sampling (Beta-Bernoulli multi-armed bandit)
    judge       — GPT-4o judge, helpfulness eval, API cost tracking
    attacks     — 5 attack strategy functions (CS, RP, MTE, CS-RP, CS-OBF)
    defense     — SFT/DPO data construction
    metrics     — ASR, RG, HR, FRR, bootstrap CI, McNemar
    figures     — 8 publication-quality figure generators

Usage:
    from medics.utils import load_jsonl, save_jsonl, load_config
    from medics.bandit import ThompsonBandit
    from medics.judge import (call_judge, judge_response_batch,
                              judge_helpfulness_batch, print_session_cost,
                              reset_session_usage, track_external_usage)
    from medics.attacks import apply_strategy, get_available_strategies
    from medics.defense import build_sft_data, build_dpo_pairs
    from medics.metrics import compute_asr, compute_all_metrics, bootstrap_ci
"""

__version__ = "2.1.0"
