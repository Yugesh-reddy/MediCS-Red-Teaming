"""
MediCS — Agentic Adversarial Training for Medical AI Safety
============================================================

Core Python package. All logic lives here as importable, testable modules.

Usage:
    from medics.utils import load_jsonl, save_jsonl, load_config
    from medics.bandit import ThompsonBandit
    from medics.judge import call_judge, judge_response_batch
    from medics.attacks import apply_strategy
    from medics.defense import build_sft_data, build_dpo_pairs
    from medics.metrics import compute_asr, compute_all_metrics
"""

__version__ = "2.0.0"
