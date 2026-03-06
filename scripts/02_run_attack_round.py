#!/usr/bin/env python3
"""
Run one attack round. Selects strategies via Thompson Sampling,
applies attacks, calls GPT-4o judge. Does NOT run inference
(that requires GPU — see colab/run_inference.py).

Workflow:
  1. This script generates attack prompts → saves to attacks/round_N/prompts.jsonl
  2. GPU script runs inference on prompts → saves to attacks/round_N/responses.jsonl
  3. This script judges responses → saves to attacks/round_N/results.jsonl

Usage:
  python scripts/02_run_attack_round.py --round 1 --phase generate
  # ... run colab/run_inference.py on Colab ...
  python scripts/02_run_attack_round.py --round 1 --phase judge
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.bandit import ThompsonBandit
from medics.attacks import apply_strategy, get_available_strategies
from medics.judge import judge_response_batch
from medics.metrics import compute_asr
from medics.utils import load_jsonl, save_jsonl, load_json, load_config


def generate_attacks(config, round_num, attack_pool, keywords):
    """Phase 1: Generate attack prompts using Thompson Sampling."""
    available = get_available_strategies(round_num)
    categories = config.get("categories", ["TOX", "SH", "MIS", "ULP", "PPV", "UCA"])
    bandit = ThompsonBandit(arms=available, categories=categories)

    # Load previous round's bandit state if available
    prev_state = Path(f"results/attacks/round_{round_num - 1}/bandit_state.json")
    if prev_state.exists():
        print(f"Loading bandit state from round {round_num - 1}")
        bandit = ThompsonBandit.load(prev_state)
        # Filter to only available strategies for this round
        if set(bandit.arms) != set(available):
            print(f"Reinitializing bandit for new strategies: {available}")
            bandit = ThompsonBandit(arms=available, categories=categories)

    languages = [lang["code"] for lang in config["dataset"]["languages"]]
    attacks = []
    n_attacks = config["red_team"].get("attacks_per_round", 150)

    for i in range(n_attacks):
        seed = random.choice(attack_pool)
        min_pulls = config["red_team"].get("min_exploration_pulls", 10)
        strategy = bandit.select_with_exploration(
            category=seed.get("category"), min_pulls=min_pulls
        )
        language = random.choice(languages)

        attack = apply_strategy(seed, strategy, keywords, language)
        attack["round"] = round_num
        attacks.append(attack)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{n_attacks}] generated")

    return attacks, bandit


def judge_responses(responses):
    """Phase 2: Judge model responses via GPT-4o."""
    return judge_response_batch(responses)


def main():
    parser = argparse.ArgumentParser(description="MediCS attack round runner")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--phase", choices=["generate", "judge"], required=True)
    parser.add_argument("--config", default="config/experiment_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    round_dir = Path(f"results/attacks/round_{args.round}")
    round_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "generate":
        attack_pool = load_jsonl("data/splits/attack_pool.jsonl")
        keywords = load_json("data/seeds/keywords_checkpoint.json")
        if keywords is None:
            keywords = {}

        attacks, bandit = generate_attacks(config, args.round, attack_pool, keywords)
        save_jsonl(attacks, round_dir / "prompts.jsonl")
        bandit.save(round_dir / "bandit_state.json")

        print(f"\nGenerated {len(attacks)} attack prompts → {round_dir}/prompts.jsonl")
        print(f"Strategy distribution: {bandit.get_pull_counts()}")
        print("Next: run colab/run_inference.py on Colab, then come back with --phase judge")

    elif args.phase == "judge":
        responses = load_jsonl(round_dir / "responses.jsonl")
        if not responses:
            print(f"ERROR: No responses found at {round_dir}/responses.jsonl")
            print("Run colab/run_inference.py first!")
            return

        results = judge_responses(responses)
        save_jsonl(results, round_dir / "results.jsonl")

        asr = compute_asr(results)
        n_harmful = sum(1 for r in results if r.get("judge_label") == "harmful")
        print(f"\nRound {args.round} ASR: {asr:.1%} ({n_harmful}/{len(results)})")

        # Update bandit with results
        bandit_path = round_dir / "bandit_state.json"
        if bandit_path.exists():
            bandit = ThompsonBandit.load(bandit_path)
            for r in results:
                reward = 1.0 if r.get("judge_label") == "harmful" else 0.0
                bandit.update(r.get("strategy", "CS"), reward,
                              category=r.get("category"))
            bandit.save(bandit_path)
            print(f"Updated bandit state: {bandit.get_estimated_rates()}")


if __name__ == "__main__":
    main()
