#!/usr/bin/env python3
"""
Build training data for SFT and DPO from attack round results.
Runs locally — no GPU needed.

Usage:
  python scripts/03_build_defense_data.py --rounds 1 --type sft
  python scripts/03_build_defense_data.py --rounds 1,2,3 --type dpo
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.defense import build_sft_data, build_dpo_pairs
from medics.utils import load_jsonl, save_json, load_json


def main():
    parser = argparse.ArgumentParser(description="Build defense training data")
    parser.add_argument("--rounds", type=str, required=True,
                        help="Comma-separated round numbers: 1 or 1,2,3")
    parser.add_argument("--type", choices=["sft", "dpo"], required=True)
    parser.add_argument("--config", default="config/experiment_config.yaml")
    args = parser.parse_args()

    rounds = [int(r) for r in args.rounds.split(",")]

    # Ensure output dir exists
    Path("data/defense").mkdir(parents=True, exist_ok=True)

    if args.type == "sft":
        # Collect jailbreaks from specified rounds
        all_jailbreaks = []
        for r in rounds:
            results_path = f"results/attacks/round_{r}/results.jsonl"
            results = load_jsonl(results_path)
            jailbreaks = [res for res in results if res.get("judge_label") == "harmful"]
            all_jailbreaks.extend(jailbreaks)
            print(f"Round {r}: {len(jailbreaks)} jailbreaks")

        benign_twins = load_jsonl("data/seeds/benign_twins.jsonl")
        print(f"Benign twins: {len(benign_twins)}")

        sft_data = build_sft_data(all_jailbreaks, benign_twins)
        output_path = f"data/defense/sft_round_{'_'.join(map(str, rounds))}.json"
        save_json(sft_data, output_path)
        print(f"SFT data saved: {output_path} ({len(sft_data)} examples)")

    elif args.type == "dpo":
        # DPO uses accumulated data from all rounds — $0 additional API
        all_jailbreaks = []
        all_benign_eval = []

        for r in rounds:
            results = load_jsonl(f"results/attacks/round_{r}/results.jsonl")
            all_jailbreaks.extend(
                [res for res in results if res.get("judge_label") == "harmful"]
            )

            benign_path = f"results/eval/sft/round_{r}/benign_results.jsonl"
            if Path(benign_path).exists():
                all_benign_eval.extend(load_jsonl(benign_path))

        refusals = load_json("data/defense/refusals.json")
        if refusals is None:
            refusals = {}
            print("WARNING: No refusals.json found. DPO pairs will only include "
                  "over-refusal corrections.")

        dpo_pairs = build_dpo_pairs(all_jailbreaks, refusals, all_benign_eval)
        save_json(dpo_pairs, "data/defense/dpo_pairs.json")
        print(f"DPO pairs saved: data/defense/dpo_pairs.json ({len(dpo_pairs)} pairs)")


if __name__ == "__main__":
    main()
