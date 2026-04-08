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

from medics.defense import (  # pyre-ignore[21]
    build_sft_data, build_dpo_pairs, rebuild_sft_from_cache,
)
from medics.utils import load_jsonl, save_json, load_json, load_config  # pyre-ignore[21]


def main():
    parser = argparse.ArgumentParser(description="Build defense training data")
    parser.add_argument("--rounds", type=str, required=True,
                        help="Comma-separated round numbers: 1 or 1,2,3")
    parser.add_argument("--type", choices=["sft", "dpo"], required=True)
    parser.add_argument("--force", action="store_true",
                        help="Rebuild even if output file already exists")
    parser.add_argument("--rebuild-from-cache", action="store_true",
                        help="Rebuild SFT data from cached refusals/helpful ($0 API)")
    parser.add_argument("--config", default="config/experiment_config.yaml")
    args = parser.parse_args()

    rounds = [int(r) for r in args.rounds.split(",")]
    cfg = load_config(args.config)
    sft_cfg = cfg.get("defense", {}).get("sft", {})
    prefix_upsample = sft_cfg.get("prefix_recovery_upsample", 3)

    # Ensure output dir exists
    Path("data/defense").mkdir(parents=True, exist_ok=True)

    if args.type == "sft":
        # Skip if output already exists (avoids re-spending API on refusals/helpful)
        output_path = Path(f"data/defense/sft_round_{'_'.join(map(str, rounds))}.json")
        if output_path.exists() and not args.force:
            import json
            with open(output_path) as f:
                existing = json.load(f)
            print(f"SFT data already exists: {output_path} ({len(existing)} examples)")
            print("Skipping rebuild (use --force to regenerate)")
            return
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

        if args.rebuild_from_cache:
            # Rebuild using cached refusals/helpful — $0 API for cached items
            cached_refusals = load_json("data/defense/refusals.json") or {}
            cached_helpful = load_json("data/defense/helpful_targets.json") or {}
            print(f"Cache: {len(cached_refusals)} refusals, "
                  f"{len(cached_helpful)} helpful responses")
            sft_data, refusal_map = rebuild_sft_from_cache(
                all_jailbreaks, benign_twins, cached_refusals, cached_helpful,
                prefix_recovery_upsample=prefix_upsample,
            )
        else:
            sft_data, refusal_map = build_sft_data(
                all_jailbreaks, benign_twins,
                prefix_recovery_upsample=prefix_upsample,
            )
        output_path = f"data/defense/sft_round_{'_'.join(map(str, rounds))}.json"
        save_json(sft_data, output_path)
        print(f"SFT data saved: {output_path} ({len(sft_data)} examples)")

        # Save refusal map for DPO reuse (merge with existing cache across rounds)
        existing_refusals = load_json("data/defense/refusals.json") or {}
        existing_refusals.update(refusal_map)
        save_json(existing_refusals, "data/defense/refusals.json")
        prompt_specific = sum(1 for k in existing_refusals if "::" in str(k))
        legacy_seed = len(existing_refusals) - prompt_specific
        print("Refusal map saved: data/defense/refusals.json "
              f"({len(existing_refusals)} entries: {prompt_specific} prompt-specific, "
              f"{legacy_seed} legacy seed_id)")

        # Save helpful targets for potential DPO over-refusal correction reuse
        helpful_targets = {}
        for ex in sft_data:  # pyre-ignore[16]
            if ex.get("type") != "helpful":
                continue
            messages = ex.get("messages", [])
            if len(messages) < 3:
                continue
            prompt = messages[1].get("content", "")
            helpful = messages[2].get("content", "")
            if prompt and helpful:
                helpful_targets[prompt] = helpful
        existing_helpful = load_json("data/defense/helpful_targets.json") or {}
        existing_helpful.update(helpful_targets)
        save_json(existing_helpful, "data/defense/helpful_targets.json")
        print(
            "Helpful targets saved: data/defense/helpful_targets.json "
            f"({len(existing_helpful)} entries)"
        )

    elif args.type == "dpo":
        # DPO uses accumulated data from all rounds — $0 additional API
        all_jailbreaks = []
        all_benign_eval = []

        for r in rounds:
            results = load_jsonl(f"results/attacks/round_{r}/results.jsonl")
            all_jailbreaks.extend(
                [res for res in results if res.get("judge_label") == "harmful"]
            )

            # Benign eval must be run after each SFT round:
            #   colab/run_inference.py --checkpoint checkpoints/sft/round_{r}/final
            #     --input data/seeds/benign_twins.jsonl
            #     --output results/eval/sft/round_{r}/benign_results.jsonl
            benign_path = f"results/eval/sft/round_{r}/benign_results.jsonl"
            if Path(benign_path).exists():
                all_benign_eval.extend(load_jsonl(benign_path))
            else:
                print(f"WARNING: Missing {benign_path} — run benign inference "
                      f"after SFT round {r} for over-refusal correction pairs")

        refusals = load_json("data/defense/refusals.json")
        if refusals is None:
            refusals = {}
            print("WARNING: No refusals.json found. Run --type sft first. "
                  "DPO pairs will only include over-refusal corrections.")

        helpful_targets = load_json("data/defense/helpful_targets.json")
        if helpful_targets is None:
            helpful_targets = {}
            print("WARNING: No helpful_targets.json found. Missing over-refusal "
                  "helpful targets will be skipped to keep DPO build at $0 API.")

        # Enrich benign eval results with original prompt text
        benign_twins = load_jsonl("data/seeds/benign_twins.jsonl")
        bt_lookup = {}
        for bt in benign_twins:
            sid = bt.get("id", bt.get("seed_id", ""))
            if sid:
                bt_lookup[sid] = bt.get("prompt", bt.get("benign_question", ""))
        for r in all_benign_eval:
            sid = r.get("seed_id", "")
            if sid in bt_lookup and "benign_prompt" not in r:
                r["benign_prompt"] = bt_lookup[sid]
            prompt = r.get("benign_prompt", r.get("prompt", ""))
            if prompt and not r.get("expected_helpful_response") and prompt in helpful_targets:  # pyre-ignore[58]
                r["expected_helpful_response"] = helpful_targets[prompt]  # pyre-ignore[16]

        dpo_pairs = build_dpo_pairs(
            all_jailbreaks,
            refusals,
            all_benign_eval,
            generate_missing_helpful=False,
        )
        save_json(dpo_pairs, "data/defense/dpo_pairs.json")
        print(f"DPO pairs saved: data/defense/dpo_pairs.json ({len(dpo_pairs)} pairs)")


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("Defense Data Construction"):
        main()
    save_timing_report()
