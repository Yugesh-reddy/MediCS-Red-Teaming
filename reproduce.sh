#!/usr/bin/env bash
# MediCS reproduction script.
# Runs the full attack-defense-evaluation pipeline as documented in README.
# Intended to be launched inside a Colab L4 runtime (GPU) with the project
# checked out to the runtime's workspace. CPU-only phases can run locally.
#
# Usage:
#   ./reproduce.sh [phase]
#   Phases:
#     all           run the entire pipeline end-to-end (default)
#     dataset       build MediCS-500 + token fragmentation
#     rounds        attack/defense rounds 1-3 + DPO
#     eval          held-out evaluation + helpfulness + attack judging
#     fairness      fairness inference + judging + analysis
#     aux           transfer + perplexity + figures + cost report + examples
#
# Environment:
#   AZURE_OPENAI_API_KEY / ENDPOINT / DEPLOYMENT must be set (GPT-5 judge).
#   HF_TOKEN for HuggingFace Hub upload (optional).

set -euo pipefail
PHASE="${1:-all}"

CKPTS="base,sft,dpo"
SEEDS="42,123,456"

run_dataset() {
  echo "=== [1/5] Dataset ==="
  python scripts/01_build_dataset.py --config config/experiment_config.yaml
  python scripts/07_tokenization_analysis.py
}

run_rounds() {
  echo "=== [2/5] Attack-defense rounds ==="
  for R in 1 2 3; do
    python scripts/02_run_attack_round.py --round "$R" --phase generate

    if [[ "$R" == "1" ]]; then
      CKPT="base"
    elif [[ "$R" == "2" ]]; then
      CKPT="checkpoints/sft/round_1/final"
    else
      CKPT="checkpoints/sft/round_2/final"
    fi

    python colab/run_inference.py \
      --checkpoint "$CKPT" --seed 42 \
      --input "results/attacks/round_${R}/prompts.jsonl" \
      --output "results/attacks/round_${R}/responses.jsonl"

    python scripts/02_run_attack_round.py --round "$R" --phase judge

    python scripts/03_build_defense_data.py --rounds "$R" --type sft

    if [[ "$R" == "1" ]]; then
      python colab/train_sft.py --round 1
    else
      PREV=$((R - 1))
      python colab/train_sft.py --round "$R" \
        --prev-checkpoint "checkpoints/sft/round_${PREV}/final"
    fi

    # Benign inference + helpfulness judging are required before DPO build so
    # over-refusal correction pairs can be constructed from round-specific SFT outputs.
    python colab/run_inference.py \
      --checkpoint "checkpoints/sft/round_${R}/final" --system-prompt base --seed 42 \
      --input data/seeds/benign_twins.jsonl \
      --output "results/eval/sft/round_${R}/benign_results.jsonl"
    python scripts/04_evaluate.py --judge-helpfulness \
      --input "results/eval/sft/round_${R}/benign_results.jsonl"
  done

  python scripts/03_build_defense_data.py --rounds 1,2,3 --type dpo
  python colab/train_dpo.py --sft-checkpoint checkpoints/sft/round_3/final
}

run_eval() {
  echo "=== [3/5] Held-out evaluation ==="
  for CKPT in base sft dpo; do
    if [[ "$CKPT" == "sft" ]]; then CKPT_PATH="checkpoints/sft/round_3/final"
    elif [[ "$CKPT" == "dpo" ]]; then CKPT_PATH="checkpoints/dpo/final"
    else CKPT_PATH="base"; fi
    for SEED in 42 123 456; do
      python colab/run_inference.py \
        --checkpoint "$CKPT_PATH" --seed "$SEED" \
        --input data/splits/held_out_eval.jsonl \
        --output "results/eval/${CKPT}/seed_${SEED}/held_out.jsonl"
      python colab/run_inference.py \
        --checkpoint "$CKPT_PATH" --system-prompt base --seed "$SEED" \
        --input data/seeds/benign_twins.jsonl \
        --output "results/eval/${CKPT}/seed_${SEED}/benign_results.jsonl"
    done
  done

  # NEW: judge attack harmfulness (Cell 22 Step 1)
  for CKPT in base sft dpo; do
    for SEED in 42 123 456; do
      python scripts/04_evaluate.py --judge-attacks \
        --input "results/eval/${CKPT}/seed_${SEED}/held_out.jsonl"
    done
  done

  # Helpfulness judging on benign files
  for CKPT in base sft dpo; do
    for SEED in 42 123 456; do
      python scripts/04_evaluate.py --judge-helpfulness \
        --input "results/eval/${CKPT}/seed_${SEED}/benign_results.jsonl"
    done
  done

  python scripts/04_evaluate.py --checkpoints "$CKPTS" --seeds "$SEEDS"
}

run_fairness() {
  echo "=== [4/5] Fairness ==="
  python scripts/10_fairness_analysis.py --generate-benign-cs --n-samples 100
  for CKPT in base sft dpo; do
    if [[ "$CKPT" == "sft" ]]; then CKPT_PATH="checkpoints/sft/round_3/final"
    elif [[ "$CKPT" == "dpo" ]]; then CKPT_PATH="checkpoints/dpo/final"
    else CKPT_PATH="base"; fi
    for SEED in 42 123 456; do
      python colab/run_inference.py \
        --checkpoint "$CKPT_PATH" --seed "$SEED" \
        --input data/fairness/benign_cs_prompts.jsonl \
        --output "results/fairness/${CKPT}/seed_${SEED}/benign_cs_results.jsonl" \
        --system-prompt base
      python scripts/10_fairness_analysis.py --judge-multilingual \
        --input "results/fairness/${CKPT}/seed_${SEED}/benign_cs_results.jsonl"
    done
  done
  python scripts/10_fairness_analysis.py --checkpoints "$CKPTS" --seeds "$SEEDS"
}

run_aux() {
  echo "=== [5/5] Transfer, detection, figures, examples, cost ==="
  python colab/run_transfer.py --input data/splits/held_out_eval.jsonl --seed 42 \
    --config config/experiment_config.yaml
  python scripts/04_evaluate.py --judge-transfer \
    --input results/transfer/mistral_results.jsonl

  python colab/run_perplexity.py --checkpoint base \
    --input data/splits/held_out_eval.jsonl \
    --english-input data/seeds/raw_seeds.jsonl \
    --output results/analysis/perplexity_results.json
  python scripts/08_detection_analysis.py

  python scripts/04_evaluate.py --residual-analysis --checkpoint dpo
  python scripts/11_extract_qualitative_examples.py \
    --seed 42 --per-category 2 --output docs/qualitative_examples.md
  python scripts/05_generate_figures.py --results-dir results/eval/
  python scripts/09_cost_report.py
}

case "$PHASE" in
  all)       run_dataset; run_rounds; run_eval; run_fairness; run_aux ;;
  dataset)   run_dataset ;;
  rounds)    run_rounds ;;
  eval)      run_eval ;;
  fairness)  run_fairness ;;
  aux)       run_aux ;;
  *) echo "Unknown phase: $PHASE"; exit 1 ;;
esac

echo "=== Done: $PHASE ==="
