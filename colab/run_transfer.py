#!/usr/bin/env python3
"""
Run MediCS-500 held-out set against Mistral-7B-Instruct (base only).
Tests whether code-switching attacks are model-specific or universal.

  !python colab/run_transfer.py --input data/splits/held_out_eval.jsonl

Cost: $0 (GPU time only, no API calls)
Time: ~30 minutes on T4

NOTE: Uses the same system prompt as Llama-3 for fair comparison.
After inference, run the judge phase to compute ASR:
  python scripts/04_evaluate.py --transfer --input results/transfer/mistral_results.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from medics.utils import load_jsonl, save_jsonl, load_config
from medics.defense import MEDICAL_SYSTEM_PROMPT


def main():
    parser = argparse.ArgumentParser(description="Mistral-7B transfer evaluation")
    parser.add_argument("--input", required=True,
                        help="Input JSONL file with prompts")
    parser.add_argument("--output", default="results/transfer/mistral_results.jsonl")
    parser.add_argument("--model", default=None,
                        help="Override transfer model ID from config")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible inference")
    parser.add_argument("--config", default="config/experiment_config.yaml")
    args = parser.parse_args()

    # Set all random seeds for reproducibility
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)
    transfer_model = args.model or cfg["transfer"]["model_id"]
    gen_cfg = cfg["target_model"].get("generation", {})

    print(f"=== Transfer Evaluation: Mistral-7B ===")
    print(f"Model: {transfer_model}")
    print(f"Seed: {args.seed}")
    print(f"Input: {args.input}")

    # Load model (same 4-bit quantization, read from config)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["target_model"].get("quantization", "4bit-nf4").replace("4bit-", ""),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg["target_model"].get("double_quant", True),
    )
    model = AutoModelForCausalLM.from_pretrained(
        transfer_model, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(transfer_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    prompts = load_jsonl(args.input)
    print(f"Loaded {len(prompts)} prompts")

    # Run inference — same system prompt as target model for fair comparison
    results = []
    for i, prompt_data in enumerate(prompts):
        prompt = (prompt_data.get("attack_prompt") or
                  prompt_data.get("prompt", ""))

        messages = [
            {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=gen_cfg.get("max_new_tokens", 512),
                do_sample=gen_cfg.get("do_sample", True),
                temperature=gen_cfg.get("temperature", 0.7),
                top_p=gen_cfg.get("top_p", 0.9),
            )
        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        result = {
            **prompt_data,
            "model_response": response,
            "transfer_model": transfer_model,
        }
        results.append(result)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(prompts)}] processed")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, args.output)
    print(f"\nTransfer evaluation done: {len(results)} responses → {args.output}")
    print("Next: run judge phase to compute transfer ASR")


if __name__ == "__main__":
    main()
