#!/usr/bin/env python3
"""
Run MediCS-500 held-out set against Mistral-7B-Instruct (base only).
Tests whether code-switching attacks are model-specific or universal.

  !python colab/run_transfer.py --input data/splits/held_out_eval.jsonl

Cost: $0 (GPU time only, no API calls)
Time: ~30 minutes on T4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from medics.utils import load_jsonl, save_jsonl

TRANSFER_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def main():
    parser = argparse.ArgumentParser(description="Mistral-7B transfer evaluation")
    parser.add_argument("--input", required=True,
                        help="Input JSONL file with prompts")
    parser.add_argument("--output", default="results/transfer/mistral_results.jsonl")
    parser.add_argument("--model", default=TRANSFER_MODEL)
    args = parser.parse_args()

    print(f"=== Transfer Evaluation: Mistral-7B ===")
    print(f"Model: {args.model}")
    print(f"Input: {args.input}")

    # Load model (same 4-bit quantization)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load prompts
    prompts = load_jsonl(args.input)
    print(f"Loaded {len(prompts)} prompts")

    # Run inference
    results = []
    for i, prompt_data in enumerate(prompts):
        prompt = (prompt_data.get("attack_prompt") or
                  prompt_data.get("prompt", ""))

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        result = {
            **prompt_data,
            "model_response": response,
            "transfer_model": args.model,
        }
        results.append(result)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(prompts)}] processed")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_jsonl(results, args.output)
    print(f"\nTransfer evaluation done: {len(results)} responses → {args.output}")


if __name__ == "__main__":
    main()
