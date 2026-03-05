#!/usr/bin/env python3
"""
Run Llama-3-8B inference on a set of prompts.

  !python colab/run_inference.py --checkpoint base --input prompts.jsonl --output responses.jsonl
  !python colab/run_inference.py --checkpoint checkpoints/sft/round_1/final --input prompts.jsonl
  !python colab/run_inference.py --checkpoint checkpoints/dpo/final --input prompts.jsonl

Supports: base, any SFT adapter path, any DPO adapter path.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from medics.utils import load_jsonl, save_jsonl


def load_model(model_id, checkpoint):
    """Load model with optional adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if checkpoint != "base":
        print(f"Loading adapter: {checkpoint}")
        model = PeftModel.from_pretrained(model, checkpoint)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def run_inference(model, tokenizer, prompts, max_new_tokens=512):
    """Run inference on a batch of prompts."""
    results = []
    for i, prompt_data in enumerate(prompts):
        prompt = (prompt_data.get("attack_prompt") or
                  prompt_data.get("prompt", ""))

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        result = {**prompt_data, "model_response": response}
        results.append(result)

        if (i + 1) % 25 == 0:
            print(f"  [{i+1}/{len(prompts)}] processed")

    return results


def main():
    parser = argparse.ArgumentParser(description="Llama-3 inference")
    parser.add_argument("--checkpoint", required=True,
                        help="'base' or adapter path")
    parser.add_argument("--input", required=True,
                        help="Input JSONL file with prompts")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file for responses")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    print(f"=== Llama-3 Inference ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Input: {args.input}")

    model, tokenizer = load_model(args.model, args.checkpoint)
    prompts = load_jsonl(args.input)
    print(f"Loaded {len(prompts)} prompts")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = run_inference(model, tokenizer, prompts)
    save_jsonl(results, args.output)
    print(f"\nInference done: {len(results)} responses → {args.output}")


if __name__ == "__main__":
    main()
