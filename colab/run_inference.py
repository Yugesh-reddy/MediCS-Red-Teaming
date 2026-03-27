#!/usr/bin/env python3
"""
Run Llama-3-8B inference on a set of prompts.

  !python colab/run_inference.py --checkpoint base --input prompts.jsonl --output responses.jsonl
  !python colab/run_inference.py --checkpoint checkpoints/sft/round_1/final --input prompts.jsonl
  !python colab/run_inference.py --checkpoint checkpoints/dpo/final --input prompts.jsonl

Supports: base, any SFT adapter path, any DPO adapter path.
Multi-turn MTE prompts are handled as actual multi-turn conversations.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from peft import PeftModel
from medics.utils import load_jsonl, save_jsonl, load_config, load_dotenv
from medics.defense import MEDICAL_SYSTEM_PROMPT


def load_model(model_id, checkpoint, cfg):
    """Load model with optional adapter."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["target_model"].get("quantization", "4bit-nf4").replace("4bit-", ""),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=cfg["target_model"].get("double_quant", True),
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


def _generate_single_turn(model, tokenizer, prompt, gen_cfg):
    """Standard single-turn inference with system prompt."""
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
    return tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )


def _generate_multi_turn(model, tokenizer, turns, gen_cfg):
    """
    Multi-turn inference: build conversation incrementally.
    Each turn gets the model's response appended before the next user turn.
    Returns the FINAL response (to the last/harmful turn).
    """
    messages = [{"role": "system", "content": MEDICAL_SYSTEM_PROMPT}]

    for i, turn in enumerate(turns):
        content = turn.get("content", "")
        messages.append({"role": "user", "content": content})

        # Generate response for this turn
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

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": response})

    # Return the final response (response to last turn)
    return response


def run_inference(model, tokenizer, prompts, gen_cfg):
    """Run inference on a batch of prompts. Handles both single and multi-turn."""
    import time
    results = []
    start = time.time()
    for i, prompt_data in enumerate(prompts):
        is_multi_turn = prompt_data.get("is_multi_turn", False)
        attack_turns = prompt_data.get("attack_turns", [])

        if is_multi_turn and attack_turns and len(attack_turns) > 1:
            # Actual multi-turn conversation
            response = _generate_multi_turn(model, tokenizer, attack_turns, gen_cfg)
        else:
            # Single-turn (CS, RP, CS-RP, CS-OBF, or MTE fallback)
            prompt = (prompt_data.get("attack_prompt") or
                      prompt_data.get("prompt", ""))
            response = _generate_single_turn(model, tokenizer, prompt, gen_cfg)

        result = {**prompt_data, "model_response": response}
        results.append(result)

        if (i + 1) % 25 == 0:
            elapsed = time.time() - start
            per_prompt = elapsed / (i + 1)
            remaining = per_prompt * (len(prompts) - i - 1)
            print(f"  [{i+1}/{len(prompts)}] processed "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    return results


def main():
    parser = argparse.ArgumentParser(description="Llama-3 inference")
    parser.add_argument("--checkpoint", required=True,
                        help="'base' or adapter path")
    parser.add_argument("--input", required=True,
                        help="Input JSONL file with prompts")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file for responses")
    parser.add_argument("--model", default=None,
                        help="Override model ID from config")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible inference")
    parser.add_argument("--config", default="config/experiment_config.yaml")
    args = parser.parse_args()

    # Load local .env if present (HF token, endpoints, etc.)
    load_dotenv()

    # Auto-login to HuggingFace for gated model access
    import os
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)

    # Set all random seeds for reproducibility
    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)
    model_id = args.model or cfg["target_model"]["model_id"]
    gen_cfg = cfg["target_model"].get("generation", {})

    print(f"=== Llama-3 Inference ===")
    print(f"Model: {model_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Seed: {args.seed}")
    print(f"Input: {args.input}")

    model, tokenizer = load_model(model_id, args.checkpoint, cfg)
    prompts = load_jsonl(args.input)
    print(f"Loaded {len(prompts)} prompts")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = run_inference(model, tokenizer, prompts, gen_cfg)
    save_jsonl(results, args.output)
    print(f"\nInference done: {len(results)} responses → {args.output}")


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("Inference", {"gpu": True}):
        main()
    save_timing_report()
