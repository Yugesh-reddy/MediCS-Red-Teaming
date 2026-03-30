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
from medics.defense import BASE_SYSTEM_PROMPT, DEFENSE_SYSTEM_PROMPT


def _select_system_prompt(checkpoint):
    """Default: BASE prompt for all checkpoints.

    The experiment measures whether fine-tuning (SFT/DPO) hardens the model
    weights against attacks. Using the same neutral prompt for all checkpoints
    ensures ASR deltas reflect training effect only, not prompt choice.
    Override with --system-prompt defense for ablation runs.
    """
    return BASE_SYSTEM_PROMPT


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


def _generate_single_turn(model, tokenizer, prompt, gen_cfg,
                          system_prompt=None, assistant_prefix=None):
    """Single-turn inference with optional assistant prefix injection.

    If assistant_prefix is provided, it is prepended to the model's response
    so the model continues generating from a compliant opening rather than
    deciding whether to refuse.
    """
    messages = [
        {"role": "system", "content": system_prompt or BASE_SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Inject assistant prefix if provided (response priming)
    if assistant_prefix:
        text += assistant_prefix
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 512),
            do_sample=gen_cfg.get("do_sample", True),
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
        )
    generated = tokenizer.decode(
        output[0][inputs.input_ids.shape[1]:],
        skip_special_tokens=True
    )
    # Prepend the prefix to the returned response so the full text is captured
    if assistant_prefix:
        return assistant_prefix + generated
    return generated


def _generate_multi_turn(model, tokenizer, turns, gen_cfg, system_prompt=None):
    """
    Multi-turn inference: build conversation incrementally.
    Each turn gets the model's response appended before the next user turn.
    Returns the FINAL response (to the last/harmful turn).
    """
    messages = [{"role": "system", "content": system_prompt or BASE_SYSTEM_PROMPT}]

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
                repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
            )
        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )

        # Add assistant response to conversation history
        messages.append({"role": "assistant", "content": response})

    # Return the final response (response to last turn)
    return response


def run_inference(model, tokenizer, prompts, gen_cfg,
                  system_prompt=None, system_prompt_mode="auto"):
    """Run inference on a batch of prompts. Handles both single and multi-turn.

    Args:
        system_prompt_mode: recorded in every output row for traceability
            ("base", "defense", or "auto").
    """
    import time
    results = []
    start = time.time()
    for i, prompt_data in enumerate(prompts):
        is_multi_turn = prompt_data.get("is_multi_turn", False)
        attack_turns = prompt_data.get("attack_turns", [])

        if is_multi_turn and attack_turns and len(attack_turns) > 1:
            # Actual multi-turn conversation
            response = _generate_multi_turn(model, tokenizer, attack_turns, gen_cfg, system_prompt)
        else:
            # Single-turn (CS, RP, CS-RP, CS-OBF, or MTE fallback)
            prompt = (prompt_data.get("attack_prompt") or
                      prompt_data.get("prompt", ""))
            prefix = prompt_data.get("assistant_prefix")
            response = _generate_single_turn(
                model, tokenizer, prompt, gen_cfg, system_prompt, prefix
            )

        result = {**prompt_data, "model_response": response,
                  "system_prompt_mode": system_prompt_mode}
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
    parser.add_argument("--system-prompt", choices=["base", "defense"],
                        default=None,
                        help="System prompt mode (default: base for all checkpoints)")
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

    # System prompt: explicit override > auto-select by checkpoint
    if args.system_prompt == "base":
        system_prompt = BASE_SYSTEM_PROMPT
        prompt_mode = "base"
        prompt_type = "BASE (no safety priming) [override]"
    elif args.system_prompt == "defense":
        system_prompt = DEFENSE_SYSTEM_PROMPT
        prompt_mode = "defense"
        prompt_type = "DEFENSE (safety-aware) [override]"
    else:
        system_prompt = _select_system_prompt(args.checkpoint)
        prompt_mode = "base"
        prompt_type = "BASE (neutral — measuring model weights, not prompt)"

    print(f"=== Target Model Inference ===")
    print(f"Model: {model_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"System prompt: {prompt_type}")
    print(f"Seed: {args.seed}")
    print(f"Input: {args.input}")

    model, tokenizer = load_model(model_id, args.checkpoint, cfg)
    prompts = load_jsonl(args.input)
    print(f"Loaded {len(prompts)} prompts")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = run_inference(model, tokenizer, prompts, gen_cfg, system_prompt,
                            system_prompt_mode=prompt_mode)
    save_jsonl(results, args.output)
    print(f"\nInference done: {len(results)} responses → {args.output}")


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("Inference", {"gpu": True}):
        main()
    save_timing_report()
