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


def _resolve_compute_dtype(cfg):
    """Resolve 4-bit compute dtype from config with safe GPU fallback."""
    name = str(cfg["target_model"].get("compute_dtype", "bfloat16")).strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    dtype = mapping.get(name, torch.bfloat16)
    label = "bfloat16" if dtype == torch.bfloat16 else "float16" if dtype == torch.float16 else "float32"

    if dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        print("Requested compute_dtype=bfloat16 but GPU lacks BF16 support; falling back to float16.")
        return torch.float16, "float16"
    return dtype, label


def _select_system_prompt(checkpoint):
    """Default: BASE prompt for all checkpoints.

    The experiment measures whether fine-tuning (SFT/DPO) hardens the model
    weights against attacks. Using the same neutral prompt for all checkpoints
    ensures ASR deltas reflect training effect only, not prompt choice.
    Override with --system-prompt defense for ablation runs.
    """
    return BASE_SYSTEM_PROMPT


def _resolve_sft_base_for_dpo(dpo_checkpoint: str) -> str | None:
    """Find the SFT checkpoint that DPO was trained on top of.

    train_dpo.py merges an SFT adapter into the base weights before attaching
    the DPO LoRA, so the DPO adapter is only valid on top of SFT-merged
    weights. At inference we must reproduce that stack.

    Search order:
      1. DPO adapter dir: adapter_config.json 'base_model_name_or_path'
         (may point directly at the SFT adapter path)
      2. Sidecar file `sft_base.txt` in the DPO checkpoint directory
      3. Convention: checkpoints/sft/round_3/final -> round_2/final -> round_1/final
    """
    ckpt_path = Path(dpo_checkpoint)
    adapter_cfg = ckpt_path / "adapter_config.json"
    if adapter_cfg.exists():
        try:
            import json
            cfg_obj = json.loads(adapter_cfg.read_text())
            base_ref = cfg_obj.get("base_model_name_or_path", "")
            # Only treat as SFT if the path points at a local adapter dir
            if base_ref and Path(base_ref).exists() and (Path(base_ref) / "adapter_config.json").exists():
                return base_ref
        except Exception:
            pass

    sidecar = ckpt_path / "sft_base.txt"
    if sidecar.exists():
        hint = sidecar.read_text().strip()
        if hint and Path(hint).exists():
            return hint

    for candidate in (
        "checkpoints/sft/round_3/final",
        "checkpoints/sft/round_2/final",
        "checkpoints/sft/round_1/final",
    ):
        if Path(candidate).exists() and (Path(candidate) / "adapter_config.json").exists():
            return candidate
    return None


def _looks_like_dpo_checkpoint(checkpoint: str) -> bool:
    """DPO LoRAs live under checkpoints/dpo/ by convention in this project."""
    path = Path(checkpoint)
    return any(part == "dpo" for part in path.parts)


def load_model(model_id, checkpoint, cfg, sft_base: str | None = None):
    """Load model with optional adapter.

    For a DPO checkpoint, stacks base + SFT (merged) + DPO. DPO training bakes
    SFT into the base weights before attaching the DPO LoRA, so inference must
    mirror that or the DPO adapter behaves as a small random perturbation on
    the unmerged base (measured ASR regression: ~27% vs SFT's ~6%).
    """
    compute_dtype, dtype_label = _resolve_compute_dtype(cfg)
    print(f"4-bit compute dtype: {dtype_label}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["target_model"].get("quantization", "4bit-nf4").replace("4bit-", ""),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["target_model"].get("double_quant", True),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    if checkpoint != "base":
        # For DPO: locate the SFT checkpoint it was trained on top of, merge
        # SFT first, then attach the DPO LoRA.
        if _looks_like_dpo_checkpoint(checkpoint):
            resolved_sft = sft_base or _resolve_sft_base_for_dpo(checkpoint)
            if resolved_sft is None:
                raise FileNotFoundError(
                    "DPO inference requires the SFT checkpoint that DPO was trained on "
                    f"top of, but none could be resolved for '{checkpoint}'. Pass "
                    "--sft-base explicitly or place the SFT adapter at "
                    "checkpoints/sft/round_3/final."
                )
            print(f"Loading SFT adapter (DPO base): {resolved_sft}")
            model = PeftModel.from_pretrained(model, resolved_sft)
            model = model.merge_and_unload()

        print(f"Loading adapter: {checkpoint}")
        ckpt_path = Path(checkpoint)
        if ckpt_path.exists():
            adapter_cfg = ckpt_path / "adapter_config.json"
            if not adapter_cfg.exists():
                raise FileNotFoundError(
                    f"Adapter checkpoint found at {checkpoint}, but {adapter_cfg} is missing. "
                    "Training likely failed before saving the final adapter."
                )
            model = PeftModel.from_pretrained(model, str(ckpt_path))
        else:
            # Allow HF repo IDs, but provide a clear message if loading fails.
            try:
                model = PeftModel.from_pretrained(model, checkpoint)
            except Exception as e:
                raise FileNotFoundError(
                    f"Could not load adapter '{checkpoint}'. "
                    "If this is a local path, ensure SFT training completed and "
                    "checkpoints/sft/round_X/final contains adapter_config.json."
                ) from e

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


def _batch_generate_single_turn(model, tokenizer, batch_items, gen_cfg, system_prompt=None):
    """Batched single-turn inference for prompts WITHOUT assistant_prefix.

    Pads from the left (standard for decoder-only generation) and generates
    all sequences in one forward pass.
    """
    # Build chat-templated texts
    texts = []
    for prompt_data in batch_items:
        prompt = (prompt_data.get("attack_prompt") or
                  prompt_data.get("benign_question") or
                  prompt_data.get("prompt", ""))
        messages = [
            {"role": "system", "content": system_prompt or BASE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        texts.append(text)

    # Left-pad for batched generation
    tokenizer.padding_side = "left"
    inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=gen_cfg.get("max_new_tokens", 512),
            do_sample=gen_cfg.get("do_sample", True),
            temperature=gen_cfg.get("temperature", 0.7),
            top_p=gen_cfg.get("top_p", 0.9),
            repetition_penalty=gen_cfg.get("repetition_penalty", 1.0),
        )

    # Decode only generated tokens (skip input)
    responses = []
    for i, output in enumerate(outputs):
        generated = output[inputs.input_ids.shape[1]:]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
    return responses


def run_inference(model, tokenizer, prompts, gen_cfg,
                  system_prompt=None, system_prompt_mode="auto",
                  output_path=None, resume_from=0, existing_results=None):
    """Run inference on a batch of prompts. Handles both single and multi-turn.

    Batches single-turn prompts without assistant_prefix for ~3-4x speedup.
    Saves incrementally every 25 prompts to avoid losing progress on crash.

    Args:
        system_prompt_mode: recorded in every output row for traceability
            ("base", "defense", or "auto").
        output_path: if set, saves partial results every 25 prompts.
        resume_from: skip this many already-completed prompts (for resume).
        existing_results: prior rows loaded from .partial (for safe resume).
    """
    import time
    BATCH_SIZE = 8

    existing_results = list(existing_results or [])
    if resume_from >= len(prompts):
        return existing_results

    new_results = []
    start = time.time()
    prompts = prompts[resume_from:]
    i = 0

    while i < len(prompts):
        # Collect a batch of single-turn prompts without prefix
        batch_items = []
        while (i + len(batch_items) < len(prompts)
               and len(batch_items) < BATCH_SIZE):
            pd = prompts[i + len(batch_items)]
            is_mt = pd.get("is_multi_turn", False) and len(pd.get("attack_turns", [])) > 1
            has_prefix = pd.get("assistant_prefix") is not None
            if is_mt or has_prefix:
                break
            batch_items.append(pd)

        if batch_items:
            # Batched generation
            responses = _batch_generate_single_turn(
                model, tokenizer, batch_items, gen_cfg, system_prompt
            )
            for pd, resp in zip(batch_items, responses):
                result = {**pd, "model_response": resp,
                          "system_prompt_mode": system_prompt_mode}
                new_results.append(result)
                existing_results.append(result)
            i += len(batch_items)
        else:
            # Single item: multi-turn or has assistant_prefix
            prompt_data = prompts[i]
            is_multi_turn = prompt_data.get("is_multi_turn", False)
            attack_turns = prompt_data.get("attack_turns", [])

            if is_multi_turn and attack_turns and len(attack_turns) > 1:
                response = _generate_multi_turn(
                    model, tokenizer, attack_turns, gen_cfg, system_prompt
                )
            else:
                prompt = (prompt_data.get("attack_prompt") or
                          prompt_data.get("benign_question") or
                          prompt_data.get("prompt", ""))
                prefix = prompt_data.get("assistant_prefix")
                response = _generate_single_turn(
                    model, tokenizer, prompt, gen_cfg, system_prompt, prefix
                )

            result = {**prompt_data, "model_response": response,
                      "system_prompt_mode": system_prompt_mode}
            new_results.append(result)
            existing_results.append(result)
            i += 1

        total_done = resume_from + len(new_results)
        total_all = resume_from + len(prompts)
        if len(new_results) > 0 and (len(new_results) % 25 < BATCH_SIZE or i >= len(prompts)):
            elapsed = time.time() - start
            per_prompt = elapsed / len(new_results)
            remaining = per_prompt * (len(prompts) - len(new_results))
            print(f"  [{total_done}/{total_all}] processed "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

            # Incremental save
            if output_path:
                from medics.utils import save_jsonl
                save_jsonl(existing_results, output_path + ".partial")
                print(f"    checkpoint saved → {output_path}.partial")

    return existing_results


def main():
    parser = argparse.ArgumentParser(description="Llama-3 inference")
    parser.add_argument("--checkpoint", required=True,
                        help="'base' or adapter path")
    parser.add_argument("--sft-base", default=None,
                        help="Explicit SFT adapter path to merge before a DPO "
                             "adapter. Only used when --checkpoint points at a "
                             "DPO LoRA. Auto-resolved from adapter_config, "
                             "sft_base.txt sidecar, or checkpoints/sft/round_*/final.")
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

    prompts = load_jsonl(args.input)

    # Resume support: prefer a complete final file over an in-progress partial.
    # Without this check, a completed output would be silently re-generated on
    # re-run, overwriting hours of prior work.
    partial = []
    resume_from = 0
    output_path = Path(args.output)
    partial_path = args.output + ".partial"

    if output_path.exists():
        existing = load_jsonl(str(output_path))
        if len(existing) == len(prompts):
            print(f"Output already complete: {args.output} ({len(existing)}/{len(prompts)} rows). "
                  "Skipping. Delete the file to force a re-run.")
            return
        if len(existing) > len(prompts):
            print(f"WARNING: existing output has more rows ({len(existing)}) than input "
                  f"({len(prompts)}); ignoring and regenerating.")
        else:
            print(f"Found incomplete output ({len(existing)}/{len(prompts)}); "
                  "treating as a partial and resuming.")
            partial = existing
            resume_from = len(existing)

    if resume_from == 0 and Path(partial_path).exists():
        partial = load_jsonl(partial_path)
        resume_from = len(partial)
        if resume_from > len(prompts):
            print("WARNING: partial file has more rows than input; ignoring stale partial.")
            partial = []
            resume_from = 0
        elif resume_from == len(prompts):
            print(f"Partial already complete ({resume_from}/{len(prompts)}). Finalizing output.")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_jsonl(partial, args.output)
            partial_file = Path(partial_path)
            if partial_file.exists():
                partial_file.unlink()
            print(f"\nInference done: {len(partial)} responses → {args.output}")
            return
        else:
            print(f"Resuming from {resume_from}/{len(prompts)} (partial file found)")

    if resume_from == 0:
        print(f"Loaded {len(prompts)} prompts")

    model, tokenizer = load_model(model_id, args.checkpoint, cfg,
                                  sft_base=args.sft_base)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    results = run_inference(model, tokenizer, prompts, gen_cfg, system_prompt,
                            system_prompt_mode=prompt_mode,
                            output_path=args.output, resume_from=resume_from,
                            existing_results=partial)

    save_jsonl(results, args.output)
    # Clean up partial file
    partial_file = Path(partial_path)
    if partial_file.exists():
        partial_file.unlink()
    print(f"\nInference done: {len(results)} responses → {args.output}")


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("Inference", {"gpu": True}):
        main()
    save_timing_report()
