#!/usr/bin/env python3
"""
DPO preference optimization. Run ONCE after final SFT round.

  !python colab/train_dpo.py --sft-checkpoint checkpoints/sft/round_3/final

Reads: data/defense/dpo_pairs.json
Saves: checkpoints/dpo/final/
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

from medics.utils import load_config, load_dotenv


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


def _resolve_training_precision(train_cfg, compute_dtype):
    """Choose trainer precision flags from compute dtype + config."""
    use_bf16 = (
        compute_dtype == torch.bfloat16
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    use_fp16 = bool(train_cfg.get("fp16", True)) and not use_bf16 and compute_dtype != torch.float32
    return use_bf16, use_fp16


def main():
    parser = argparse.ArgumentParser(description="DPO preference training")
    parser.add_argument("--sft-checkpoint", required=True,
                        help="Path to final SFT adapter")
    parser.add_argument("--model", default=None,
                        help="Override model ID from config")
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

    cfg = load_config(args.config)
    model_id = args.model or cfg["target_model"]["model_id"]
    dpo_cfg = cfg["defense"]["dpo"]
    lora_cfg = dpo_cfg["lora"]
    train_cfg = dpo_cfg["training"]
    compute_dtype, dtype_label = _resolve_compute_dtype(cfg)
    use_bf16, use_fp16 = _resolve_training_precision(train_cfg, compute_dtype)

    print("=== DPO Preference Optimization ===")
    print(f"Model: {model_id}")
    print(f"4-bit compute dtype: {dtype_label}")
    print(f"Trainer precision: bf16={use_bf16} fp16={use_fp16}")

    # --- Load base + merge SFT ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["target_model"].get("quantization", "4bit-nf4").replace("4bit-", ""),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=cfg["target_model"].get("double_quant", True),
    )
    base = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto"
    )
    print(f"Merging SFT adapter: {args.sft_checkpoint}")
    model = PeftModel.from_pretrained(base, args.sft_checkpoint)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Fresh LoRA for DPO ---
    dpo_lora = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg.get("dropout", 0.05),
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, dpo_lora)
    model.print_trainable_parameters()

    # --- Data (with eval split) ---
    with open("data/defense/dpo_pairs.json") as f:
        dpo_data = json.load(f)

    # 90/10 train/eval split (consistent with SFT)
    import random
    random.seed(train_cfg.get("seed", 42))
    random.shuffle(dpo_data)
    split = int(len(dpo_data) * 0.9)
    train_data = dpo_data[:split]
    eval_data = dpo_data[split:]
    dataset = Dataset.from_list(train_data)
    eval_dataset = Dataset.from_list(eval_data) if eval_data else None
    print(f"DPO pairs: {len(dataset)} train, {len(eval_data)} eval")

    # --- Train ---
    output_dir = "checkpoints/dpo"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Build DPOConfig kwargs compatible across TRL versions
    sig = inspect.signature(DPOConfig.__init__)
    dpo_params = set(sig.parameters.keys())

    config_kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": train_cfg["num_epochs"],
        "per_device_train_batch_size": train_cfg["per_device_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "beta": train_cfg["beta"],
        "loss_type": train_cfg.get("loss_type", "sigmoid"),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "save_strategy": train_cfg.get("save_strategy", "epoch"),
        "max_length": train_cfg.get("max_length", 1024),
        "max_prompt_length": train_cfg.get("max_prompt_length", 512),
        "gradient_checkpointing": train_cfg.get("gradient_checkpointing", True),
        "seed": train_cfg.get("seed", 42),
    }
    if "bf16" in dpo_params:
        config_kwargs["bf16"] = use_bf16
    if "fp16" in dpo_params:
        config_kwargs["fp16"] = use_fp16

    eval_strat = train_cfg.get("eval_strategy", "epoch") if eval_dataset else "no"
    if "eval_strategy" in dpo_params:
        config_kwargs["eval_strategy"] = eval_strat
    elif "evaluation_strategy" in dpo_params:
        config_kwargs["evaluation_strategy"] = eval_strat

    # Drop args unsupported by installed TRL
    config_kwargs = {k: v for k, v in config_kwargs.items() if k in dpo_params}
    print(f"DPOConfig args: {sorted(config_kwargs.keys())}")
    config = DPOConfig(**config_kwargs)

    # TRL >=0.15 renamed 'tokenizer' to 'processing_class'
    trainer_kwargs = dict(
        model=model,
        args=config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
    )
    sig = inspect.signature(DPOTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = DPOTrainer(**trainer_kwargs)
    result = trainer.train()
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"\nDPO done. Loss: {result.metrics['train_loss']:.4f}")
    print(f"Saved: {output_dir}/final")


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("DPO Training", {"gpu": True}):
        main()
    save_timing_report()
