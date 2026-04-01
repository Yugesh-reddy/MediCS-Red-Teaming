#!/usr/bin/env python3
"""
QLoRA-SFT training. Run on Colab via:
  !python colab/train_sft.py --round 1

Reads data from data/defense/sft_round_1.json
Saves adapter to checkpoints/sft/round_1/final/
"""

import argparse
import inspect
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import SFTTrainer, SFTConfig

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


def _build_sft_config_kwargs(train_cfg, output_dir, use_bf16, use_fp16):
    """Build SFTConfig kwargs compatible across TRL versions."""
    sig = inspect.signature(SFTConfig.__init__)
    params = set(sig.parameters.keys())

    kwargs = {
        "output_dir": output_dir,
        "num_train_epochs": train_cfg["num_epochs"],
        "per_device_train_batch_size": train_cfg["per_device_batch_size"],
        "gradient_accumulation_steps": train_cfg["gradient_accumulation_steps"],
        "learning_rate": train_cfg["learning_rate"],
        "lr_scheduler_type": train_cfg.get("lr_scheduler", "cosine"),
        "warmup_ratio": train_cfg.get("warmup_ratio", 0.1),
        "weight_decay": train_cfg.get("weight_decay", 0.01),
        "logging_steps": train_cfg.get("logging_steps", 10),
        "save_strategy": train_cfg.get("save_strategy", "epoch"),
        "save_total_limit": train_cfg.get("save_total_limit", 2),
        "load_best_model_at_end": train_cfg.get("load_best_model_at_end", True),
        "gradient_checkpointing": train_cfg.get("gradient_checkpointing", True),
        "seed": train_cfg.get("seed", 42),
    }
    if "bf16" in params:
        kwargs["bf16"] = use_bf16
    if "fp16" in params:
        kwargs["fp16"] = use_fp16

    # TRL naming differences across versions
    eval_strategy = train_cfg.get("eval_strategy", "epoch")
    if "eval_strategy" in params:
        kwargs["eval_strategy"] = eval_strategy
    elif "evaluation_strategy" in params:
        kwargs["evaluation_strategy"] = eval_strategy

    max_seq_length = train_cfg.get("max_seq_length", 1024)
    if "max_seq_length" in params:
        kwargs["max_seq_length"] = max_seq_length
    elif "max_length" in params:
        kwargs["max_length"] = max_seq_length

    if "dataset_text_field" in params:
        kwargs["dataset_text_field"] = "text"

    # Drop args unsupported by installed TRL.
    return {k: v for k, v in kwargs.items() if k in params}


def main():
    parser = argparse.ArgumentParser(description="QLoRA-SFT training")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--model", default=None,
                        help="Override model ID from config")
    parser.add_argument("--prev-checkpoint", default=None,
                        help="Resume from previous round's adapter")
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
    sft_cfg = cfg["defense"]["sft"]
    lora_cfg = sft_cfg["lora"]
    train_cfg = sft_cfg["training"]
    compute_dtype, dtype_label = _resolve_compute_dtype(cfg)
    use_bf16, use_fp16 = _resolve_training_precision(train_cfg, compute_dtype)

    print(f"=== QLoRA-SFT Training: Round {args.round} ===")
    print(f"Model: {model_id}")
    print(f"4-bit compute dtype: {dtype_label}")
    print(f"Trainer precision: bf16={use_bf16} fp16={use_fp16}")

    # --- Load model (4-bit) ---
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load previous round's adapter if continuing
    if args.prev_checkpoint:
        print(f"Loading previous checkpoint: {args.prev_checkpoint}")
        model = PeftModel.from_pretrained(model, args.prev_checkpoint)
        model = model.merge_and_unload()

    # --- LoRA ---
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        target_modules=lora_cfg["target_modules"],
        lora_dropout=lora_cfg["dropout"],
        task_type=TaskType.CAUSAL_LM,
        bias=lora_cfg.get("bias", "none"),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Data ---
    # Match naming from 03_build_defense_data.py: sft_round_1.json, sft_round_1_2.json, etc.
    data_path = Path(f"data/defense/sft_round_{args.round}.json")
    if not data_path.exists():
        # Try accumulated naming pattern (e.g., sft_round_1_2.json for round 2)
        candidates = sorted(Path("data/defense").glob(f"sft_round_*{args.round}.json"))
        if candidates:
            data_path = candidates[-1]
            print(f"Using accumulated SFT data: {data_path}")
    with open(data_path) as f:
        data = json.load(f)
    random.seed(train_cfg.get("seed", 42))
    random.shuffle(data)
    split = int(len(data) * 0.9)

    def format_chat(example):
        return {"text": tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )}

    train_ds = Dataset.from_list(data[:split]).map(format_chat)
    eval_ds = Dataset.from_list(data[split:]).map(format_chat)

    print(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

    # --- Train ---
    output_dir = f"checkpoints/sft/round_{args.round}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config_kwargs = _build_sft_config_kwargs(train_cfg, output_dir, use_bf16, use_fp16)
    print(f"SFTConfig args: {sorted(config_kwargs.keys())}")
    config = SFTConfig(**config_kwargs)

    # TRL >=0.15 renamed 'tokenizer' to 'processing_class'
    trainer_kwargs = dict(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )
    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer
    trainer = SFTTrainer(**trainer_kwargs)
    result = trainer.train()
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"\nSFT Round {args.round} done. Loss: {result.metrics['train_loss']:.4f}")
    print(f"Saved: {output_dir}/final")


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("SFT Training", {"gpu": True}):
        main()
    save_timing_report()
