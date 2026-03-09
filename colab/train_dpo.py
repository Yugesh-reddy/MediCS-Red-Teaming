#!/usr/bin/env python3
"""
DPO preference optimization. Run ONCE after final SFT round.

  !python colab/train_dpo.py --sft-checkpoint checkpoints/sft/round_3/final

Reads: data/defense/dpo_pairs.json
Saves: checkpoints/dpo/final/
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

from medics.utils import load_config


def main():
    parser = argparse.ArgumentParser(description="DPO preference training")
    parser.add_argument("--sft-checkpoint", required=True,
                        help="Path to final SFT adapter")
    parser.add_argument("--model", default=None,
                        help="Override model ID from config")
    parser.add_argument("--config", default="config/experiment_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = args.model or cfg["target_model"]["model_id"]
    dpo_cfg = cfg["defense"]["dpo"]
    lora_cfg = dpo_cfg["lora"]
    train_cfg = dpo_cfg["training"]

    print("=== DPO Preference Optimization ===")
    print(f"Model: {model_id}")

    # --- Load base + merge SFT ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg["target_model"].get("quantization", "4bit-nf4").replace("4bit-", ""),
        bnb_4bit_compute_dtype=torch.bfloat16,
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

    config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["per_device_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        beta=train_cfg["beta"],
        loss_type=train_cfg.get("loss_type", "sigmoid"),
        fp16=train_cfg.get("fp16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        eval_strategy=train_cfg.get("eval_strategy", "epoch") if eval_dataset else "no",
        max_length=train_cfg.get("max_length", 1024),
        max_prompt_length=train_cfg.get("max_prompt_length", 512),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        seed=train_cfg.get("seed", 42),
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"\nDPO done. Loss: {result.metrics['train_loss']:.4f}")
    print(f"Saved: {output_dir}/final")


if __name__ == "__main__":
    main()
