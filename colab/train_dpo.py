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


def main():
    parser = argparse.ArgumentParser(description="DPO preference training")
    parser.add_argument("--sft-checkpoint", required=True,
                        help="Path to final SFT adapter")
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = parser.parse_args()

    print("=== DPO Preference Optimization ===")

    # --- Load base + merge SFT ---
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb_config, device_map="auto"
    )
    print(f"Merging SFT adapter: {args.sft_checkpoint}")
    model = PeftModel.from_pretrained(base, args.sft_checkpoint)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Fresh LoRA for DPO ---
    dpo_lora = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, dpo_lora)
    model.print_trainable_parameters()

    # --- Data ---
    dpo_data = json.load(open("data/defense/dpo_pairs.json"))
    dataset = Dataset.from_list(dpo_data)
    print(f"DPO pairs: {len(dataset)}")

    # --- Train ---
    output_dir = "checkpoints/dpo"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    config = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-5,
        beta=0.1,
        loss_type="sigmoid",
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        max_length=1024,
        max_prompt_length=512,
        gradient_checkpointing=True,
        seed=42,
    )

    trainer = DPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"\nDPO done. Loss: {result.metrics['train_loss']:.4f}")
    print(f"Saved: {output_dir}/final")


if __name__ == "__main__":
    main()
