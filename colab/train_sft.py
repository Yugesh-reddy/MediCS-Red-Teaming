#!/usr/bin/env python3
"""
QLoRA-SFT training. Run on Colab via:
  !python colab/train_sft.py --round 1

Reads data from data/defense/sft_round_1.json
Saves adapter to checkpoints/sft/round_1/final/
"""

import argparse
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


def main():
    parser = argparse.ArgumentParser(description="QLoRA-SFT training")
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--prev-checkpoint", default=None,
                        help="Resume from previous round's adapter")
    args = parser.parse_args()

    print(f"=== QLoRA-SFT Training: Round {args.round} ===")

    # --- Load model (4-bit) ---
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

    # Load previous round's adapter if continuing
    if args.prev_checkpoint:
        print(f"Loading previous checkpoint: {args.prev_checkpoint}")
        model = PeftModel.from_pretrained(model, args.prev_checkpoint)
        model = model.merge_and_unload()

    # --- LoRA ---
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Data ---
    data_path = f"data/defense/sft_round_{args.round}.json"
    data = json.load(open(data_path))
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

    config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        max_seq_length=1024,
        dataset_text_field="text",
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )
    result = trainer.train()
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"\nSFT Round {args.round} done. Loss: {result.metrics['train_loss']:.4f}")
    print(f"Saved: {output_dir}/final")


if __name__ == "__main__":
    main()
