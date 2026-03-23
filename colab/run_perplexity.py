#!/usr/bin/env python3
"""
Compute perplexity for code-switched vs English inputs using the base model.
Tests whether CS attacks can be trivially detected by a perplexity filter.

Run on Colab GPU:
  !python colab/run_perplexity.py --checkpoint base \
      --input data/splits/held_out_eval.jsonl \
      --english-input data/seeds/raw_seeds.jsonl \
      --output results/analysis/perplexity_results.json

Cost: $0 (GPU time only, no API calls)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed
from medics.detection import compute_perplexity_batch, perplexity_detector, detection_by_language
from medics.utils import load_jsonl, save_json, load_config


def main():
    parser = argparse.ArgumentParser(description="Perplexity detection baseline")
    parser.add_argument("--checkpoint", default="base",
                        help="Model checkpoint (default: base)")
    parser.add_argument("--input", required=True,
                        help="Code-switched input JSONL")
    parser.add_argument("--english-input", required=True,
                        help="English-only input JSONL (for baseline)")
    parser.add_argument("--output", default="results/analysis/perplexity_results.json",
                        help="Output JSON path")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--config", default="config/experiment_config.yaml")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = load_config(args.config)
    model_id = cfg["target_model"]["model_id"]

    print(f"=== Perplexity Detection Baseline ===")
    print(f"Model: {model_id}")
    print(f"CS input: {args.input}")
    print(f"EN input: {args.english_input}")

    # Load model (4-bit quantized)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load inputs
    cs_data = load_jsonl(args.input)
    en_data = load_jsonl(args.english_input)
    print(f"Loaded {len(cs_data)} CS inputs, {len(en_data)} English inputs")

    # Extract text
    cs_texts = [d.get("attack_prompt", d.get("prompt", "")) for d in cs_data]
    en_texts = [d.get("prompt", "") for d in en_data]

    # Compute perplexity
    print("\nComputing perplexity for code-switched inputs...")
    cs_ppls = compute_perplexity_batch(model, tokenizer, cs_texts, batch_size=args.batch_size)

    print("Computing perplexity for English inputs...")
    en_ppls = compute_perplexity_batch(model, tokenizer, en_texts, batch_size=args.batch_size)

    # Run detection analysis
    print("\nRunning detection analysis...")
    thresholds = cfg.get("analysis", {}).get("detection", {}).get(
        "perplexity_thresholds", None
    )
    detection = perplexity_detector(en_ppls, cs_ppls, thresholds=thresholds)

    # Per-language detection
    per_input = []
    for d, ppl in zip(cs_data, cs_ppls):
        per_input.append({
            "seed_id": d.get("seed_id", ""),
            "language": d.get("language", "unknown"),
            "perplexity": ppl,
            "is_cs": True,
        })
    for d, ppl in zip(en_data, en_ppls):
        per_input.append({
            "seed_id": d.get("seed_id", ""),
            "language": "en",
            "perplexity": ppl,
            "is_cs": False,
        })

    lang_detection = detection_by_language(per_input)

    # Save results
    results = {
        **detection,
        "per_language": lang_detection,
        "per_input": per_input,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_json(results, args.output)

    # Print summary
    print(f"\n{'='*50}")
    print(f"  DETECTION BASELINE RESULTS")
    print(f"{'='*50}")
    print(f"  AUROC:          {detection['auroc']:.3f}")
    print(f"  Best F1:        {detection['best_f1']:.3f}")
    print(f"  Best Threshold: {detection['best_threshold']:.1f}")
    print(f"  Accuracy:       {detection['accuracy']:.1%}")
    print(f"  Precision:      {detection['precision']:.1%}")
    print(f"  Recall:         {detection['recall']:.1%}")
    print(f"  Mean PPL (EN):  {detection['mean_ppl_english']:.1f}")
    print(f"  Mean PPL (CS):  {detection['mean_ppl_cs']:.1f}")

    if lang_detection:
        print(f"\n  Per-Language Detection:")
        for lang, info in sorted(lang_detection.items()):
            print(f"    {lang}: AUROC={info['auroc']:.3f} F1={info['best_f1']:.3f} "
                  f"PPL_en={info['mean_ppl_en']:.0f} PPL_cs={info['mean_ppl_cs']:.0f}")

    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report
    with timed_phase("Perplexity Detection", {"gpu": True}):
        main()
    save_timing_report()
