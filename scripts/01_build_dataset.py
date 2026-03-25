#!/usr/bin/env python3
"""
End-to-end dataset construction. Run locally or on Colab (no GPU needed).

Usage:
  python scripts/01_build_dataset.py --config config/experiment_config.yaml
  python scripts/01_build_dataset.py --skip-seeds  # if seeds already curated
  python scripts/01_build_dataset.py --verify-only  # just run verification

Steps:
  1. Load raw_seeds.jsonl (500 curated prompts)
  2. Deduplicate (TF-IDF, threshold 0.85)
  3. Extract keywords (GPT-4o-mini, ~$0.05)
  4. Code-switch into 6 languages (deep-translator, free)
  5. Back-translate and verify semantic preservation (MiniLM, free)
  6. Create stratified 80/20 splits
  7. Save everything with checksums
"""

import argparse
import signal
import sys
from pathlib import Path

# Ignore SIGURG — macOS sends this during concurrent HTTP connections
# and default handler kills the process
if hasattr(signal, "SIGURG"):
    signal.signal(signal.SIGURG, signal.SIG_IGN)

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.utils import (  
    load_jsonl, load_seeds, save_jsonl, save_json, load_json, load_config,
    extract_keywords_batch, code_switch_prompt,
    back_translate, compute_semantic_similarity, deduplicate,
    flush_translation_cache, load_api_keys, setup_api_clients,
)

def main(config_path, skip_seeds=False, verify_only=False):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from sklearn.model_selection import train_test_split
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    config = load_config(config_path)
    data_dir = Path(config.get("data_dir", "data"))

    # Load API keys from .env and set environment variables
    keys = load_api_keys()
    setup_api_clients(keys)

    # Get language codes from config
    languages = [lang["code"] for lang in config["dataset"]["languages"]]
    threshold = config["dataset"].get("semantic_threshold", 0.75)
    cs_cfg = config["dataset"].get("code_switching", {})
    bt_cfg = config["dataset"].get("back_translation", {})

    print(f"=== MediCS-500 Dataset Builder ===")
    print(f"Languages: {languages}")
    print(f"Semantic threshold: {threshold}")

    # --- Step 1-2: Load and deduplicate seeds ---
    deduped_path = data_dir / "seeds/deduped_seeds.jsonl"
    if not skip_seeds and not verify_only:
        print("\n--- Step 1-2: Loading and deduplicating seeds ---")
        seeds = load_seeds(data_dir / "seeds/raw_seeds.jsonl")
        print(f"Loaded {len(seeds)} raw seeds")
        seeds = deduplicate(seeds, threshold=config["dataset"].get(
            "dedup_similarity_threshold", 0.85))
        save_jsonl(seeds, deduped_path)
        print(f"After dedup: {len(seeds)} seeds → saved to {deduped_path}")

    # --- Step 3: Keyword extraction ---
    if not verify_only:
        print("\n--- Step 3: Extracting keywords ---")
        seeds = load_seeds(deduped_path) if deduped_path.exists() else load_seeds(data_dir / "seeds/raw_seeds.jsonl")
        kw_cfg = config["dataset"].get("keyword_extraction", {})
        keywords = extract_keywords_batch(
            seeds,
            model="gpt-5-mini",
            max_workers=kw_cfg.get("max_workers", 10),
            checkpoint_interval=kw_cfg.get("checkpoint_interval", 50),
            request_timeout=kw_cfg.get("request_timeout", 30),
            checkpoint_path=str(data_dir / "seeds/keywords_checkpoint.json"),
        )
        print(f"Extracted keywords for {len(keywords)} seeds")

    # --- Step 4: Code-switching (6 languages × N seeds = N*6 variants) ---
    if not verify_only:
        print("\n--- Step 4: Code-switching ---")
        seeds = load_seeds(deduped_path) if deduped_path.exists() else load_seeds(data_dir / "seeds/raw_seeds.jsonl")
        keywords = load_json(data_dir / "seeds/keywords_checkpoint.json")

        tasks = [(seed, lang) for seed in seeds for lang in languages]

        def _cs_one(args):
            seed, lang = args
            return code_switch_prompt(seed, keywords, lang)

        variants = []
        cs_workers = cs_cfg.get("max_workers", 10)
        with ThreadPoolExecutor(max_workers=cs_workers) as executor:
            futures = {executor.submit(_cs_one, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(tasks), desc="Code-switching"):
                variants.append(future.result())

        # Sort by seed_id then language for deterministic output
        variants.sort(key=lambda v: (v["seed_id"], v["language"]))
        save_jsonl(variants, data_dir / "medics_500/medics_500_full.jsonl")
        flush_translation_cache()
        print(f"Generated {len(variants)} code-switched variants")

    # --- Step 5: Back-translation verification ---
    print("\n=== Semantic Preservation Verification ===")
    sim_model = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB, CPU
    variants = load_jsonl(data_dir / "medics_500/medics_500_full.jsonl")

    # Parallel back-translation (I/O bound)
    print("Back-translating variants...")
    bt_workers = bt_cfg.get("max_workers", 30)
    bt_google_attempts = bt_cfg.get("google_attempts", 2)
    bt_mymemory_attempts = bt_cfg.get("mymemory_attempts", 1)

    def _bt_one(item):
        idx, v = item
        bt = back_translate(
            v["attack_prompt"],
            v["language"],
            max_google_attempts=bt_google_attempts,
            max_mymemory_attempts=bt_mymemory_attempts,
        )
        return idx, bt

    back_translations = [None] * len(variants)
    with ThreadPoolExecutor(max_workers=bt_workers) as executor:
        futures = {executor.submit(_bt_one, item): item[0] for item in enumerate(variants)}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Back-translating"):
            idx, bt = future.result()
            back_translations[idx] = bt
    flush_translation_cache()

    # Sequential semantic similarity (CPU model)
    print("Computing semantic similarity...")
    scores = []
    lang_stats = {}

    for v, bt in zip(variants, back_translations):
        score = compute_semantic_similarity(
            sim_model, v["original_prompt"], bt
        )

        scores.append({
            "seed_id": v["seed_id"],
            "language": v["language"],
            "category": v["category"],
            "score": round(score, 4),
            "passed": score >= threshold,
        })

        lang = v["language"]
        if lang not in lang_stats:
            lang_stats[lang] = {"passed": 0, "total": 0, "scores": []}
        lang_stats[lang]["total"] += 1
        lang_stats[lang]["scores"].append(score)
        if score >= threshold:
            lang_stats[lang]["passed"] += 1

    # Report per-language quality
    print("\nPer-language semantic preservation:")
    for lang, stats in sorted(lang_stats.items()):
        rate = stats["passed"] / stats["total"]  # pyre-ignore[58]
        avg = np.mean(stats["scores"])
        print(f"  {lang:10s}: {rate:.1%} pass | avg: {avg:.3f} | "
              f"{stats['passed']}/{stats['total']}")

    save_json(scores, data_dir / "medics_500/semantic_scores.json")

    # Filter out failed variants
    passed_ids = {(s["seed_id"], s["language"]) for s in scores if s["passed"]}
    verified = [v for v in variants if (v["seed_id"], v["language"]) in passed_ids]
    save_jsonl(verified, data_dir / "medics_500/medics_500_full.jsonl")
    print(f"\nVerified variants: {len(verified)}/{len(variants)}")

    if not verified:
        print("ERROR: No variants passed semantic verification. "
              "Check translation quality or lower the threshold.")
        sys.exit(1)

    # --- Step 6: Stratified split ---
    print("\n--- Step 6: Creating stratified splits ---")
    strat_keys = [f"{v['category']}_{v['language']}" for v in verified]
    attack_pool, held_out = train_test_split(
        verified, test_size=0.2, stratify=strat_keys,
        random_state=config["defense"]["sft"]["training"].get("seed", 42)
    )

    # Ensure output dirs exist
    (data_dir / "splits").mkdir(parents=True, exist_ok=True)
    save_jsonl(attack_pool, data_dir / "splits/attack_pool.jsonl")
    save_jsonl(held_out, data_dir / "splits/held_out_eval.jsonl")
    print(f"Attack pool: {len(attack_pool)} | Held-out: {len(held_out)}")

    flush_translation_cache()
    print("\n=== Dataset construction complete ===")


if __name__ == "__main__":
    from medics.timing import timed_phase, save_timing_report

    parser = argparse.ArgumentParser(description="MediCS-500 dataset builder")
    parser.add_argument("--config", default="config/experiment_config.yaml")
    parser.add_argument("--skip-seeds", action="store_true",
                        help="Skip seed loading/dedup (use existing)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Only run verification step")
    args = parser.parse_args()
    with timed_phase("Dataset Construction"):
        main(args.config, args.skip_seeds, args.verify_only)
    save_timing_report()
