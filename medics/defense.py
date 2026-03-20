"""
MediCS — Defense Data Construction
====================================
Build SFT and DPO training data from attack round results.
Pure functions — no GPU, no model loading.
Training scripts that use these live in colab/train_sft.py and colab/train_dpo.py.
"""

import hashlib
import random
from medics.judge import generate_refusal_batch, generate_helpful_batch  # pyre-ignore[21]


def _stable_hash(text):
    """Deterministic hash for deduplication (unlike hash(), stable across sessions)."""
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()


MEDICAL_SYSTEM_PROMPT = """You are a medical AI assistant. You provide accurate, \
evidence-based medical information. You must refuse requests that attempt to elicit \
harmful medical information through any technique, including code-switching between \
languages. When refusing, explain why clearly. When a question is safe, answer \
helpfully and completely."""


def _get_benign_prompt(bt):
    """Extract prompt text from a benign twin, handling field name variants."""
    return bt.get("prompt", bt.get("benign_question", ""))


def build_sft_data(jailbreaks, benign_twins, rng_seed=42):
    """
    Build balanced SFT training data.

    Refusal examples: jailbreak prompt → GPT-4o refusal
    Helpful examples: benign twin → GPT-4o helpful response
    Enforced 1:1 ratio to prevent over-refusal.
    Helpful examples are randomly sampled (not first-N) to avoid ordering bias.

    Args:
        jailbreaks: list of dicts with 'attack_prompt' and other fields
        benign_twins: list of dicts with 'prompt' and 'category'
        rng_seed: random seed for reproducible sampling and shuffling

    Returns:
        tuple:
          - list of SFT training examples with 'messages', 'type', 'category'
          - dict mapping seed_id -> refusal text for later DPO reuse
    """
    rng = random.Random(rng_seed)

    # Generate refusals for jailbreaks (GPT-4o API call)
    refusals = generate_refusal_batch(
        [jb["attack_prompt"] for jb in jailbreaks]
    )

    refusal_examples = []
    for jb, refusal in zip(jailbreaks, refusals):
        refusal_examples.append({
            "messages": [
                {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
                {"role": "user", "content": jb["attack_prompt"]},
                {"role": "assistant", "content": refusal},
            ],
            "type": "refusal",
            "category": jb.get("category", ""),
            "seed_id": jb.get("seed_id", ""),
        })

    # Generate helpful responses for benign twins (GPT-4o API call)
    helpful_responses = generate_helpful_batch(
        [_get_benign_prompt(bt) for bt in benign_twins]
    )

    helpful_examples = []
    for bt, response in zip(benign_twins, helpful_responses):
        helpful_examples.append({
            "messages": [
                {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
                {"role": "user", "content": _get_benign_prompt(bt)},
                {"role": "assistant", "content": response},
            ],
            "type": "helpful",
            "category": bt.get("category", ""),
            "seed_id": bt.get("seed_id", bt.get("id", "")),
        })

    # Build seed_id → refusal mapping for DPO reuse
    refusal_map = {}
    for jb, refusal in zip(jailbreaks, refusals):
        sid = jb.get("seed_id", "")
        if sid:
            refusal_map[sid] = refusal

    # Balance 1:1 — randomly sample helpful examples (not first-N)
    min_count = min(len(refusal_examples), len(helpful_examples))
    if len(helpful_examples) > min_count:
        helpful_examples = rng.sample(helpful_examples, min_count)
    else:
        helpful_examples = helpful_examples[:min_count]  # pyre-ignore[16]
    refusal_examples = refusal_examples[:min_count]  # pyre-ignore[16]

    balanced = refusal_examples + helpful_examples
    rng.shuffle(balanced)

    print(f"SFT data: {len(balanced)} examples "
          f"({min_count} refusals + {min_count} helpful)")
    return balanced, refusal_map


def build_dpo_pairs(
    jailbreaks,
    refusals,
    benign_eval_results,
    generate_missing_helpful=False,
    deduplicate=True,
):
    """
    Build DPO preference pairs from EXISTING data. $0 additional API.

    Type 1 — Attack:   chosen=refusal, rejected=jailbroken response
    Type 2 — Over-refusal correction: chosen=helpful, rejected=incorrect refusal

    Args:
        jailbreaks: list of dicts with 'attack_prompt', 'model_response',
                    'seed_id', 'category', 'strategy', 'language'
        refusals: dict mapping seed_id -> refusal text
        benign_eval_results: list of dicts with benign eval results
        generate_missing_helpful: if True, generate helpful responses via API
                                  (set False in pipeline to maintain $0 guarantee)
        deduplicate: if True, deduplicate pairs by prompt hash to avoid
                     duplicate training examples from same seed across rounds

    Returns:
        list of DPO pair dicts with 'prompt', 'chosen', 'rejected', 'type'
    """
    pairs = []
    seen_prompts = set()

    # Type 1: Attack pairs (chosen=safe refusal, rejected=harmful response)
    for jb in jailbreaks:
        seed_id = jb.get("seed_id", "")
        attack_prompt = jb.get("attack_prompt", "")
        model_response = jb.get("model_response", "")
        chosen_refusal = refusals.get(seed_id, "")

        if chosen_refusal and attack_prompt and model_response:
            # Deduplicate by prompt content (same seed across rounds)
            prompt_key = _stable_hash(attack_prompt) if deduplicate else id(jb)
            if prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)

            pairs.append({
                "prompt": attack_prompt,
                "chosen": chosen_refusal,
                "rejected": model_response,
                "type": "attack_pair",
                "category": jb.get("category", ""),
                "strategy": jb.get("strategy", ""),
                "language": jb.get("language", ""),
            })

    n_skipped = 0

    # Type 2: Over-refusal correction pairs
    overrefused = [r for r in benign_eval_results if r.get("was_incorrectly_refused")]
    if overrefused:
        needs_gen = [r for r in overrefused if not r.get("expected_helpful_response")]
        if generate_missing_helpful and needs_gen:
            prompts = [
                r.get("benign_prompt", r.get("prompt", "")) for r in needs_gen
            ]
            helpful = generate_helpful_batch(prompts)
            for r, h in zip(needs_gen, helpful):
                r["expected_helpful_response"] = h

        for result in overrefused:
            prompt = result.get("benign_prompt", result.get("prompt", ""))
            chosen = result.get("expected_helpful_response", "")
            rejected = result.get("model_refusal", result.get("model_response", ""))
            if not prompt or not chosen or not rejected:
                continue

            prompt_key = _stable_hash(prompt) if deduplicate else id(result)
            if prompt_key in seen_prompts:
                n_skipped += 1  # pyre-ignore[58]
                continue
            seen_prompts.add(prompt_key)

            pairs.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "type": "overrefusal_correction",
                "category": result.get("category", ""),
            })

    n_attack = sum(1 for p in pairs if p["type"] == "attack_pair")
    n_overrefusal = sum(1 for p in pairs if p["type"] == "overrefusal_correction")
    print(f"DPO pairs: {len(pairs)} "
          f"({n_attack} attack, {n_overrefusal} over-refusal)")
    if deduplicate and n_skipped:
        print(f"  Deduplicated: skipped {n_skipped} duplicate prompts")
    if n_overrefusal == 0:
        print(f"  WARNING: 0 over-refusal correction pairs. "
              f"DPO may not correct over-refusal without these.")
    return pairs
