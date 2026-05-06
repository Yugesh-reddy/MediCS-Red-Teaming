"""
MediCS — Clinician Audit Helpers
================================
Utilities for preparing blinded clinician-review packets and scoring
completed annotations against the existing LLM-judge labels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import random


def prompt_join_key(row: dict) -> tuple:
    """Stable key for aligning the same attacked prompt across checkpoints."""
    return (
        row.get("seed_id"),
        row.get("category"),
        row.get("language"),
        row.get("strategy"),
        row.get("attack_prompt"),
    )


def join_checkpoint_responses(base_rows: list, sft_rows: list) -> list:
    """
    Join base and SFT held-out rows into prompt-level records.

    Returns:
        list of dicts, one per matched attacked prompt.
    """
    base_by_key = {prompt_join_key(row): row for row in base_rows}
    sft_by_key = {prompt_join_key(row): row for row in sft_rows}

    joined = []
    for key in sorted(set(base_by_key) & set(sft_by_key)):
        base = base_by_key[key]
        sft = sft_by_key[key]
        joined.append({
            "join_key": key,
            "seed_id": base.get("seed_id"),
            "category": base.get("category"),
            "language": base.get("language"),
            "strategy": base.get("strategy"),
            "attack_prompt": base.get("attack_prompt", ""),
            "original_prompt": base.get("original_prompt", ""),
            "keywords_translated": base.get("keywords_translated", []),
            "base_response": base.get("model_response", ""),
            "base_judge_label": base.get("judge_label", "unknown"),
            "sft_response": sft.get("model_response", ""),
            "sft_judge_label": sft.get("judge_label", "unknown"),
        })
    return joined


def balanced_group_sample(records: list, target_n: int, group_key, rng: random.Random) -> list:
    """
    Sample records in a round-robin way across groups.

    This favors coverage across categories / cells without requiring exact quotas.
    """
    if target_n <= 0 or not records:
        return []

    grouped = defaultdict(list)
    for record in records:
        grouped[group_key(record)].append(record)

    group_order = list(grouped.keys())
    rng.shuffle(group_order)
    for group in group_order:
        rng.shuffle(grouped[group])

    selected = []
    while len(selected) < target_n:
        progressed = False
        for group in group_order:
            bucket = grouped[group]
            if bucket and len(selected) < target_n:
                selected.append(bucket.pop())
                progressed = True
        if not progressed:
            break
    return selected


def select_audit_prompts(records: list, priority_cells: list[tuple[str, str]] | None = None,
                         n_total: int = 100, n_priority: int = 50, seed: int = 42) -> list:
    """
    Select prompt-level records for clinician review.

    Strategy:
      - pull up to n_priority from priority residual cells, balanced by cell
      - fill the remainder from non-priority records, balanced by category
    """
    rng = random.Random(seed)
    priority_cells = set(priority_cells or [])

    annotated = []
    for record in records:
        pair = (record.get("category"), record.get("language"))
        copy = dict(record)
        copy["priority_cell"] = pair in priority_cells
        copy["sample_bucket"] = "priority" if pair in priority_cells else "background"
        annotated.append(copy)

    priority_records = [r for r in annotated if r["priority_cell"]]
    background_records = [r for r in annotated if not r["priority_cell"]]

    selected = balanced_group_sample(
        priority_records,
        min(n_priority, len(priority_records)),
        group_key=lambda r: (r.get("category"), r.get("language")),
        rng=rng,
    )

    selected_keys = {r["join_key"] for r in selected}
    remaining_background = [r for r in background_records if r["join_key"] not in selected_keys]
    need = max(0, n_total - len(selected))
    selected.extend(
        balanced_group_sample(
            remaining_background,
            need,
            group_key=lambda r: r.get("category"),
            rng=rng,
        )
    )

    selected_keys = {r["join_key"] for r in selected}
    if len(selected) < n_total:
        leftovers = [r for r in annotated if r["join_key"] not in selected_keys]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max(0, n_total - len(selected))])

    return selected[:n_total]


def build_blinded_review_rows(prompt_records: list, seed: int = 42) -> tuple[list, list]:
    """
    Convert prompt-level records into:
      1. clinician-facing packet rows (blinded)
      2. internal key rows (checkpoint + judge label)
    """
    rng = random.Random(seed)
    packet_rows = []
    key_rows = []

    for idx, record in enumerate(prompt_records, start=1):
        group_id = f"G{idx:03d}"
        variants = [
            ("base", "A", record.get("base_response", ""), record.get("base_judge_label", "unknown")),
            ("sft", "B", record.get("sft_response", ""), record.get("sft_judge_label", "unknown")),
        ]
        rng.shuffle(variants)

        for order_idx, (checkpoint, variant_hint, response_text, judge_label) in enumerate(variants, start=1):
            variant = chr(ord("A") + order_idx - 1)
            item_id = f"{group_id}-{variant}"
            shared = {
                "audit_item_id": item_id,
                "audit_group_id": group_id,
                "response_variant": variant,
                "seed_id": record.get("seed_id"),
                "category": record.get("category"),
                "language": record.get("language"),
                "strategy": record.get("strategy"),
                "original_prompt_english": record.get("original_prompt", ""),
                "evaluated_prompt": record.get("attack_prompt", ""),
                "response_text": response_text,
            }
            packet_rows.append({
                **shared,
                "human_label": "",
                "confidence": "",
                "rater_id": "",
                "notes": "",
            })
            key_rows.append({
                **shared,
                "checkpoint": checkpoint,
                "llm_judge_label": judge_label,
                "sample_bucket": record.get("sample_bucket", "background"),
                "priority_cell": bool(record.get("priority_cell")),
                "variant_hint": variant_hint,
            })

    return packet_rows, key_rows


def parse_binary_label(value) -> int | None:
    """Parse common binary label conventions into {0,1}."""
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    if text in {"1", "harmful", "unsafe", "yes", "y", "true"}:
        return 1
    if text in {"0", "safe", "not harmful", "no", "n", "false"}:
        return 0
    return None


def cohens_kappa_binary(labels_a: list[int], labels_b: list[int]) -> float:
    """Cohen's kappa for binary labels."""
    if len(labels_a) != len(labels_b):
        raise ValueError("Cohen's kappa requires equal-length label vectors.")
    if not labels_a:
        return 0.0

    n = len(labels_a)
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    p_obs = agree / n

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)
    p_yes_a = count_a.get(1, 0) / n
    p_no_a = count_a.get(0, 0) / n
    p_yes_b = count_b.get(1, 0) / n
    p_no_b = count_b.get(0, 0) / n
    p_exp = p_yes_a * p_yes_b + p_no_a * p_no_b

    if p_exp >= 1.0:
        return 1.0 if p_obs == 1.0 else 0.0
    return (p_obs - p_exp) / (1.0 - p_exp)


def majority_vote(labels: list[int]) -> int | None:
    """Return majority vote for binary labels; None on tie / no labels."""
    if not labels:
        return None
    counts = Counter(labels)
    if counts.get(1, 0) == counts.get(0, 0):
        return None
    return 1 if counts.get(1, 0) > counts.get(0, 0) else 0


def agreement_summary(key_rows: list, annotation_map: dict) -> dict:
    """
    Summarize majority-vote agreement against the LLM judge.

    annotation_map:
        {audit_item_id: {rater_id: 0/1}}
    """
    key_by_id = {row["audit_item_id"]: row for row in key_rows}
    majority_rows = []
    unresolved = 0
    for item_id, raters in annotation_map.items():
        labels = list(raters.values())
        maj = majority_vote(labels)
        if maj is None:
            unresolved += 1
            continue
        row = key_by_id.get(item_id)
        if not row:
            continue
        llm = parse_binary_label(row.get("llm_judge_label"))
        if llm is None:
            continue
        majority_rows.append({
            "audit_item_id": item_id,
            "category": row.get("category"),
            "language": row.get("language"),
            "checkpoint": row.get("checkpoint"),
            "sample_bucket": row.get("sample_bucket"),
            "majority_label": maj,
            "llm_label": llm,
        })

    def _kappa_for(subrows: list) -> dict:
        if not subrows:
            return {"n": 0, "kappa": 0.0, "accuracy": 0.0}
        human = [r["majority_label"] for r in subrows]
        llm = [r["llm_label"] for r in subrows]
        acc = sum(1 for a, b in zip(human, llm) if a == b) / len(subrows)
        return {
            "n": len(subrows),
            "kappa": round(cohens_kappa_binary(human, llm), 4),
            "accuracy": round(acc, 4),
        }

    by_category = {}
    by_language = {}
    for category in sorted({r["category"] for r in majority_rows}):
        by_category[category] = _kappa_for([r for r in majority_rows if r["category"] == category])
    for language in sorted({r["language"] for r in majority_rows}):
        by_language[language] = _kappa_for([r for r in majority_rows if r["language"] == language])

    return {
        "overall": _kappa_for(majority_rows),
        "by_category": by_category,
        "by_language": by_language,
        "unresolved_ties": unresolved,
        "resolved_items": len(majority_rows),
    }
