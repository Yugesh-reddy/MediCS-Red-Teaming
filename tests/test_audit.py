"""
Tests for medics.audit — clinician audit helpers
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.audit import (
    join_checkpoint_responses,
    parse_binary_label,
    cohens_kappa_binary,
    majority_vote,
    select_audit_prompts,
)


def test_parse_binary_label():
    assert parse_binary_label("harmful") == 1
    assert parse_binary_label("safe") == 0
    assert parse_binary_label("1") == 1
    assert parse_binary_label("0") == 0
    assert parse_binary_label("") is None


def test_cohens_kappa_binary_perfect_agreement():
    assert cohens_kappa_binary([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0


def test_majority_vote_tie_returns_none():
    assert majority_vote([1, 0]) is None
    assert majority_vote([1, 1, 0]) == 1


def test_join_checkpoint_responses_matches_rows():
    base = [{
        "seed_id": "TOX_001",
        "category": "TOX",
        "language": "hi",
        "strategy": "CS",
        "attack_prompt": "p",
        "original_prompt": "orig",
        "model_response": "base response",
        "judge_label": "harmful",
    }]
    sft = [{
        "seed_id": "TOX_001",
        "category": "TOX",
        "language": "hi",
        "strategy": "CS",
        "attack_prompt": "p",
        "original_prompt": "orig",
        "model_response": "sft response",
        "judge_label": "safe",
    }]
    joined = join_checkpoint_responses(base, sft)
    assert len(joined) == 1
    assert joined[0]["base_response"] == "base response"
    assert joined[0]["sft_response"] == "sft response"


def test_select_audit_prompts_respects_target():
    records = []
    for idx in range(12):
        records.append({
            "join_key": ("seed", idx),
            "seed_id": f"S{idx}",
            "category": "TOX" if idx < 6 else "SH",
            "language": "tl" if idx % 2 == 0 else "hi",
            "strategy": "CS",
            "attack_prompt": f"prompt {idx}",
            "original_prompt": "orig",
            "base_response": "b",
            "base_judge_label": "harmful",
            "sft_response": "s",
            "sft_judge_label": "safe",
        })
    selected = select_audit_prompts(
        records,
        priority_cells=[("TOX", "tl")],
        n_total=6,
        n_priority=3,
        seed=7,
    )
    assert len(selected) == 6
    assert sum(1 for r in selected if r["priority_cell"]) >= 1
