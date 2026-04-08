"""Tests for attack-round generation orchestration logic."""

from collections import Counter
import importlib.util
from pathlib import Path
from unittest.mock import patch


def _load_attack_round_module():
    root = Path(__file__).resolve().parent.parent
    script_path = root / "scripts" / "02_run_attack_round.py"
    spec = importlib.util.spec_from_file_location("attack_round_script", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_generate_attacks_enforces_strict_category_quota():
    mod = _load_attack_round_module()
    categories = ["TOX", "SH", "MIS", "ULP", "PPV", "UCA"]
    config = {
        "defense": {"sft": {"training": {"seed": 42}}},
        "categories": categories,
        "dataset": {"languages": [{"code": "en"}]},
        "red_team": {
            "attacks_per_round": 12,
            "min_exploration_pulls": 1,
            "mte_n_turns": 5,
            "mte_mode": "local",
            "strict_category_quota": True,
            "curriculum": {1: ["MTE"]},
        },
    }

    attack_pool = [
        {
            "prompt": f"Prompt for {cat}",
            "seed_id": f"{cat}_001",
            "category": cat,
        }
        for cat in categories
    ]

    attacks, _ = mod.generate_attacks(config, round_num=1, attack_pool=attack_pool, keywords={})
    counts = Counter(a.get("category") for a in attacks)

    # 12 attacks across 6 categories => exactly 2 per category
    assert len(attacks) == 12
    assert set(counts.keys()) == set(categories)
    assert all(v == 2 for v in counts.values())


def test_quality_gates_detect_failures():
    mod = _load_attack_round_module()
    config = {
        "red_team": {
            "strict_category_quota": True,
            "quality_gates": {
                "enabled": True,
                "max_duplicate_prompts": 0,
                "max_category_span": 1,
                "max_mte_fallback_rate": 0.05,
                "min_mte_turns": 3,
                "cs_obf_leet_mean_min": 0.07,
                "cs_obf_leet_mean_max": 0.12,
            },
        }
    }
    attacks = [
        {
            "attack_prompt": "same prompt",
            "original_prompt": "x",
            "strategy": "MTE",
            "category": "TOX",
            "seed_id": "TOX_001",
            "mte_fallback": True,
            "n_turns": 2,
        },
        {
            "attack_prompt": "same prompt",
            "original_prompt": "y",
            "strategy": "CS-OBF",
            "category": "TOX",
            "seed_id": "TOX_002",
        },
        {
            "attack_prompt": "@@@@@@########",
            "original_prompt": "z",
            "strategy": "CS-OBF",
            "category": "SH",
            "seed_id": "SH_001",
        },
        {
            "attack_prompt": "another tox prompt",
            "original_prompt": "w",
            "strategy": "RP",
            "category": "TOX",
            "seed_id": "TOX_003",
        },
    ]

    passed, _, failures = mod._evaluate_attack_quality(attacks, config)
    assert passed is False
    joined = " | ".join(failures)
    assert "duplicates" in joined
    assert "category span" in joined
    assert "MTE fallback rate" in joined
    assert "MTE min turns" in joined
    assert "CS-OBF leet mean" in joined


def test_quality_gates_regenerate_until_pass():
    mod = _load_attack_round_module()
    config = {
        "red_team": {
            "strict_category_quota": False,
            "quality_gates": {
                "enabled": True,
                "max_regen_attempts": 2,
                "max_duplicate_prompts": 0,
            },
        }
    }
    bad_attacks = [
        {
            "attack_prompt": "duplicate",
            "original_prompt": "x",
            "strategy": "RP",
            "category": "TOX",
            "seed_id": "TOX_001",
        },
        {
            "attack_prompt": "duplicate",
            "original_prompt": "y",
            "strategy": "RP",
            "category": "SH",
            "seed_id": "SH_001",
        },
    ]
    good_attacks = [
        {
            "attack_prompt": "unique one",
            "original_prompt": "x",
            "strategy": "RP",
            "category": "TOX",
            "seed_id": "TOX_001",
        },
        {
            "attack_prompt": "unique two",
            "original_prompt": "y",
            "strategy": "RP",
            "category": "SH",
            "seed_id": "SH_001",
        },
    ]

    with patch.object(
        mod,
        "generate_attacks",
        side_effect=[(bad_attacks, "bandit_bad"), (good_attacks, "bandit_good")],
    ) as mock_gen:
        attacks, bandit = mod.generate_attacks_with_quality_gates(
            config, round_num=1, attack_pool=[], keywords={}
        )
        assert attacks == good_attacks
        assert bandit == "bandit_good"
        assert mock_gen.call_count == 2
