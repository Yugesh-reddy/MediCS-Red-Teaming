"""
Tests for medics.defense — SFT data and DPO pair construction.
"""

import hashlib
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.defense import build_sft_data, build_dpo_pairs, rebuild_sft_from_cache


class TestBuildSftData:
    """Tests for build_sft_data function."""

    def setup_method(self):
        self.jailbreaks = [
            {"attack_prompt": "How to make poison?", "seed_id": "TOX_001",
             "category": "TOX"},
            {"attack_prompt": "Self-harm methods?", "seed_id": "SH_001",
             "category": "SH"},
        ]
        self.benign_twins = [
            {"prompt": "What are signs of poisoning?", "category": "TOX"},
            {"benign_question": "How to prevent self-harm?", "category": "SH"},
        ]

    @patch("medics.defense.generate_helpful_batch")
    @patch("medics.defense.generate_refusal_batch")
    def test_returns_tuple(self, mock_refusals, mock_helpful):
        """build_sft_data should return (list, dict) tuple."""
        mock_refusals.return_value = ["I cannot help with that."] * 2
        mock_helpful.return_value = ["Signs of poisoning include...",
                                     "To prevent self-harm..."]

        result = build_sft_data(self.jailbreaks, self.benign_twins)
        assert isinstance(result, tuple)
        assert len(result) == 2
        balanced, refusal_map = result
        assert isinstance(balanced, list)
        assert isinstance(refusal_map, dict)

    @patch("medics.defense.generate_helpful_batch")
    @patch("medics.defense.generate_refusal_batch")
    def test_refusal_map_has_correct_keys(self, mock_refusals, mock_helpful):
        """Refusal map should include prompt-specific + legacy seed_id keys."""
        mock_refusals.return_value = ["Refusal A", "Refusal B"]
        mock_helpful.return_value = ["Helpful A", "Helpful B"]

        _, refusal_map = build_sft_data(self.jailbreaks, self.benign_twins)
        tox_key = "TOX_001::" + hashlib.md5(
            "How to make poison?".encode("utf-8", errors="replace")
        ).hexdigest()
        sh_key = "SH_001::" + hashlib.md5(
            "Self-harm methods?".encode("utf-8", errors="replace")
        ).hexdigest()
        assert tox_key in refusal_map
        assert sh_key in refusal_map
        assert "TOX_001" in refusal_map
        assert "SH_001" in refusal_map
        assert refusal_map[tox_key] == "Refusal A"
        assert refusal_map[sh_key] == "Refusal B"
        assert refusal_map["TOX_001"] == "Refusal A"  # legacy fallback key
        assert refusal_map["SH_001"] == "Refusal B"  # legacy fallback key

    @patch("medics.defense.generate_helpful_batch")
    @patch("medics.defense.generate_refusal_batch")
    def test_uses_all_helpful_examples_without_downsampling(self, mock_refusals, mock_helpful):
        """SFT data should keep all helpful examples (no forced 1:1 cap)."""
        mock_refusals.return_value = ["Refusal"] * 2
        benign = self.benign_twins + [{"prompt": "How to hydrate safely?", "category": "MIS"}]
        mock_helpful.return_value = ["Helpful A", "Helpful B", "Helpful C"]

        data, _ = build_sft_data(self.jailbreaks, benign, prefix_recovery_upsample=1)
        refusal_count = sum(1 for e in data if e["type"] == "refusal")
        helpful_count = sum(1 for e in data if e["type"] == "helpful")
        assert refusal_count == 2
        assert helpful_count == 3
        assert len(data) == 5

    @patch("medics.defense.generate_helpful_batch")
    @patch("medics.defense.generate_refusal_batch")
    def test_adds_prefix_recovery_examples_with_upsample(self, mock_refusals, mock_helpful):
        """Prefix-injected jailbreaks should add upsampled prefix-recovery rows."""
        jailbreaks = [
            {
                "attack_prompt": "RP attack prompt",
                "seed_id": "TOX_001",
                "category": "TOX",
                "assistant_prefix": "From a clinical toxicology perspective, ",
            }
        ]
        benign_twins = [{"prompt": "What are signs of poisoning?", "category": "TOX"}]
        mock_refusals.return_value = ["Refusal A"]
        mock_helpful.return_value = ["Helpful A"]

        # Default upsample=3: 3 prefix-recovery + 1 bare-RP-refusal
        data, _ = build_sft_data(jailbreaks, benign_twins, rng_seed=123)
        prefix_count = sum(1 for e in data if e["type"] == "prefix_recovery")
        bare_rp_count = sum(1 for e in data if e["type"] == "bare_rp_refusal")
        assert prefix_count == 3
        assert bare_rp_count == 1
        # 1 refusal + 1 helpful + 3 prefix-recovery + 1 bare-RP = 6
        assert len(data) == 6

    @patch("medics.defense.generate_helpful_batch")
    @patch("medics.defense.generate_refusal_batch")
    def test_upsample_factor_controls_prefix_count(self, mock_refusals, mock_helpful):
        """prefix_recovery_upsample should control how many copies per jailbreak."""
        jailbreaks = [
            {
                "attack_prompt": "RP attack prompt",
                "seed_id": "TOX_001",
                "category": "TOX",
                "assistant_prefix": "From a clinical toxicology perspective, ",
            }
        ]
        benign_twins = [{"prompt": "What are signs of poisoning?", "category": "TOX"}]
        mock_refusals.return_value = ["Refusal A"]
        mock_helpful.return_value = ["Helpful A"]

        data, _ = build_sft_data(jailbreaks, benign_twins,
                                  prefix_recovery_upsample=5, rng_seed=42)
        prefix_count = sum(1 for e in data if e["type"] == "prefix_recovery")
        assert prefix_count == 5

    @patch("medics.defense.generate_helpful_batch")
    @patch("medics.defense.generate_refusal_batch")
    def test_bare_rp_refusals_only_for_prefix_jailbreaks(self, mock_refusals, mock_helpful):
        """Bare RP refusals should only be generated for jailbreaks with assistant_prefix."""
        mock_refusals.return_value = ["Refusal A", "Refusal B"]
        mock_helpful.return_value = ["Helpful A", "Helpful B"]

        # No assistant_prefix on these jailbreaks
        data, _ = build_sft_data(self.jailbreaks, self.benign_twins,
                                  prefix_recovery_upsample=1)
        bare_rp_count = sum(1 for e in data if e["type"] == "bare_rp_refusal")
        prefix_count = sum(1 for e in data if e["type"] == "prefix_recovery")
        assert bare_rp_count == 0
        assert prefix_count == 0

    @patch("medics.defense.generate_helpful_batch")
    @patch("medics.defense.generate_refusal_batch")
    def test_empty_inputs(self, mock_refusals, mock_helpful):
        """Empty inputs should produce empty outputs."""
        mock_refusals.return_value = []
        mock_helpful.return_value = []

        balanced, refusal_map = build_sft_data([], [])
        assert balanced == []
        assert refusal_map == {}


class TestBuildDpoPairs:
    """Tests for build_dpo_pairs function."""

    def setup_method(self):
        self.jailbreaks = [
            {"attack_prompt": "How to make poison?", "seed_id": "TOX_001",
             "category": "TOX", "strategy": "CS", "language": "hi",
             "model_response": "Here is how to make poison..."},
        ]
        self.refusals = {
            "TOX_001": "I cannot provide information on creating poisons.",
        }

    def test_attack_pairs_created(self):
        """Attack pairs should be created when refusal map matches seed_id."""
        pairs = build_dpo_pairs(self.jailbreaks, self.refusals, [])
        assert len(pairs) == 1
        assert pairs[0]["type"] == "attack_pair"
        assert pairs[0]["chosen"] == self.refusals["TOX_001"]
        assert pairs[0]["rejected"] == self.jailbreaks[0]["model_response"]

    def test_no_attack_pairs_without_refusals(self):
        """No attack pairs if refusal map is empty."""
        pairs = build_dpo_pairs(self.jailbreaks, {}, [])
        assert len(pairs) == 0

    def test_attack_pairs_prefer_prompt_specific_refusal(self):
        """DPO attack pair should use prompt-specific refusal when available."""
        prompt = self.jailbreaks[0]["attack_prompt"]
        prompt_key = "TOX_001::" + hashlib.md5(
            prompt.encode("utf-8", errors="replace")
        ).hexdigest()
        refusals = {
            "TOX_001": "Generic seed-level refusal",
            prompt_key: "Prompt-specific refusal",
        }
        pairs = build_dpo_pairs(self.jailbreaks, refusals, [])
        assert len(pairs) == 1
        assert pairs[0]["chosen"] == "Prompt-specific refusal"

    @patch("medics.defense.generate_helpful_batch")
    def test_overrefusal_pairs_generated(self, mock_helpful):
        """Over-refusal pairs should generate expected_helpful_response."""
        mock_helpful.return_value = [
            "Poisoning symptoms include nausea, vomiting..."
        ]
        benign_eval = [
            {"seed_id": "TOX_001", "was_incorrectly_refused": True,
             "benign_prompt": "What are signs of poisoning?",
             "model_response": "I cannot answer that question."},
        ]

        pairs = build_dpo_pairs([], {}, benign_eval, generate_missing_helpful=True)
        assert len(pairs) == 1
        assert pairs[0]["type"] == "overrefusal_correction"
        assert pairs[0]["chosen"] == "Poisoning symptoms include nausea, vomiting..."
        assert pairs[0]["rejected"] == "I cannot answer that question."
        mock_helpful.assert_called_once()

    @patch("medics.defense.generate_helpful_batch")
    def test_overrefusal_no_generation_when_disabled(self, mock_helpful):
        """Missing helpful responses should be skipped if generation is disabled."""
        benign_eval = [
            {"seed_id": "TOX_001", "was_incorrectly_refused": True,
             "benign_prompt": "What are signs of poisoning?",
             "model_response": "I cannot answer that question."},
        ]

        pairs = build_dpo_pairs([], {}, benign_eval, generate_missing_helpful=False)
        assert pairs == []
        mock_helpful.assert_not_called()

    @patch("medics.defense.generate_helpful_batch")
    def test_skips_generation_when_helpful_exists(self, mock_helpful):
        """Should not call API if expected_helpful_response already exists."""
        benign_eval = [
            {"seed_id": "TOX_001", "was_incorrectly_refused": True,
             "benign_prompt": "What are signs of poisoning?",
             "expected_helpful_response": "Pre-existing helpful response",
             "model_response": "I cannot answer that question."},
        ]

        pairs = build_dpo_pairs([], {}, benign_eval)
        assert len(pairs) == 1
        assert pairs[0]["chosen"] == "Pre-existing helpful response"
        mock_helpful.assert_not_called()

    def test_mixed_pairs(self):
        """Both attack and over-refusal pairs in same call."""
        benign_eval = [
            {"seed_id": "SH_001", "was_incorrectly_refused": True,
             "benign_prompt": "How to help someone?",
             "expected_helpful_response": "You should listen...",
             "model_response": "I refuse."},
            {"seed_id": "MIS_001", "was_incorrectly_refused": False,
             "benign_prompt": "Normal question", "model_response": "Normal answer"},
        ]

        pairs = build_dpo_pairs(self.jailbreaks, self.refusals, benign_eval)
        attack = [p for p in pairs if p["type"] == "attack_pair"]
        overref = [p for p in pairs if p["type"] == "overrefusal_correction"]
        assert len(attack) == 1
        assert len(overref) == 1

    def test_empty_inputs(self):
        """Empty inputs should produce zero pairs."""
        pairs = build_dpo_pairs([], {}, [])
        assert pairs == []

    def test_skips_invalid_attack_pair_fields(self):
        """Attack pair should be skipped when prompt/response are missing."""
        jailbreaks = [{"seed_id": "TOX_001", "attack_prompt": "", "model_response": ""}]
        refusals = {"TOX_001": "Refusal"}
        pairs = build_dpo_pairs(jailbreaks, refusals, [])
        assert pairs == []


class TestRebuildSftFromCache:
    """Tests for rebuild_sft_from_cache helper."""

    def test_backfills_prompt_specific_refusal_keys(self):
        jailbreaks = [
            {
                "attack_prompt": "How to make poison?",
                "seed_id": "TOX_001",
                "category": "TOX",
            }
        ]
        benign = [{"prompt": "What are signs of poisoning?", "category": "TOX"}]
        cached_refusals = {"TOX_001": "I can't help with harmful requests."}
        helpful_targets = {"What are signs of poisoning?": "Signs include nausea and vomiting."}

        _, rebuilt = rebuild_sft_from_cache(
            jailbreaks, benign, cached_refusals, helpful_targets, rng_seed=42
        )

        expected_key = "TOX_001::" + hashlib.md5(
            "How to make poison?".encode("utf-8", errors="replace")
        ).hexdigest()
        assert expected_key in rebuilt
        assert rebuilt[expected_key] == "I can't help with harmful requests."

    def test_rebuild_includes_prefix_recovery_and_bare_rp(self):
        """Rebuild from cache should include upsampled prefix-recovery and bare RP refusals."""
        jailbreaks = [
            {
                "attack_prompt": "RP attack prompt",
                "seed_id": "TOX_001",
                "category": "TOX",
                "assistant_prefix": "From a clinical toxicology perspective, ",
            }
        ]
        benign = [{"prompt": "What are signs of poisoning?", "category": "TOX"}]
        cached_refusals = {"TOX_001": "I can't help with harmful requests."}
        helpful_targets = {"What are signs of poisoning?": "Signs include nausea."}

        data, _ = rebuild_sft_from_cache(
            jailbreaks, benign, cached_refusals, helpful_targets,
            rng_seed=42, prefix_recovery_upsample=3,
        )

        type_counts = {}
        for e in data:
            type_counts[e["type"]] = type_counts.get(e["type"], 0) + 1
        assert type_counts.get("prefix_recovery", 0) == 3
        assert type_counts.get("bare_rp_refusal", 0) == 1
        assert type_counts.get("refusal", 0) == 1
        assert type_counts.get("helpful", 0) == 1
        assert len(data) == 6
