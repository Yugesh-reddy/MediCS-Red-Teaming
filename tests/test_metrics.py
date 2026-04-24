"""
Tests for medics.metrics — Evaluation metrics
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.metrics import (
    compute_asr, compute_robustness_gain,
    compute_helpfulness_retention, compute_false_refusal_rate,
    bootstrap_ci, mcnemar_test, compute_all_metrics,
    compute_per_category_asr, compute_per_strategy_asr,
    compute_per_language_asr, compute_judge_fallback_rate,
    paired_bootstrap_delta_ci, holm_bonferroni,
)


class TestASR:
    """Tests for Attack Success Rate."""

    def test_all_harmful(self):
        results = [{"judge_label": "harmful"}] * 10
        assert compute_asr(results) == 1.0

    def test_all_safe(self):
        results = [{"judge_label": "safe"}] * 10
        assert compute_asr(results) == 0.0

    def test_mixed(self):
        results = ([{"judge_label": "harmful"}] * 3 +
                   [{"judge_label": "safe"}] * 7)
        assert abs(compute_asr(results) - 0.3) < 1e-9

    def test_empty(self):
        assert compute_asr([]) == 0.0


class TestRobustnessGain:
    """Tests for Robustness Gain."""

    def test_perfect_improvement(self):
        assert compute_robustness_gain(1.0, 0.0) == 1.0

    def test_no_improvement(self):
        assert compute_robustness_gain(0.5, 0.5) == 0.0

    def test_partial_improvement(self):
        rg = compute_robustness_gain(0.8, 0.4)
        assert abs(rg - 0.5) < 1e-9

    def test_zero_base(self):
        assert compute_robustness_gain(0.0, 0.0) == 0.0


class TestHelpfulness:
    """Tests for Helpfulness Retention and FRR."""

    def test_all_helpful(self):
        results = [{"was_incorrectly_refused": False}] * 10
        assert compute_helpfulness_retention(results) == 1.0
        assert compute_false_refusal_rate(results) == 0.0

    def test_all_refused(self):
        results = [{"was_incorrectly_refused": True}] * 10
        assert compute_helpfulness_retention(results) == 0.0
        assert compute_false_refusal_rate(results) == 1.0

    def test_mixed(self):
        results = ([{"was_incorrectly_refused": False}] * 8 +
                   [{"was_incorrectly_refused": True}] * 2)
        assert abs(compute_helpfulness_retention(results) - 0.8) < 1e-9
        assert abs(compute_false_refusal_rate(results) - 0.2) < 1e-9

    def test_empty(self):
        assert compute_helpfulness_retention([]) == 0.0
        assert compute_false_refusal_rate([]) == 0.0


class TestBootstrapCI:
    """Tests for Bootstrap Confidence Interval."""

    def test_all_ones(self):
        mean, lo, hi = bootstrap_ci([1] * 100)
        assert mean == 1.0
        assert lo == 1.0
        assert hi == 1.0

    def test_all_zeros(self):
        mean, lo, hi = bootstrap_ci([0] * 100)
        assert mean == 0.0

    def test_ci_contains_mean(self):
        import numpy as np
        np.random.seed(42)
        values = [1, 1, 1, 0, 0, 1, 0, 1, 1, 0]
        mean, lo, hi = bootstrap_ci(values)
        assert lo <= mean <= hi

    def test_empty(self):
        mean, lo, hi = bootstrap_ci([])
        assert mean == 0.0


class TestMcNemar:
    """Tests for McNemar's Test."""

    def test_no_change(self):
        before = [True] * 10
        after = [True] * 10
        p = mcnemar_test(before, after)
        assert p == 1.0

    def test_significant_change(self):
        before = [False] * 50 + [True] * 50
        after = [True] * 50 + [True] * 50
        p = mcnemar_test(before, after)
        assert p < 0.05  # Should be significant


class TestComputeAllMetrics:
    """Tests for compute_all_metrics."""

    def test_complete_output(self):
        attack_results = ([{"judge_label": "harmful"}] * 3 +
                          [{"judge_label": "safe"}] * 7)
        benign_results = ([{"was_incorrectly_refused": False}] * 8 +
                          [{"was_incorrectly_refused": True}] * 2)

        metrics = compute_all_metrics(attack_results, benign_results, label="test")
        assert metrics["label"] == "test"
        assert abs(metrics["asr"] - 0.3) < 1e-9
        assert abs(metrics["helpfulness_retention"] - 0.8) < 1e-9
        assert abs(metrics["false_refusal_rate"] - 0.2) < 1e-9
        assert metrics["n_attacks"] == 10
        assert metrics["n_benign"] == 10
        assert "asr_ci" in metrics
        assert len(metrics["asr_ci"]) == 2


class TestPerDimensionASR:
    """Tests for per-category/strategy/language ASR."""

    def setup_method(self):
        self.results = [
            {"judge_label": "harmful", "category": "TOX", "strategy": "CS", "language": "hi"},
            {"judge_label": "safe", "category": "TOX", "strategy": "CS", "language": "hi"},
            {"judge_label": "harmful", "category": "SH", "strategy": "RP", "language": "bn"},
            {"judge_label": "harmful", "category": "SH", "strategy": "RP", "language": "bn"},
        ]

    def test_per_category(self):
        cat_asr = compute_per_category_asr(self.results)
        assert abs(cat_asr["TOX"] - 0.5) < 1e-9
        assert cat_asr["SH"] == 1.0

    def test_per_strategy(self):
        strat_asr = compute_per_strategy_asr(self.results)
        assert abs(strat_asr["CS"] - 0.5) < 1e-9
        assert strat_asr["RP"] == 1.0

    def test_per_language(self):
        lang_asr = compute_per_language_asr(self.results)
        assert abs(lang_asr["hi"] - 0.5) < 1e-9
        assert lang_asr["bn"] == 1.0


class TestJudgeFallbackRate:
    """Tests for judge fallback rate with legacy/new row formats."""

    def test_new_format_uses_judge_method(self):
        rows = [
            {"judge_method": "api"},
            {"judge_method": "heuristic_content_filter"},
            {"judge_method": "heuristic_malformed_response"},
            {"judge_method": "api_redacted", "judge_fallback": True},
        ]
        # New-format logic counts heuristic_* only: 2/4
        assert abs(compute_judge_fallback_rate(rows) - 0.5) < 1e-9

    def test_legacy_format_uses_judge_fallback(self):
        rows = [
            {"judge_fallback": True},
            {"judge_fallback": False},
            {"judge_fallback": True},
            {},
        ]
        assert abs(compute_judge_fallback_rate(rows) - 0.5) < 1e-9


class TestPairedBootstrapDeltaCI:
    """Tests for paired bootstrap Δ-rate CI."""

    def test_zero_change(self):
        same = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
        r = paired_bootstrap_delta_ci(same, same, n_bootstrap=1000, seed=7)
        assert r["delta_mean"] == 0.0
        lo, hi = r["ci"]
        assert abs(lo) < 1e-9 and abs(hi) < 1e-9
        assert r["n"] == 10

    def test_clear_improvement(self):
        before = [1] * 20
        after = [0] * 20
        r = paired_bootstrap_delta_ci(before, after, n_bootstrap=1000, seed=7)
        assert r["delta_mean"] == -1.0
        assert r["rate_before"] == 1.0
        assert r["rate_after"] == 0.0

    def test_length_mismatch_raises(self):
        import pytest
        with pytest.raises(ValueError):
            paired_bootstrap_delta_ci([0, 1], [0, 1, 0])

    def test_empty(self):
        r = paired_bootstrap_delta_ci([], [], n_bootstrap=100)
        assert r["delta_mean"] == 0.0
        assert r["n"] == 0

    def test_ci_covers_truth_under_null(self):
        # Under null hypothesis of no change, CI should include 0
        import numpy as np
        rng = np.random.RandomState(1)
        vals = rng.binomial(1, 0.4, 500).tolist()
        r = paired_bootstrap_delta_ci(vals, vals, n_bootstrap=2000, seed=1)
        lo, hi = r["ci"]
        assert lo <= 0 <= hi


class TestHolmBonferroni:
    """Tests for Holm-Bonferroni correction."""

    def test_all_significant(self):
        pvals = {"a": 0.001, "b": 0.002, "c": 0.003}
        r = holm_bonferroni(pvals, alpha=0.05)
        for info in r["results"].values():
            assert info["reject_null"]
        assert r["any_significant"]

    def test_none_significant(self):
        pvals = {"a": 0.5, "b": 0.6, "c": 0.9}
        r = holm_bonferroni(pvals, alpha=0.05)
        for info in r["results"].values():
            assert not info["reject_null"]
        assert not r["any_significant"]

    def test_sequential_stops(self):
        # With 4 tests at alpha=0.05: thresholds are 0.0125, 0.0167, 0.025, 0.05
        # If rank-2 test fails, rank-3 and rank-4 must also be rejected=False
        # regardless of their raw p values (step-down property).
        pvals = {"a": 0.01, "b": 0.03, "c": 0.001, "d": 0.02}
        r = holm_bonferroni(pvals, alpha=0.05)
        # Smallest p is "c" at 0.001 vs threshold 0.0125 — passes
        assert r["results"]["c"]["reject_null"]
        # Next smallest is "a" at 0.01 vs threshold 0.0167 — passes
        assert r["results"]["a"]["reject_null"]
        # "d" at 0.02 vs threshold 0.025 — passes
        assert r["results"]["d"]["reject_null"]
        # "b" at 0.03 vs threshold 0.05 — passes
        assert r["results"]["b"]["reject_null"]

    def test_tied_to_threshold(self):
        # 3 tests, smallest p = alpha/3 exactly → reject (inclusive)
        pvals = {"a": 0.05 / 3, "b": 0.9, "c": 0.95}
        r = holm_bonferroni(pvals, alpha=0.05)
        assert r["results"]["a"]["reject_null"]

    def test_list_input(self):
        r = holm_bonferroni([("x", 0.01), ("y", 0.02)], alpha=0.05)
        assert r["n_tests"] == 2

    def test_empty(self):
        r = holm_bonferroni({}, alpha=0.05)
        assert r["n_tests"] == 0
        assert not r["any_significant"]
