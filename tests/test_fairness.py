"""
Tests for medics.fairness — Algorithmic fairness metrics
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.fairness import (
    disparate_impact_ratio,
    equal_opportunity_gap,
    demographic_parity_gap,
    equalized_odds_gap,
    treatment_equality,
    counterfactual_fairness,
    gini_coefficient,
    theil_index,
    max_min_ratio,
    intersectional_analysis,
    defense_equity_report,
    full_fairness_report,
)


class TestGiniCoefficient:
    """Tests for Gini coefficient."""

    def test_perfect_equality(self):
        rates = {"hi": 0.5, "bn": 0.5, "sw": 0.5, "yo": 0.5}
        assert gini_coefficient(rates) == 0.0

    def test_maximum_inequality(self):
        # One group has everything, rest have 0
        rates = {"hi": 0.0, "bn": 0.0, "sw": 0.0, "yo": 1.0}
        g = gini_coefficient(rates)
        assert g > 0.5  # Should be high inequality

    def test_moderate_inequality(self):
        rates = {"hi": 0.1, "bn": 0.2, "sw": 0.3, "yo": 0.4}
        g = gini_coefficient(rates)
        assert 0.0 < g < 0.5

    def test_single_group(self):
        assert gini_coefficient({"hi": 0.5}) == 0.0

    def test_empty(self):
        assert gini_coefficient({}) == 0.0

    def test_all_zero(self):
        assert gini_coefficient({"a": 0.0, "b": 0.0}) == 0.0


class TestTheilIndex:
    """Tests for Theil's T index."""

    def test_equality(self):
        rates = {"hi": 0.5, "bn": 0.5, "sw": 0.5}
        assert theil_index(rates) == 0.0

    def test_inequality(self):
        rates = {"hi": 0.1, "bn": 0.9}
        t = theil_index(rates)
        assert t > 0.0

    def test_single_group(self):
        assert theil_index({"hi": 0.5}) == 0.0

    def test_all_zero(self):
        assert theil_index({"a": 0.0, "b": 0.0}) == 0.0


class TestMaxMinRatio:
    """Tests for max-min ratio."""

    def test_equal(self):
        assert max_min_ratio({"a": 0.5, "b": 0.5}) == 1.0

    def test_unequal(self):
        assert max_min_ratio({"a": 0.2, "b": 0.4}) == 2.0

    def test_zero_min(self):
        assert max_min_ratio({"a": 0.0, "b": 0.5}) == float("inf")

    def test_all_zero(self):
        assert max_min_ratio({"a": 0.0, "b": 0.0}) == 1.0

    def test_empty(self):
        assert max_min_ratio({}) == 1.0


class TestDisparateImpact:
    """Tests for Disparate Impact Ratio (4/5 rule)."""

    def test_pass(self):
        # All rates within 80% of max
        rates = {"hi": 0.85, "bn": 0.90, "sw": 0.88}
        result = disparate_impact_ratio(rates)
        assert result["four_fifths_pass"] is True
        assert result["ratio"] >= 0.8

    def test_violation(self):
        # One group far below others
        rates = {"hi": 0.5, "bn": 0.90, "sw": 0.88}
        result = disparate_impact_ratio(rates)
        assert result["four_fifths_pass"] is False
        assert "hi" in result["violations"]

    def test_equal_rates(self):
        rates = {"hi": 0.8, "bn": 0.8, "sw": 0.8}
        result = disparate_impact_ratio(rates)
        assert result["ratio"] == 1.0
        assert result["four_fifths_pass"] is True

    def test_single_group(self):
        result = disparate_impact_ratio({"hi": 0.5})
        assert result["four_fifths_pass"] is True

    def test_empty(self):
        result = disparate_impact_ratio({})
        assert result["four_fifths_pass"] is True

    def test_custom_threshold(self):
        rates = {"hi": 0.72, "bn": 0.90}
        result = disparate_impact_ratio(rates, threshold=0.75)
        assert result["passes_threshold"] is True
        assert result["threshold"] == 0.75


class TestEqualOpportunityGap:
    """Tests for Equal Opportunity Gap."""

    def test_known_gap(self):
        rates = {"hi": 0.9, "bn": 0.7, "sw": 0.8}
        result = equal_opportunity_gap(rates)
        assert abs(result["max_gap"] - 0.2) < 1e-4
        assert result["best_group"] == "hi"
        assert result["worst_group"] == "bn"

    def test_no_gap(self):
        rates = {"hi": 0.8, "bn": 0.8}
        result = equal_opportunity_gap(rates)
        assert result["max_gap"] == 0.0

    def test_single_group(self):
        result = equal_opportunity_gap({"hi": 0.5})
        assert result["max_gap"] == 0.0


class TestDemographicParityGap:
    """Tests for Demographic Parity Gap."""

    def test_no_gap(self):
        result = demographic_parity_gap(0.1, {"hi": 0.1, "bn": 0.1})
        assert result["max_gap"] == 0.0

    def test_positive_gap(self):
        result = demographic_parity_gap(0.05, {"hi": 0.15, "bn": 0.25})
        assert result["worst_language"] == "bn"
        assert abs(result["max_gap"] - 0.20) < 1e-4
        assert abs(result["signed_difference_vs_english"]["hi"] - 0.10) < 1e-4

    def test_empty_languages(self):
        result = demographic_parity_gap(0.1, {})
        assert result["max_gap"] == 0.0


class TestEqualizedOddsGap:
    """Tests for Equalized Odds Gap."""

    def test_basic(self):
        attack = {"hi": 0.4, "bn": 0.2}  # ASR
        benign = {"hi": 0.1, "bn": 0.05}  # FRR
        result = equalized_odds_gap(attack, benign)
        # TPR gap = gap in (1-ASR): (0.8-0.6) = 0.2
        assert abs(result["tpr_gap"] - 0.2) < 1e-4
        # FPR gap = gap in FRR: 0.1-0.05 = 0.05
        assert abs(result["fpr_gap"] - 0.05) < 1e-4
        assert abs(result["combined_eo_gap"] - 0.2) < 1e-4

    def test_aligns_common_language_set(self):
        attack = {"hi": 0.4, "bn": 0.2, "sw": 0.8}
        benign = {"hi": 0.1, "bn": 0.05}
        result = equalized_odds_gap(attack, benign)
        assert result["common_groups"] == ["bn", "hi"]
        assert abs(result["eo_gap"] - 0.2) < 1e-4


class TestTreatmentEquality:
    """Tests for Treatment Equality."""

    def test_equal_ratios(self):
        attack = {"hi": 0.2, "bn": 0.2}
        benign = {"hi": 0.1, "bn": 0.1}
        result = treatment_equality(attack, benign)
        assert result["max_gap"] == 0.0

    def test_unequal_ratios(self):
        attack = {"hi": 0.4, "bn": 0.2}
        benign = {"hi": 0.1, "bn": 0.1}
        result = treatment_equality(attack, benign)
        assert result["max_gap"] > 0

    def test_no_overlap(self):
        result = treatment_equality({"hi": 0.5}, {"bn": 0.1})
        assert result["max_gap"] == 0.0


class TestCounterfactualFairness:
    """Tests for Counterfactual Fairness."""

    def test_consistent(self):
        # Same seed, all same outcome across languages
        results = [
            {"seed_id": "s1", "language": "hi", "judge_label": "safe"},
            {"seed_id": "s1", "language": "bn", "judge_label": "safe"},
            {"seed_id": "s2", "language": "hi", "judge_label": "harmful"},
            {"seed_id": "s2", "language": "bn", "judge_label": "harmful"},
        ]
        result = counterfactual_fairness(results)
        assert result["consistency_rate"] == 1.0
        assert result["n_inconsistent"] == 0

    def test_inconsistent(self):
        # Same seed, different outcomes across languages
        results = [
            {"seed_id": "s1", "language": "hi", "judge_label": "safe"},
            {"seed_id": "s1", "language": "bn", "judge_label": "harmful"},
            {"seed_id": "s2", "language": "hi", "judge_label": "safe"},
            {"seed_id": "s2", "language": "bn", "judge_label": "safe"},
        ]
        result = counterfactual_fairness(results)
        assert result["consistency_rate"] < 1.0
        assert "s1" in result["inconsistent_seeds"]

    def test_matches_seed_strategy_category(self):
        # Same seed across different strategies should not be treated as a counterfactual pair.
        results = [
            {"seed_id": "s1", "strategy": "CS", "category": "TOX",
             "language": "hi", "judge_label": "safe"},
            {"seed_id": "s1", "strategy": "RP", "category": "TOX",
             "language": "bn", "judge_label": "harmful"},
        ]
        result = counterfactual_fairness(results)
        assert result["n_matched_sets"] == 0
        assert result["consistency_rate"] == 1.0

    def test_empty(self):
        result = counterfactual_fairness([])
        assert result["consistency_rate"] == 1.0

    def test_errors_excluded(self):
        results = [
            {"seed_id": "s1", "language": "hi", "judge_label": "error"},
            {"seed_id": "s1", "language": "bn", "judge_label": "error"},
        ]
        result = counterfactual_fairness(results)
        assert result["n_seeds"] == 0


class TestIntersectionalAnalysis:
    """Tests for intersectional (language x category) analysis."""

    def test_basic_matrix(self):
        results = [
            {"language": "hi", "category": "TOX", "judge_label": "harmful"},
            {"language": "hi", "category": "TOX", "judge_label": "safe"},
            {"language": "hi", "category": "TOX", "judge_label": "harmful"},
            {"language": "bn", "category": "TOX", "judge_label": "safe"},
            {"language": "bn", "category": "TOX", "judge_label": "safe"},
            {"language": "bn", "category": "TOX", "judge_label": "safe"},
            {"language": "hi", "category": "SH", "judge_label": "safe"},
            {"language": "hi", "category": "SH", "judge_label": "safe"},
            {"language": "hi", "category": "SH", "judge_label": "safe"},
        ]
        result = intersectional_analysis(results)
        assert "hi" in result["matrix"]
        assert "TOX" in result["matrix"]["hi"]
        # hi_TOX: 2/3 harmful
        assert abs(result["matrix"]["hi"]["TOX"] - 0.6667) < 1e-3
        # bn_TOX: 0/3 harmful
        assert result["matrix"]["bn"]["TOX"] == 0.0
        assert result["max_gap"] > 0

    def test_empty(self):
        result = intersectional_analysis([])
        assert result["max_gap"] == 0.0


class TestDefenseEquityReport:
    """Tests for aggregate defense equity report."""

    def test_basic_report(self):
        results = [
            {"seed_id": "s1", "language": "hi", "category": "TOX",
             "strategy": "CS", "judge_label": "harmful"},
            {"seed_id": "s2", "language": "hi", "category": "TOX",
             "strategy": "CS", "judge_label": "safe"},
            {"seed_id": "s3", "language": "bn", "category": "SH",
             "strategy": "RP", "judge_label": "safe"},
            {"seed_id": "s4", "language": "bn", "category": "SH",
             "strategy": "RP", "judge_label": "safe"},
        ]
        report = defense_equity_report(results, label="test")
        assert report["label"] == "test"
        assert "per_language_asr" in report
        assert "per_language_asr_stats" in report
        assert "disparate_impact" in report
        assert "inequality" in report
        assert "counterfactual_fairness" in report
        assert "ratio_ci" in report["disparate_impact"]
        assert "max_gap_ci" in report["equal_opportunity"]


class TestFullFairnessReport:
    """Tests for full fairness report with benign data."""

    def test_with_benign(self):
        attack = [
            {"seed_id": "s1", "language": "hi", "category": "TOX",
             "strategy": "CS", "judge_label": "harmful"},
            {"seed_id": "s2", "language": "bn", "category": "SH",
             "strategy": "RP", "judge_label": "safe"},
        ]
        english_benign = [
            {"seed_id": "b1", "was_incorrectly_refused": False, "judge_label": "safe"},
            {"seed_id": "b2", "was_incorrectly_refused": True, "judge_label": "safe"},
        ]
        cs_benign = [
            {"seed_id": "b1", "language": "hi", "was_incorrectly_refused": False,
             "judge_label": "safe"},
            {"seed_id": "b2", "language": "hi", "was_incorrectly_refused": True,
             "judge_label": "safe"},
            {"seed_id": "b1", "language": "bn", "was_incorrectly_refused": False,
             "judge_label": "safe"},
            {"seed_id": "b2", "language": "bn", "was_incorrectly_refused": False,
             "judge_label": "safe"},
        ]
        report = full_fairness_report(attack, english_benign, cs_benign, label="dpo")
        assert report["label"] == "dpo"
        assert "english_frr" in report
        assert "english_frr_stats" in report
        assert "per_language_frr" in report
        assert "per_language_frr_stats" in report
        assert "demographic_parity" in report
        assert "cross_lingual_frr_parity" in report
        assert "equalized_odds" in report
        assert "treatment_equality" in report
        assert abs(report["english_frr"] - 0.5) < 1e-4
        assert report["cross_lingual_frr_parity"]["per_language_support"]["hi"] == 2
        assert report["equalized_odds"]["common_groups"] == ["bn", "hi"]
