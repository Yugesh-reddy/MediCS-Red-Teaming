"""Tests for medics.detection module."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.detection import perplexity_detector, detection_by_language


class TestPerplexityDetector:
    def test_perfect_separation(self):
        en_ppls = [10.0, 12.0, 8.0, 11.0, 9.0]
        cs_ppls = [100.0, 120.0, 90.0, 110.0, 95.0]
        result = perplexity_detector(en_ppls, cs_ppls)
        assert result["auroc"] >= 0.9
        assert result["best_f1"] >= 0.8

    def test_random_overlap(self):
        rng = np.random.RandomState(42)
        en_ppls = rng.normal(50, 20, 100).tolist()
        cs_ppls = rng.normal(55, 20, 100).tolist()
        result = perplexity_detector(en_ppls, cs_ppls)
        # With overlapping distributions, detection should be poor
        assert result["auroc"] < 0.8

    def test_empty_inputs(self):
        result = perplexity_detector([], [])
        assert result["auroc"] == 0.5  # default

    def test_output_structure(self):
        en_ppls = [10.0, 20.0]
        cs_ppls = [50.0, 60.0]
        result = perplexity_detector(en_ppls, cs_ppls)
        assert "best_threshold" in result
        assert "best_f1" in result
        assert "auroc" in result
        assert "per_threshold" in result
        assert "roc" in result
        assert "fpr" in result["roc"]
        assert "tpr" in result["roc"]

    def test_custom_thresholds(self):
        en_ppls = [10.0, 20.0, 15.0]
        cs_ppls = [50.0, 60.0, 55.0]
        result = perplexity_detector(en_ppls, cs_ppls, thresholds=[25.0, 35.0, 45.0])
        assert len(result["per_threshold"]) == 3


class TestDetectionByLanguage:
    def test_basic(self):
        entries = [
            {"language": "hi", "perplexity": 100.0, "is_cs": True},
            {"language": "hi", "perplexity": 110.0, "is_cs": True},
            {"language": "hi", "perplexity": 10.0, "is_cs": False},
            {"language": "hi", "perplexity": 12.0, "is_cs": False},
            {"language": "sw", "perplexity": 50.0, "is_cs": True},
            {"language": "sw", "perplexity": 55.0, "is_cs": True},
            {"language": "sw", "perplexity": 11.0, "is_cs": False},
            {"language": "sw", "perplexity": 13.0, "is_cs": False},
        ]
        result = detection_by_language(entries)
        assert "hi" in result
        assert "sw" in result
        assert result["hi"]["n_cs"] == 2
        assert result["hi"]["n_en"] == 2

    def test_empty(self):
        result = detection_by_language([])
        assert result == {}
