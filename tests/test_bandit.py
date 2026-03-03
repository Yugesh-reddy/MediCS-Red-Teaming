"""
Tests for medics.bandit — Thompson Sampling
"""

import json
import os
import tempfile
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.bandit import ThompsonBandit


class TestThompsonBandit:
    """Test suite for ThompsonBandit."""

    def test_initialization(self):
        """Test default initialization."""
        bandit = ThompsonBandit()
        assert bandit.n_arms == 5
        assert len(bandit.arms) == 5
        assert "CS" in bandit.arms
        assert len(bandit.history) == 0

    def test_custom_arms(self):
        """Test initialization with custom arms."""
        bandit = ThompsonBandit(arms=["CS", "RP"], categories=["TOX", "SH"])
        assert bandit.n_arms == 2
        assert bandit.arms == ["CS", "RP"]
        assert len(bandit.categories) == 2

    def test_select_returns_valid_strategy(self):
        """Test that select returns a valid strategy name."""
        bandit = ThompsonBandit()
        strategy = bandit.select()
        assert strategy in bandit.arms

    def test_select_with_category(self):
        """Test per-category selection."""
        bandit = ThompsonBandit()
        strategy = bandit.select(category="TOX")
        assert strategy in bandit.arms

    def test_update(self):
        """Test update modifies posteriors."""
        bandit = ThompsonBandit()
        initial_alpha = bandit.global_alpha.copy()

        bandit.update("CS", reward=1.0, category="TOX")
        assert bandit.global_alpha[0] > initial_alpha[0]
        assert len(bandit.history) == 1

    def test_update_failure(self):
        """Test update with failure modifies beta."""
        bandit = ThompsonBandit()
        initial_beta = bandit.global_beta.copy()

        bandit.update("CS", reward=0.0, category="TOX")
        assert bandit.global_beta[0] > initial_beta[0]

    def test_update_invalid_strategy(self):
        """Test update with invalid strategy raises error."""
        bandit = ThompsonBandit()
        with pytest.raises(ValueError):
            bandit.update("INVALID", reward=1.0)

    def test_get_estimated_rates(self):
        """Test estimated rates are valid probabilities."""
        bandit = ThompsonBandit()
        bandit.update("CS", 1.0)
        bandit.update("CS", 1.0)
        bandit.update("RP", 0.0)

        rates = bandit.get_estimated_rates()
        assert all(0.0 <= r <= 1.0 for r in rates.values())
        assert rates["CS"] > rates["RP"]  # CS should have higher rate

    def test_get_pull_counts(self):
        """Test pull count tracking."""
        bandit = ThompsonBandit()
        bandit.update("CS", 1.0)
        bandit.update("CS", 0.0)
        bandit.update("RP", 1.0)

        counts = bandit.get_pull_counts()
        assert counts["CS"] == 2
        assert counts["RP"] == 1
        assert counts["MTE"] == 0

    def test_per_category_counts(self):
        """Test per-category pull counts."""
        bandit = ThompsonBandit()
        bandit.update("CS", 1.0, category="TOX")
        bandit.update("CS", 0.0, category="SH")
        bandit.update("RP", 1.0, category="TOX")

        tox_counts = bandit.get_pull_counts(category="TOX")
        assert tox_counts["CS"] == 1
        assert tox_counts["RP"] == 1

    def test_exploration(self):
        """Test exploration ensures minimum pulls."""
        bandit = ThompsonBandit(arms=["CS", "RP"])
        # With min_pulls=10, should explore under-explored arms
        strategy = bandit.select_with_exploration(min_pulls=10)
        assert strategy in bandit.arms

    def test_save_load(self):
        """Test save and load roundtrip."""
        bandit = ThompsonBandit()
        bandit.update("CS", 1.0, category="TOX")
        bandit.update("RP", 0.0, category="SH")
        bandit.update("MTE", 1.0)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name

        try:
            bandit.save(temp_path)

            loaded = ThompsonBandit.load(temp_path)
            assert loaded.n_arms == bandit.n_arms
            assert loaded.arms == bandit.arms
            assert len(loaded.history) == 3
            np.testing.assert_array_equal(loaded.global_alpha, bandit.global_alpha)
            np.testing.assert_array_equal(loaded.global_beta, bandit.global_beta)
        finally:
            os.unlink(temp_path)

    def test_repr(self):
        """Test string representation."""
        bandit = ThompsonBandit(arms=["CS", "RP"])
        bandit.update("CS", 1.0)
        s = repr(bandit)
        assert "ThompsonBandit" in s
        assert "CS" in s
