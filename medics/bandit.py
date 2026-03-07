"""
MediCS — Thompson Sampling (Beta-Bernoulli Bandit)
===================================================
Per-category bandit for attack strategy selection.
Supports save/load for checkpointing across rounds.
"""

import json
import os
from datetime import datetime

import numpy as np


class ThompsonBandit:
    """
    Beta-Bernoulli Thompson Sampling for attack strategy selection.
    
    Supports per-category arm tracking: each (category, strategy) pair
    has its own Beta posterior, so the bandit can learn that e.g.
    code-switching works better on TOX but roleplay works better on MIS.
    
    Args:
        arms: list of strategy names (e.g., ["CS", "RP", "MTE", "CS-RP", "CS-OBF"])
        categories: list of category codes (e.g., ["TOX", "SH", ...])
        prior_alpha: Beta prior alpha (default 1.0 = uniform)
        prior_beta: Beta prior beta (default 1.0 = uniform)
    """

    def __init__(self, arms=None, categories=None,
                 prior_alpha=1.0, prior_beta=1.0):
        if arms is None:
            arms = ["CS", "RP", "MTE", "CS-RP", "CS-OBF"]
        if categories is None:
            categories = ["TOX", "SH", "MIS", "ULP", "PPV", "UCA"]

        self.arms = list(arms)
        self.categories = list(categories)
        self.n_arms = len(self.arms)

        # Per-category posteriors: {category: {"alpha": [...], "beta": [...]}}
        self.posteriors = {}
        for cat in self.categories:
            self.posteriors[cat] = {
                "alpha": np.full(self.n_arms, prior_alpha),
                "beta": np.full(self.n_arms, prior_beta),
            }

        # Global posteriors (fallback if category not specified)
        self.global_alpha = np.full(self.n_arms, prior_alpha)
        self.global_beta = np.full(self.n_arms, prior_beta)

        self.history = []

    def select(self, category=None) -> str:
        """
        Sample from Beta posteriors, return strategy name with highest sample.

        Args:
            category: optional category code for per-category selection

        Returns:
            str: selected strategy name (e.g., "CS", "RP")
        """
        if category and category in self.posteriors:
            alpha = self.posteriors[category]["alpha"]
            beta = self.posteriors[category]["beta"]
        else:
            alpha = self.global_alpha
            beta = self.global_beta

        samples = np.array([
            np.random.beta(alpha[i], beta[i])
            for i in range(self.n_arms)
        ])
        arm_idx = int(np.argmax(samples))
        return self.arms[arm_idx]

    def select_with_exploration(self, category=None, min_pulls=10) -> str:
        """Ensure minimum exploration before pure Thompson Sampling."""
        counts = self._get_pull_counts(category)
        under_explored = np.where(counts < min_pulls)[0]
        if len(under_explored) > 0:
            return self.arms[int(np.random.choice(under_explored))]
        return self.select(category)

    def update(self, strategy: str, reward: float, category=None):
        """
        Update posterior for a strategy.

        Args:
            strategy: strategy name (e.g., "CS")
            reward: 1.0 for successful attack, 0.0 for failure
            category: optional category code
        """
        if strategy not in self.arms:
            raise ValueError(f"Unknown strategy: {strategy}. Known: {self.arms}")

        arm_idx = self.arms.index(strategy)

        # Update global
        if reward > 0.5:
            self.global_alpha[arm_idx] += 1.0
        else:
            self.global_beta[arm_idx] += 1.0

        # Update per-category
        if category and category in self.posteriors:
            if reward > 0.5:
                self.posteriors[category]["alpha"][arm_idx] += 1.0
            else:
                self.posteriors[category]["beta"][arm_idx] += 1.0

        self.history.append({
            "strategy": strategy,
            "arm_idx": arm_idx,
            "reward": reward,
            "category": category,
            "timestamp": datetime.now().isoformat(),
        })

    def expand_arms(self, new_arms, prior_alpha=1.0, prior_beta=1.0):
        """
        Add new strategy arms while preserving existing posteriors and history.

        Args:
            new_arms: list of new strategy names to add
            prior_alpha: Beta prior alpha for new arms
            prior_beta: Beta prior beta for new arms
        """
        for arm in new_arms:
            if arm in self.arms:
                continue
            self.arms.append(arm)
            self.global_alpha = np.append(self.global_alpha, prior_alpha)
            self.global_beta = np.append(self.global_beta, prior_beta)
            for cat in self.categories:
                self.posteriors[cat]["alpha"] = np.append(
                    self.posteriors[cat]["alpha"], prior_alpha
                )
                self.posteriors[cat]["beta"] = np.append(
                    self.posteriors[cat]["beta"], prior_beta
                )
        self.n_arms = len(self.arms)

    def get_estimated_rates(self, category=None) -> dict:
        """Return posterior mean ASR estimate for each strategy."""
        if category and category in self.posteriors:
            alpha = self.posteriors[category]["alpha"]
            beta = self.posteriors[category]["beta"]
        else:
            alpha = self.global_alpha
            beta = self.global_beta

        rates = alpha / (alpha + beta)
        return {arm: float(rates[i]) for i, arm in enumerate(self.arms)}

    def _get_pull_counts(self, category=None) -> np.ndarray:
        """Return number of pulls per arm."""
        counts = np.zeros(self.n_arms, dtype=int)
        for h in self.history:
            if category is None or h.get("category") == category:
                counts[h["arm_idx"]] += 1
        return counts

    def get_pull_counts(self, category=None) -> dict:
        """Return pull counts as a dict."""
        counts = self._get_pull_counts(category)
        return {arm: int(counts[i]) for i, arm in enumerate(self.arms)}

    def save(self, path):
        """Save bandit state to JSON."""
        path = str(path)
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        state = {
            "arms": self.arms,
            "categories": self.categories,
            "global_alpha": self.global_alpha.tolist(),
            "global_beta": self.global_beta.tolist(),
            "posteriors": {
                cat: {
                    "alpha": self.posteriors[cat]["alpha"].tolist(),
                    "beta": self.posteriors[cat]["beta"].tolist(),
                }
                for cat in self.categories
            },
            "history": self.history,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path) -> "ThompsonBandit":
        """Load bandit state from JSON."""
        path = str(path)
        with open(path, "r") as f:
            state = json.load(f)

        bandit = cls(
            arms=state["arms"],
            categories=state.get("categories", ["TOX", "SH", "MIS", "ULP", "PPV", "UCA"]),
        )
        bandit.global_alpha = np.array(state["global_alpha"])
        bandit.global_beta = np.array(state["global_beta"])

        if "posteriors" in state:
            for cat in bandit.categories:
                if cat in state["posteriors"]:
                    bandit.posteriors[cat]["alpha"] = np.array(state["posteriors"][cat]["alpha"])
                    bandit.posteriors[cat]["beta"] = np.array(state["posteriors"][cat]["beta"])

        bandit.history = state.get("history", [])
        return bandit

    def __repr__(self):
        counts = self.get_pull_counts()
        rates = self.get_estimated_rates()
        lines = [f"ThompsonBandit(arms={self.arms})"]
        for arm in self.arms:
            lines.append(f"  {arm}: pulls={counts[arm]}, est_rate={rates[arm]:.3f}")
        return "\n".join(lines)
