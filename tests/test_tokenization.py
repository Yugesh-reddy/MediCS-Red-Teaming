"""Tests for medics.tokenization module."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medics.tokenization import (
    _tokenize_and_measure,
    _keyword_fragmentation,
    compute_oov_proxy,
    compute_fragmentation_summary,
)


def _mock_tokenizer():
    """Create a mock tokenizer that simulates BPE behavior."""
    tok = MagicMock()

    def mock_encode(text, add_special_tokens=False):
        # Simple: each word becomes a token, multi-byte chars get split
        words = text.split()
        ids = list(range(len(words)))
        return ids

    def mock_convert(ids):
        # Return placeholder tokens
        return [f"tok_{i}" for i in ids]

    tok.encode = mock_encode
    tok.convert_ids_to_tokens = mock_convert
    return tok


class TestTokenizeAndMeasure:
    def test_basic(self):
        tok = _mock_tokenizer()
        result = _tokenize_and_measure(tok, "hello world test")
        assert result["token_count"] == 3
        assert result["unique_token_pct"] == 1.0

    def test_empty(self):
        tok = _mock_tokenizer()
        tok.encode = MagicMock(return_value=[])
        tok.convert_ids_to_tokens = MagicMock(return_value=[])
        result = _tokenize_and_measure(tok, "")
        assert result["token_count"] == 0
        assert result["unique_token_pct"] == 0.0


class TestKeywordFragmentation:
    def test_ratio(self):
        tok = _mock_tokenizer()
        # English: "cyanide" = 1 token, translated = "some translated word" = 3 tokens
        result = _keyword_fragmentation(tok, "cyanide", "some translated word")
        assert result["en_tokens"] == 1
        assert result["translated_tokens"] == 3
        assert result["ratio"] == 3.0

    def test_equal_length(self):
        tok = _mock_tokenizer()
        result = _keyword_fragmentation(tok, "word", "wort")
        assert result["ratio"] == 1.0


class TestComputeOovProxy:
    def test_no_oov(self):
        tok = _mock_tokenizer()
        # Normal tokens, no byte fallback
        rate = compute_oov_proxy(tok, "hello world")
        assert rate == 0.0

    def test_with_byte_fallback(self):
        tok = _mock_tokenizer()
        tok.convert_ids_to_tokens = MagicMock(
            return_value=["hello", "<0xE0>", "<0xA4>", "<0xB9>"]
        )
        tok.encode = MagicMock(return_value=[0, 1, 2, 3])
        rate = compute_oov_proxy(tok, "hello हि")
        assert rate == 0.75  # 3 of 4 are byte fallback

    def test_empty(self):
        tok = _mock_tokenizer()
        tok.encode = MagicMock(return_value=[])
        tok.convert_ids_to_tokens = MagicMock(return_value=[])
        rate = compute_oov_proxy(tok, "")
        assert rate == 0.0


class TestFragmentationSummary:
    def test_summary_aggregation(self):
        results = [
            {"language": "hi", "token_count_ratio": 2.0, "oov_proxy_rate": 0.1, "avg_kw_ratio": 3.0},
            {"language": "hi", "token_count_ratio": 3.0, "oov_proxy_rate": 0.2, "avg_kw_ratio": 4.0},
            {"language": "sw", "token_count_ratio": 1.5, "oov_proxy_rate": 0.05, "avg_kw_ratio": 2.0},
        ]
        summary = compute_fragmentation_summary(results)
        assert "hi" in summary
        assert "sw" in summary
        assert summary["hi"]["mean_ratio"] == 2.5
        assert summary["hi"]["n_seeds"] == 2
        assert summary["sw"]["n_seeds"] == 1

    def test_empty(self):
        summary = compute_fragmentation_summary([])
        assert summary == {}
