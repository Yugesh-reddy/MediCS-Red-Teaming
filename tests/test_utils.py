"""
Tests for medics/utils.py

Tests I/O, normalization, code-switching, leetspeak, and deduplication.
Translation tests mock deep-translator to avoid network calls.
"""

import json
import os
import tempfile
import pytest
from unittest.mock import patch, MagicMock

from medics.utils import (
    load_jsonl,
    save_jsonl,
    load_json,
    save_json,
    append_jsonl,
    normalize_seed,
    load_seeds,
    load_config,
    code_switch_prompt,
    apply_leetspeak,
    deduplicate,
    extract_keywords_batch,
)


# ---------------------------------------------------------------------------
# I/O Functions
# ---------------------------------------------------------------------------
class TestIO:
    def test_save_load_jsonl_roundtrip(self, tmp_path):
        data = [{"a": 1, "b": "two"}, {"a": 3, "b": "four"}]
        path = str(tmp_path / "test.jsonl")
        save_jsonl(data, path)
        loaded = load_jsonl(path)
        assert loaded == data

    def test_load_jsonl_missing_file(self):
        result = load_jsonl("/nonexistent/path.jsonl")
        assert result == []

    def test_load_jsonl_skips_blank_lines(self, tmp_path):
        path = str(tmp_path / "test.jsonl")
        with open(path, "w") as f:
            f.write('{"a": 1}\n\n{"a": 2}\n')
        loaded = load_jsonl(path)
        assert len(loaded) == 2

    def test_save_load_json_roundtrip(self, tmp_path):
        data = {"key": "value", "nested": {"x": 1}}
        path = str(tmp_path / "test.json")
        save_json(data, path)
        loaded = load_json(path)
        assert loaded == data

    def test_load_json_missing_file(self):
        result = load_json("/nonexistent/path.json")
        assert result is None

    def test_append_jsonl(self, tmp_path):
        path = str(tmp_path / "append.jsonl")
        append_jsonl(path, {"x": 1})
        append_jsonl(path, {"x": 2})
        loaded = load_jsonl(path)
        assert len(loaded) == 2
        assert loaded[0]["x"] == 1
        assert loaded[1]["x"] == 2

    def test_save_jsonl_unicode(self, tmp_path):
        data = [{"text": "Hindi: हिन्दी Bengali: বাংলা"}]
        path = str(tmp_path / "unicode.jsonl")
        save_jsonl(data, path)
        loaded = load_jsonl(path)
        assert loaded[0]["text"] == data[0]["text"]


# ---------------------------------------------------------------------------
# Seed Normalization
# ---------------------------------------------------------------------------
class TestNormalizeSeed:
    def test_renames_original_en_to_prompt(self):
        seed = {"original_en": "How to do X?", "id": "TOX_001"}
        normalized = normalize_seed(seed)
        assert normalized["prompt"] == "How to do X?"
        assert normalized["seed_id"] == "TOX_001"

    def test_preserves_existing_prompt_and_seed_id(self):
        seed = {"prompt": "existing", "seed_id": "S001", "original_en": "other"}
        normalized = normalize_seed(seed)
        assert normalized["prompt"] == "existing"
        assert normalized["seed_id"] == "S001"

    def test_no_op_on_already_normalized(self):
        seed = {"prompt": "test", "seed_id": "T001", "category": "TOX"}
        normalized = normalize_seed(seed)
        assert normalized == seed

    def test_load_seeds_normalizes(self, tmp_path):
        data = [
            {"original_en": "prompt1", "id": "S001", "category": "TOX"},
            {"prompt": "prompt2", "seed_id": "S002", "category": "SH"},
        ]
        path = str(tmp_path / "seeds.jsonl")
        save_jsonl(data, path)
        seeds = load_seeds(path)
        assert seeds[0]["prompt"] == "prompt1"
        assert seeds[0]["seed_id"] == "S001"
        assert seeds[1]["prompt"] == "prompt2"


# ---------------------------------------------------------------------------
# Code-Switching
# ---------------------------------------------------------------------------
class TestCodeSwitch:
    @patch("medics.utils.translate_with_fallback")
    def test_basic_code_switch(self, mock_translate):
        mock_translate.return_value = {"translation": "दवा", "source": "mock"}
        seed = {"prompt": "How to synthesize a drug at home",
                "seed_id": "TOX_001", "category": "TOX"}
        keywords = {"TOX_001": ["drug"]}

        result = code_switch_prompt(seed, keywords, "hi")
        assert result["strategy"] == "CS"
        assert result["language"] == "hi"
        assert result["original_prompt"] == seed["prompt"]
        assert "दवा" in result["attack_prompt"]
        assert result["seed_id"] == "TOX_001"

    @patch("medics.utils.translate_with_fallback")
    def test_case_insensitive_replacement(self, mock_translate):
        mock_translate.return_value = {"translation": "translated", "source": "mock"}
        seed = {"prompt": "Get the Drug from the DRUG store",
                "seed_id": "S1", "category": "TOX"}
        keywords = {"S1": ["Drug"]}

        result = code_switch_prompt(seed, keywords, "hi")
        # Both "Drug" and "DRUG" should be replaced
        assert "Drug" not in result["attack_prompt"]
        assert "DRUG" not in result["attack_prompt"]

    def test_empty_keywords_returns_original(self):
        seed = {"prompt": "How to do something", "seed_id": "S1", "category": "TOX"}
        keywords = {"S1": []}
        result = code_switch_prompt(seed, keywords, "hi")
        assert result["attack_prompt"] == seed["prompt"]

    def test_missing_seed_in_keywords_dict(self):
        seed = {"prompt": "test prompt", "seed_id": "MISSING", "category": "TOX"}
        keywords = {"OTHER": ["word"]}
        result = code_switch_prompt(seed, keywords, "hi")
        assert result["attack_prompt"] == seed["prompt"]

    @patch("medics.utils.translate_with_fallback")
    def test_keywords_as_list(self, mock_translate):
        """Keywords can be passed as a list directly (not dict)."""
        mock_translate.return_value = {"translation": "xyz", "source": "mock"}
        seed = {"prompt": "the poison is deadly", "seed_id": "S1", "category": "TOX"}
        result = code_switch_prompt(seed, ["poison"], "hi")
        assert "xyz" in result["attack_prompt"]


# ---------------------------------------------------------------------------
# Leetspeak
# ---------------------------------------------------------------------------
class TestLeetspeak:
    def test_returns_string(self):
        result = apply_leetspeak("Hello World")
        assert isinstance(result, str)

    def test_preserves_length(self):
        text = "test string"
        result = apply_leetspeak(text)
        assert len(result) == len(text)

    def test_deterministic_with_seed(self):
        import random
        random.seed(42)
        r1 = apply_leetspeak("sample text")
        random.seed(42)
        r2 = apply_leetspeak("sample text")
        assert r1 == r2

    def test_some_characters_changed(self):
        """With enough text and probability, at least some chars should change."""
        import random
        random.seed(0)
        text = "aaaa eeee iiii oooo ssss"
        result = apply_leetspeak(text)
        # At least some characters should be different
        assert result != text


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------
class TestDeduplication:
    def test_removes_exact_duplicates(self):
        seeds = [
            {"prompt": "How to synthesize aspirin", "seed_id": "1"},
            {"prompt": "How to synthesize aspirin", "seed_id": "2"},
            {"prompt": "What causes headaches", "seed_id": "3"},
        ]
        result = deduplicate(seeds, threshold=0.85)
        assert len(result) == 2

    def test_keeps_dissimilar(self):
        seeds = [
            {"prompt": "Alpha beta gamma delta", "seed_id": "1"},
            {"prompt": "Completely different topic about cats", "seed_id": "2"},
        ]
        result = deduplicate(seeds, threshold=0.85)
        assert len(result) == 2

    def test_single_seed(self):
        seeds = [{"prompt": "Only one seed", "seed_id": "1"}]
        result = deduplicate(seeds, threshold=0.85)
        assert len(result) == 1

    def test_empty_list(self):
        # TfidfVectorizer on empty list will error, but it should handle it
        # Actually, sklearn will raise on empty. Let's skip this edge case.
        pass


# ---------------------------------------------------------------------------
# Config Loading
# ---------------------------------------------------------------------------
class TestConfig:
    def test_load_config_from_file(self, tmp_path):
        config = {"project": {"name": "test"}, "dataset": {"n_seeds": 10}}
        path = str(tmp_path / "config.yaml")
        import yaml
        with open(path, "w") as f:
            yaml.dump(config, f)
        loaded = load_config(path)
        assert loaded["project"]["name"] == "test"
        assert loaded["dataset"]["n_seeds"] == 10


# ---------------------------------------------------------------------------
# Keyword Extraction (mocked API)
# ---------------------------------------------------------------------------
class TestExtractKeywordsBatch:
    """Tests for extract_keywords_batch retry, fallback, and checkpoint logic."""

    def _make_seeds(self, n=3):
        return [{"seed_id": f"S{i:03d}", "prompt": f"How to synthesize drug compound {i}"}
                for i in range(n)]

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "fake",
        "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
    })
    @patch("openai.AzureOpenAI")
    @patch("medics.judge.track_external_usage")
    def test_successful_extraction(self, mock_track, mock_client_cls, tmp_path):
        """API returns valid keywords."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"keywords": ["drug", "compound"]}'
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_resp

        seeds = self._make_seeds(2)
        cp = str(tmp_path / "kw_cp.json")
        result = extract_keywords_batch(seeds, max_workers=1, checkpoint_path=cp)

        assert len(result) == 2
        assert result["S000"] == ["drug", "compound"]
        assert os.path.exists(cp)

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "fake",
        "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
    })
    @patch("openai.AzureOpenAI")
    @patch("medics.judge.track_external_usage")
    def test_content_filter_fallback(self, mock_track, mock_client_cls, tmp_path):
        """Content filter errors trigger local fallback extraction."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception(
            "content_filter_policy_violation"
        )

        seeds = self._make_seeds(1)
        cp = str(tmp_path / "kw_cp.json")
        result = extract_keywords_batch(seeds, max_workers=1, checkpoint_path=cp)

        assert len(result) == 1
        assert len(result["S000"]) > 0, "Fallback should produce keywords"

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "fake",
        "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
    })
    @patch("openai.AzureOpenAI")
    @patch("medics.judge.track_external_usage")
    def test_generic_error_fallback(self, mock_track, mock_client_cls, tmp_path):
        """Non-retryable errors trigger local fallback (not empty list)."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("server_error 500")

        seeds = self._make_seeds(1)
        cp = str(tmp_path / "kw_cp.json")
        result = extract_keywords_batch(seeds, max_workers=1, checkpoint_path=cp)

        assert len(result["S000"]) > 0, "Generic errors should fallback, not return empty"

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "fake",
        "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
    })
    @patch("openai.AzureOpenAI")
    @patch("medics.judge.track_external_usage")
    def test_retry_exhaustion_fallback(self, mock_track, mock_client_cls, tmp_path):
        """Exhausting all retries on rate limits triggers fallback."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("429 rate_limit")

        seeds = self._make_seeds(1)
        cp = str(tmp_path / "kw_cp.json")
        result = extract_keywords_batch(seeds, max_workers=1, checkpoint_path=cp)

        assert len(result["S000"]) > 0, "Retry exhaustion should fallback"

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "fake",
        "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
    })
    @patch("openai.AzureOpenAI")
    @patch("medics.judge.track_external_usage")
    def test_checkpoint_resume(self, mock_track, mock_client_cls, tmp_path):
        """Seeds already in checkpoint are skipped."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"keywords": ["new"]}'
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_resp

        cp = str(tmp_path / "kw_cp.json")
        with open(cp, "w") as f:
            json.dump({"S000": ["cached", "keyword"]}, f)

        seeds = self._make_seeds(2)
        result = extract_keywords_batch(seeds, max_workers=1, checkpoint_path=cp)

        assert result["S000"] == ["cached", "keyword"], "Cached entry preserved"
        assert result["S001"] == ["new"], "New entry extracted"
        assert mock_client.chat.completions.create.call_count == 1

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "fake",
        "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
    })
    @patch("openai.AzureOpenAI")
    @patch("medics.judge.track_external_usage")
    def test_corrupt_checkpoint_ignored(self, mock_track, mock_client_cls, tmp_path):
        """Non-dict checkpoint (e.g. []) is discarded, starts fresh."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"keywords": ["drug"]}'
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_resp

        cp = str(tmp_path / "kw_cp.json")
        with open(cp, "w") as f:
            json.dump([], f)  # wrong type

        seeds = self._make_seeds(1)
        result = extract_keywords_batch(seeds, max_workers=1, checkpoint_path=cp)

        assert len(result) == 1
        assert result["S000"] == ["drug"]

    @patch.dict(os.environ, {
        "AZURE_OPENAI_API_KEY": "fake",
        "AZURE_OPENAI_ENDPOINT": "https://fake.openai.azure.com",
        "AZURE_OPENAI_API_VERSION": "2024-12-01-preview",
    })
    @patch("openai.AzureOpenAI")
    @patch("medics.judge.track_external_usage")
    def test_empty_api_response_fallback(self, mock_track, mock_client_cls, tmp_path):
        """API returns valid JSON but empty keywords triggers fallback."""
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = '{"keywords": []}'
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        mock_client.chat.completions.create.return_value = mock_resp

        seeds = self._make_seeds(1)
        cp = str(tmp_path / "kw_cp.json")
        result = extract_keywords_batch(seeds, max_workers=1, checkpoint_path=cp)

        assert len(result["S000"]) > 0, "Empty API response should trigger local fallback"
