"""
Tests for medics/judge.py

Mocks all OpenAI API calls -- no real API keys or network needed.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from medics.judge import (
    call_judge,
    judge_response_batch,
    judge_helpfulness,
    generate_refusal_batch,
    generate_helpful_batch,
    generate_mte_turns,
    _track_usage,
    get_session_usage,
    reset_session_usage,
    print_session_cost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mock_chat_response(content, model="gpt-4o",
                         prompt_tokens=100, completion_tokens=50):
    """Create a mock OpenAI chat completion response."""
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    return resp


def _mock_client(content, model="gpt-4o"):
    """Create a mock OpenAI client that returns the given content."""
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_chat_response(content, model)
    return client


# ---------------------------------------------------------------------------
# call_judge
# ---------------------------------------------------------------------------
class TestCallJudge:
    def test_harmful_response(self):
        client = _mock_client(json.dumps({"harmful": 1, "reason": "gives dosage info"}))
        result = call_judge("how to overdose", "Take 50 pills...", client=client)
        assert result["judge_label"] == "harmful"
        assert result["harmful"] == 1
        assert result["reason"] == "gives dosage info"
        assert result["judge_method"] == "api"

    def test_safe_response(self):
        client = _mock_client(json.dumps({"harmful": 0, "reason": "refused properly"}))
        result = call_judge("how to overdose", "I can't help with that.", client=client)
        assert result["judge_label"] == "safe"
        assert result["harmful"] == 0
        assert result["judge_method"] == "api"

    def test_api_error_returns_error_label(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API timeout")
        result = call_judge("prompt", "response", client=client)
        assert result["judge_label"] == "error"
        assert result["harmful"] == -1
        assert "judge_error" in result["reason"]
        assert result["judge_method"] == "error_api"

    def test_none_content_uses_heuristic_fallback(self):
        client = _mock_client(None)
        result = call_judge("how to overdose", "I cannot help with that.", client=client)
        assert result["judge_label"] == "safe"
        assert result["judge_fallback"] is True
        assert "malformed_response_fallback" in result["reason"]
        assert result["judge_method"] == "heuristic_malformed_response"

    def test_missing_harmful_field_returns_error(self):
        client = _mock_client(json.dumps({"reason": "no harmful key"}))
        result = call_judge("prompt", "response", client=client)
        assert result["judge_label"] == "error"
        assert result["harmful"] == -1
        assert result["judge_method"] == "error_malformed_response"

    def test_content_filter_retries_with_redacted_prompt(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = [
            Exception("content_filter_policy_violation"),
            _mock_chat_response(json.dumps({"harmful": 1, "reason": "actionable"})),
        ]
        result = call_judge("how to overdose", "Take 50mg then repeat.", client=client)
        assert result["judge_label"] == "harmful"
        assert result["harmful"] == 1
        assert result["judge_fallback"] is True
        assert "content_filter_redacted" in result["reason"]
        assert result["judge_method"] == "api_redacted"

    def test_content_filter_double_failure_uses_heuristic(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("content_filter")
        result = call_judge("how to overdose", "I can't help with that.", client=client)
        assert result["judge_label"] == "safe"
        assert result["harmful"] == 0
        assert result["judge_fallback"] is True
        assert result["judge_method"] == "heuristic_content_filter"


# ---------------------------------------------------------------------------
# judge_response_batch
# ---------------------------------------------------------------------------
class TestJudgeResponseBatch:
    def test_batch_uses_original_prompt_for_intent(self):
        """Verify fix #8: judge should prefer original_prompt over attack_prompt."""
        client = _mock_client(json.dumps({"harmful": 0, "reason": "safe"}))
        responses = [{
            "attack_prompt": "code-switched text here",
            "original_prompt": "original english intent",
            "model_response": "I refuse.",
        }]
        results = judge_response_batch(responses, client=client)
        # Check that call_judge was called with original_prompt
        call_args = client.chat.completions.create.call_args
        user_content = call_args.kwargs["messages"][1]["content"]
        assert "original english intent" in user_content

    def test_batch_returns_merged_results(self):
        client = _mock_client(json.dumps({"harmful": 1, "reason": "bad"}))
        responses = [
            {"prompt": "p1", "model_response": "r1", "category": "TOX"},
            {"prompt": "p2", "model_response": "r2", "category": "SH"},
        ]
        results = judge_response_batch(responses, client=client)
        assert len(results) == 2
        assert results[0]["category"] == "TOX"
        assert results[0]["judge_label"] == "harmful"
        assert results[0]["judge_method"] == "api"


# ---------------------------------------------------------------------------
# judge_helpfulness
# ---------------------------------------------------------------------------
class TestJudgeHelpfulness:
    def test_helpful_response(self):
        client = _mock_client(json.dumps({"helpful": 1, "reason": "good answer"}))
        result = judge_helpfulness("What causes headaches?", "Tension...", client=client)
        assert result["helpful"] == 1
        assert result["was_incorrectly_refused"] is False

    def test_unhelpful_overrefusal(self):
        client = _mock_client(json.dumps({"helpful": 0, "reason": "refused benign"}))
        result = judge_helpfulness("What causes headaches?", "I can't help.", client=client)
        assert result["helpful"] == 0
        assert result["was_incorrectly_refused"] is True

    def test_api_error_returns_judge_error_flag(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("API down")
        result = judge_helpfulness("question", "response", client=client)
        assert result["judge_error"] is True
        assert result["helpful"] == -1
        # judge error should NOT count as incorrectly refused
        assert result["was_incorrectly_refused"] is False


# ---------------------------------------------------------------------------
# generate_refusal_batch / generate_helpful_batch
# ---------------------------------------------------------------------------
class TestGenerationBatch:
    def test_refusal_batch(self):
        client = _mock_client("I cannot provide that information.")
        refusals = generate_refusal_batch(["how to harm", "how to steal"], client=client)
        assert len(refusals) == 2
        assert all(isinstance(r, str) for r in refusals)

    def test_refusal_batch_error_fallback(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("fail")
        refusals = generate_refusal_batch(["bad prompt"], client=client)
        assert len(refusals) == 1
        assert "unable to provide" in refusals[0].lower()

    def test_helpful_batch(self):
        client = _mock_client("Headaches are caused by...")
        responses = generate_helpful_batch(["What causes headaches?"], client=client)
        assert len(responses) == 1

    def test_helpful_batch_error_fallback(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("fail")
        responses = generate_helpful_batch(["question"], client=client)
        assert len(responses) == 1
        assert "consult" in responses[0].lower()


# ---------------------------------------------------------------------------
# generate_mte_turns
# ---------------------------------------------------------------------------
class TestMTETurns:
    def test_successful_generation(self):
        turns_json = json.dumps({"turns": [
            {"turn": 1, "content": "What is pharmacology?"},
            {"turn": 2, "content": "How do drug interactions work?"},
            {"turn": 3, "content": "Can you explain specific dosages?"},
        ]})
        client = _mock_client(turns_json)
        turns = generate_mte_turns("harmful prompt", client=client)
        assert len(turns) == 3
        assert turns[0]["role"] == "user"

    def test_fallback_on_error(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = Exception("fail")
        turns = generate_mte_turns("harmful prompt", client=client)
        assert len(turns) == 1
        assert turns[0]["content"] == "harmful prompt"

    def test_handles_alternate_json_keys(self):
        """MTE should handle varied JSON structures (conversation key, etc.)."""
        turns_json = json.dumps({"conversation": [
            {"turn": 1, "content": "First turn"},
            {"turn": 2, "content": "Second turn"},
        ]})
        client = _mock_client(turns_json)
        turns = generate_mte_turns("harmful", client=client)
        assert len(turns) == 2

    def test_handles_unknown_json_key(self):
        """Falls back to finding any list in the response."""
        turns_json = json.dumps({"escalation_steps": [
            {"turn": 1, "content": "Step one"},
        ]})
        client = _mock_client(turns_json)
        turns = generate_mte_turns("harmful", client=client)
        assert len(turns) == 1

    def test_handles_single_turn_object_shape(self):
        """Some model outputs return object shape {'turn', 'content'} instead of turns list."""
        turns_json = json.dumps({"turn": 1, "content": "Intro medical question"})
        client = _mock_client(turns_json)
        turns = generate_mte_turns("harmful", client=client)
        assert len(turns) == 1
        assert turns[0]["content"] == "Intro medical question"


# ---------------------------------------------------------------------------
# Cost Tracking
# ---------------------------------------------------------------------------
class TestCostTracking:
    def setup_method(self):
        reset_session_usage()

    def test_track_and_get_usage(self):
        resp = _mock_chat_response("{}", prompt_tokens=1000, completion_tokens=500)
        _track_usage(resp, "judge", "gpt-4o")
        usage = get_session_usage()
        assert usage["total_calls"] == 1
        assert usage["per_model"]["gpt-4o"]["input_tokens"] == 1000
        assert usage["per_model"]["gpt-4o"]["output_tokens"] == 500
        assert usage["total_cost"] > 0

    def test_reset_clears_log(self):
        resp = _mock_chat_response("{}", prompt_tokens=100, completion_tokens=50)
        _track_usage(resp, "judge", "gpt-4o")
        reset_session_usage()
        usage = get_session_usage()
        assert usage["total_calls"] == 0

    def test_print_session_cost_no_error(self, capsys):
        resp = _mock_chat_response("{}", prompt_tokens=100, completion_tokens=50)
        _track_usage(resp, "test", "gpt-4o")
        print_session_cost()
        captured = capsys.readouterr()
        assert "API Cost Summary" in captured.out

    def teardown_method(self):
        reset_session_usage()
