"""
Tests for colab/run_inference.py resume + partial-save behavior.
"""

import importlib.util
from pathlib import Path

from medics.utils import load_jsonl


MODULE_PATH = Path(__file__).resolve().parent.parent / "colab" / "run_inference.py"
SPEC = importlib.util.spec_from_file_location("run_inference_module", MODULE_PATH)
RUN_INFERENCE_MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_INFERENCE_MODULE)


def test_resume_writes_combined_partial_without_duplicates(tmp_path, monkeypatch):
    """Resumed inference should preserve prior rows and append new rows exactly once."""
    prompts = [
        {"attack_prompt": "p0"},
        {"attack_prompt": "p1"},
        {"attack_prompt": "p2"},
    ]
    existing = [
        {"attack_prompt": "p0", "model_response": "old0", "system_prompt_mode": "base"}
    ]
    out = tmp_path / "responses.jsonl"

    def fake_batch_generate(_model, _tok, batch_items, _cfg, _sp=None):
        return [f"new::{item['attack_prompt']}" for item in batch_items]

    monkeypatch.setattr(
        RUN_INFERENCE_MODULE,
        "_batch_generate_single_turn",
        fake_batch_generate,
    )

    results = RUN_INFERENCE_MODULE.run_inference(
        model=None,
        tokenizer=None,
        prompts=prompts,
        gen_cfg={},
        system_prompt_mode="base",
        output_path=str(out),
        resume_from=1,
        existing_results=list(existing),
    )

    assert len(results) == 3
    assert [r["attack_prompt"] for r in results] == ["p0", "p1", "p2"]
    assert [r["model_response"] for r in results] == ["old0", "new::p1", "new::p2"]

    partial = load_jsonl(str(out) + ".partial")
    assert len(partial) == 3
    assert [r["attack_prompt"] for r in partial] == ["p0", "p1", "p2"]


def test_resume_no_work_when_already_complete(tmp_path):
    """If resume_from covers all prompts, run_inference should no-op cleanly."""
    prompts = [{"attack_prompt": "p0"}]
    existing = [{"attack_prompt": "p0", "model_response": "done"}]
    out = tmp_path / "responses.jsonl"

    results = RUN_INFERENCE_MODULE.run_inference(
        model=None,
        tokenizer=None,
        prompts=prompts,
        gen_cfg={},
        output_path=str(out),
        resume_from=1,
        existing_results=list(existing),
    )

    assert results == existing
    assert not Path(str(out) + ".partial").exists()
