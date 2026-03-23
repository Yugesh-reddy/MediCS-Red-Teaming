"""Tests for medics.timing module."""

import json
import time
import tempfile
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import medics.timing as timing_module
from medics.timing import (
    timed_phase, save_timing_report, compute_gpu_hours,
    reset_timing_log,
)


class TestTimedPhase:
    def setup_method(self):
        reset_timing_log()

    def test_records_duration(self):
        with timed_phase("test phase"):
            time.sleep(0.05)
        assert len(timing_module._timing_log) == 1
        assert timing_module._timing_log[0]["phase"] == "test phase"
        assert timing_module._timing_log[0]["duration_sec"] >= 0.04

    def test_metadata_passed_through(self):
        with timed_phase("gpu phase", {"gpu": True, "round": 2}):
            pass
        assert timing_module._timing_log[0]["gpu"] is True
        assert timing_module._timing_log[0]["round"] == 2

    def test_default_gpu_false(self):
        with timed_phase("cpu phase"):
            pass
        assert timing_module._timing_log[0]["gpu"] is False

    def test_records_on_exception(self):
        try:
            with timed_phase("failing phase"):
                raise ValueError("test error")
        except ValueError:
            pass
        assert len(timing_module._timing_log) == 1
        assert timing_module._timing_log[0]["phase"] == "failing phase"


class TestSaveTimingReport:
    def setup_method(self):
        reset_timing_log()

    def test_saves_to_json(self):
        with timed_phase("phase 1"):
            pass
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        save_timing_report(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["phase"] == "phase 1"

    def test_appends_to_existing(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([{"phase": "old", "duration_sec": 1.0}], f)
            path = f.name
        with timed_phase("new phase"):
            pass
        save_timing_report(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 2
        assert data[0]["phase"] == "old"
        assert data[1]["phase"] == "new phase"


class TestComputeGpuHours:
    def test_basic_aggregation(self):
        data = [
            {"phase": "SFT Training", "duration_sec": 3600, "gpu": True},
            {"phase": "Inference", "duration_sec": 1800, "gpu": True},
            {"phase": "Dataset Build", "duration_sec": 600, "gpu": False},
        ]
        result = compute_gpu_hours(data)
        assert result["total_wall_clock_hours"] == 1.67
        assert result["total_gpu_hours"] == 1.5
        assert result["training_hours"] == 1.0
        assert result["inference_hours"] == 0.5
        assert result["n_phases"] == 3

    def test_empty_data(self):
        result = compute_gpu_hours([])
        assert result["total_gpu_hours"] == 0.0
        assert result["n_phases"] == 0
