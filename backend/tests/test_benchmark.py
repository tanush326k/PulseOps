"""Unit tests for PulseOps Latency and Cost Benchmark.

Tests metric calculations, percentile math, failure handling, mock runner,
blocked LLM provider handling, and JSON/Markdown serialization.
Completely offline — makes no network calls, AWS calls, or model downloads.
"""
from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import pytest

from pulseops.benchmark import (
    ApproachSummary,
    BenchmarkReport,
    BenchmarkRun,
    LatencyStats,
    calculate_latency_stats,
    calculate_percentile,
    format_benchmark_markdown_table,
    generate_benchmark_report,
    run_decider_benchmark,
    run_llm_baseline_benchmark,
)
from pulseops.decider_client import ServerUnavailable


# -----------------------------------------------------------------------------
# Percentile and Latency Statistics Tests
# -----------------------------------------------------------------------------

def test_calculate_percentile_empty():
    assert calculate_percentile([], 50) == 0.0
    assert calculate_percentile([], 95) == 0.0


def test_calculate_percentile_single_value():
    vals = [42.0]
    assert calculate_percentile(vals, 0) == 42.0
    assert calculate_percentile(vals, 50) == 42.0
    assert calculate_percentile(vals, 100) == 42.0


def test_calculate_percentile_known_distributions():
    # [10, 20, 30, 40, 50]
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert calculate_percentile(vals, 0) == 10.0
    assert calculate_percentile(vals, 50) == 30.0
    assert calculate_percentile(vals, 100) == 50.0
    # Interpolated p90: index = 4 * 0.9 = 3.6 -> 40 + 0.6*(50-40) = 46.0
    assert math.isclose(calculate_percentile(vals, 90), 46.0)


def test_calculate_latency_stats_empty():
    stats = calculate_latency_stats([])
    assert stats.count == 0
    assert stats.mean_ms == 0.0
    assert stats.median_ms == 0.0
    assert stats.p95_ms == 0.0
    assert stats.stddev_ms == 0.0


def test_calculate_latency_stats_populated():
    latencies = [100.0, 200.0, 300.0, 400.0, 500.0]
    stats = calculate_latency_stats(latencies)
    assert stats.count == 5
    assert stats.mean_ms == 300.0
    assert stats.median_ms == 300.0
    assert stats.min_ms == 100.0
    assert stats.max_ms == 500.0
    # stddev of [100, 200, 300, 400, 500] = sqrt(20000) ~= 141.42
    assert math.isclose(stats.stddev_ms, 141.42, rel_tol=1e-2)
    assert stats.p95_ms == 480.0


# -----------------------------------------------------------------------------
# Decider Benchmark Runner Tests
# -----------------------------------------------------------------------------

def test_run_decider_benchmark_mocked_success():
    sample_dataset = [
        {"id": "scen-1", "description": "CrashLoop", "expected_action": "inspect_logs"},
        {"id": "scen-2", "description": "High CPU", "expected_action": "query_metrics"},
    ]

    mock_decider = MagicMock()
    # Mock ask() to return inspect_logs for first, query_metrics for second
    mock_decider.ask.side_effect = [
        # Warmup call
        {"action": {"choice": "inspect_logs", "confidence": 0.9}},
        # Measured runs
        {"action": {"choice": "inspect_logs", "confidence": 0.9}},
        {"action": {"choice": "query_metrics", "confidence": 0.85}},
    ]

    summary, raw_runs = run_decider_benchmark(
        dataset=sample_dataset,
        decider=mock_decider,
        repetitions=1,
        warmup_runs=1,
    )

    # 1 warmup + 2 measured calls = 3 total ask calls
    assert mock_decider.ask.call_count == 3
    assert summary.status == "measured"
    assert summary.total_runs == 2
    assert summary.successful_runs == 2
    assert summary.failed_runs == 0
    assert summary.accuracy == 1.0  # Both matched expected_action
    assert len(raw_runs) == 2
    assert raw_runs[0].correct is True
    assert raw_runs[1].correct is True


def test_run_decider_benchmark_handles_failures():
    sample_dataset = [
        {"id": "scen-1", "description": "Test failure", "expected_action": "inspect_logs"},
    ]

    mock_decider = MagicMock()
    mock_decider.ask.side_effect = ServerUnavailable("Connection refused")

    summary, raw_runs = run_decider_benchmark(
        dataset=sample_dataset,
        decider=mock_decider,
        repetitions=1,
        warmup_runs=1,
    )

    assert summary.total_runs == 1
    assert summary.successful_runs == 0
    assert summary.failed_runs == 1
    assert summary.accuracy == 0.0
    assert len(raw_runs) == 1
    assert raw_runs[0].correct is False
    assert raw_runs[0].error is not None
    assert "Connection refused" in raw_runs[0].error


# -----------------------------------------------------------------------------
# LLM Baseline Runner Tests
# -----------------------------------------------------------------------------

def test_run_llm_baseline_default_is_blocked_and_honest():
    """Default runner without live Bedrock records unmeasured/blocked truthfully."""
    sample_dataset = [
        {"id": "scen-1", "description": "High CPU", "expected_action": "query_metrics"},
    ]

    summary, raw_runs = run_llm_baseline_benchmark(sample_dataset)

    assert summary.status == "blocked"
    assert "throttling" in summary.status_reason.lower() or "quota" in summary.status_reason.lower()
    assert summary.successful_runs == 0
    assert summary.accuracy is None
    assert summary.latency_stats is None
    assert summary.total_input_tokens is None
    assert len(raw_runs) == 0
    # Published cost projection is present and documented
    assert "calculated_cost_per_1k_incidents_nova_lite" in summary.cost_details


def test_run_llm_baseline_mock_runner():
    """When a mock runner is explicitly injected (for test harnesses), it measures correctly."""
    sample_dataset = [
        {"id": "scen-1", "description": "High CPU", "expected_action": "query_metrics"},
        {"id": "scen-2", "description": "502 Bad Gateway", "expected_action": "check_dependencies"},
    ]

    mock_runner = MagicMock()
    mock_runner.side_effect = [
        {"action": "query_metrics", "input_tokens": 410, "output_tokens": 22},
        {"action": "check_dependencies", "input_tokens": 430, "output_tokens": 25},
    ]

    summary, raw_runs = run_llm_baseline_benchmark(sample_dataset, mock_runner=mock_runner)

    assert summary.status == "measured"
    assert summary.total_runs == 2
    assert summary.successful_runs == 2
    assert summary.accuracy == 1.0
    assert summary.latency_stats is not None
    assert summary.latency_stats.count == 2
    assert len(raw_runs) == 2
    assert raw_runs[0].input_tokens == 410


# -----------------------------------------------------------------------------
# Report Generation and Serialization Tests
# -----------------------------------------------------------------------------

def test_generate_benchmark_report_and_dict_serialization():
    sample_dataset = [
        {"id": "scen-1", "description": "Crash", "expected_action": "inspect_logs"},
    ]

    mock_decider = MagicMock()
    mock_decider.ask.return_value = {"action": {"choice": "inspect_logs", "confidence": 0.8}}

    report = generate_benchmark_report(
        dataset=sample_dataset,
        decider=mock_decider,
        repetitions=1,
        include_mock_llm=False,
    )

    assert isinstance(report, BenchmarkReport)
    assert report.scenarios_evaluated == 1
    assert "decider_2b" in report.approaches
    assert "llm_baseline" in report.approaches

    # Serialize to dictionary
    data = report.to_dict()
    assert isinstance(data, dict)
    assert "timestamp" in data
    assert "hardware" in data
    assert data["scenarios_evaluated"] == 1
    assert data["approaches"]["decider_2b"]["status"] == "measured"
    assert data["approaches"]["llm_baseline"]["status"] == "blocked"


def test_format_benchmark_markdown_table():
    sample_dataset = [
        {"id": "scen-1", "description": "Crash", "expected_action": "inspect_logs"},
    ]

    mock_decider = MagicMock()
    mock_decider.ask.return_value = {"action": {"choice": "inspect_logs", "confidence": 0.8}}

    report = generate_benchmark_report(
        dataset=sample_dataset,
        decider=mock_decider,
        repetitions=1,
        include_mock_llm=False,
    )

    table_md = format_benchmark_markdown_table(report)

    assert "### Benchmark: Strands Decider 2B vs. Generative LLM Baseline" in table_md
    assert "| **Status** |" in table_md
    assert "✅ Measured" in table_md
    assert "⏸ Blocked (Quota Throttled)" in table_md
    assert "| **Action Accuracy** |" in table_md
    assert "| **Warm-up Latency** |" in table_md
    assert "| **Steady-State Mean Latency** |" in table_md
    assert "| **Cost / 1,000 Decisions** |" in table_md
    assert "Methodology Note" in table_md
