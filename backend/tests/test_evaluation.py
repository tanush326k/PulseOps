"""Offline unit tests for decision evaluation, metrics, and low-confidence safety.

All tests run completely offline with mocks. No live Decider, AWS, or model downloads.
"""
from __future__ import annotations

import math
import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pulseops.decider_client import Decider, ServerUnavailable
from pulseops.decision import (
    KNOWN_ACTIONS,
    InvalidDecisionError,
    resolve_decision,
)
from pulseops.evaluation import (
    CaseResult,
    compute_metrics,
    format_evaluation_summary,
    run_evaluation,
)
from pulseops.evaluation_dataset import EVALUATION_DATASET


# ---------------------------------------------------------------------------
# 1. Dataset verification tests
# ---------------------------------------------------------------------------


class TestEvaluationDataset(unittest.TestCase):

    def test_dataset_size(self):
        self.assertEqual(len(EVALUATION_DATASET), 20)

    def test_all_five_actions_represented(self):
        actions_in_dataset = {item["expected_action"] for item in EVALUATION_DATASET}
        self.assertEqual(actions_in_dataset, KNOWN_ACTIONS)

    def test_each_action_has_equal_representation(self):
        counts = {}
        for item in EVALUATION_DATASET:
            act = item["expected_action"]
            counts[act] = counts.get(act, 0) + 1
        for act in KNOWN_ACTIONS:
            self.assertEqual(counts.get(act), 4, f"Action {act} should have exactly 4 cases")

    def test_dataset_schema(self):
        for item in EVALUATION_DATASET:
            for field in ("id", "title", "service", "severity", "description", "signals", "expected_action", "rationale"):
                self.assertIn(field, item, f"Missing {field} in {item.get('id')}")
            self.assertIn(item["expected_action"], KNOWN_ACTIONS)
            self.assertIsInstance(item["signals"], list)
            self.assertTrue(len(item["signals"]) > 0)


# ---------------------------------------------------------------------------
# 2. Low-confidence safety & decision resolution tests
# ---------------------------------------------------------------------------


class TestLowConfidenceResolution(unittest.TestCase):

    def test_above_threshold_routing(self):
        answer = {
            "choice": "compare_deployments",
            "confidence": 0.75,
            "probabilities": {"compare_deployments": 0.8, "inspect_logs": 0.2},
        }
        res = resolve_decision(answer, threshold=0.50)
        self.assertEqual(res.action, "compare_deployments")
        self.assertEqual(res.raw_action, "compare_deployments")
        self.assertFalse(res.fallback_applied)
        self.assertIsNone(res.fallback_reason)

    def test_below_threshold_fallback_to_ask_for_details(self):
        answer = {
            "choice": "compare_deployments",
            "confidence": 0.25,
            "probabilities": {"compare_deployments": 0.3, "inspect_logs": 0.25},
        }
        res = resolve_decision(answer, threshold=0.40)
        self.assertEqual(res.action, "ask_for_details")
        self.assertEqual(res.raw_action, "compare_deployments")
        self.assertTrue(res.fallback_applied)
        self.assertIn("below safety threshold", res.fallback_reason)

    def test_exact_threshold_behavior(self):
        """Confidence equal to threshold does not fall back."""
        answer = {"choice": "inspect_logs", "confidence": 0.50}
        res = resolve_decision(answer, threshold=0.50)
        self.assertEqual(res.action, "inspect_logs")
        self.assertFalse(res.fallback_applied)

    def test_zero_threshold_preserves_raw_action(self):
        answer = {"choice": "check_dependencies", "confidence": 0.05}
        res = resolve_decision(answer, threshold=0.0)
        self.assertEqual(res.action, "check_dependencies")
        self.assertFalse(res.fallback_applied)

    def test_missing_confidence_treated_as_zero(self):
        """Missing confidence must NOT be treated as high confidence."""
        answer = {"choice": "query_metrics", "confidence": None}
        res = resolve_decision(answer, threshold=0.10)
        self.assertEqual(res.confidence, 0.0)
        self.assertEqual(res.action, "ask_for_details")
        self.assertTrue(res.fallback_applied)

    def test_nan_confidence_treated_as_zero(self):
        answer = {"choice": "query_metrics", "confidence": float("nan")}
        res = resolve_decision(answer, threshold=0.10)
        self.assertEqual(res.confidence, 0.0)
        self.assertTrue(res.fallback_applied)

    def test_invalid_raw_action_raises_invalid_decision_error(self):
        """Allowlist validation is strictly enforced; unknown actions never route."""
        answer = {"choice": "reboot_everything", "confidence": 0.99}
        with self.assertRaises(InvalidDecisionError):
            resolve_decision(answer, threshold=0.50)

    def test_invalid_fallback_action_raises(self):
        answer = {"choice": "inspect_logs", "confidence": 0.10}
        with self.assertRaises(InvalidDecisionError):
            resolve_decision(answer, threshold=0.50, fallback_action="unknown_fallback")

    def test_preserves_probabilities_and_raw_action_for_audit(self):
        probs = {"inspect_logs": 0.4, "query_metrics": 0.3}
        answer = {"choice": "inspect_logs", "confidence": 0.15, "probabilities": probs}
        res = resolve_decision(answer, threshold=0.50)
        self.assertEqual(res.raw_action, "inspect_logs")
        self.assertEqual(res.probabilities, probs)


# ---------------------------------------------------------------------------
# 3. Metrics & confusion matrix tests
# ---------------------------------------------------------------------------


class TestConfusionMatrixAndMetrics(unittest.TestCase):

    def test_perfect_predictions_metrics(self):
        results = [
            CaseResult(
                id=f"T-{i}",
                title="title",
                expected="inspect_logs",
                predicted_raw="inspect_logs",
                routed_action="inspect_logs",
                confidence=0.8,
                probabilities={},
                fallback_applied=False,
                fallback_reason=None,
                correct=True,
                raw_correct=True,
            )
            for i in range(5)
        ]
        report = compute_metrics(results)
        self.assertEqual(report.accuracy, 1.0)
        self.assertEqual(report.correct_count, 5)
        self.assertEqual(report.per_action["inspect_logs"].precision, 1.0)
        self.assertEqual(report.per_action["inspect_logs"].recall, 1.0)
        self.assertEqual(report.confusion_matrix["inspect_logs"]["inspect_logs"], 5)

    def test_misclassification_and_confusion_matrix(self):
        results = [
            # 2 true inspect_logs
            CaseResult("1", "t", "inspect_logs", "inspect_logs", "inspect_logs", 0.7, {}, False, None, True, True),
            CaseResult("2", "t", "inspect_logs", "inspect_logs", "inspect_logs", 0.8, {}, False, None, True, True),
            # 1 expected inspect_logs misclassified as query_metrics
            CaseResult("3", "t", "inspect_logs", "query_metrics", "query_metrics", 0.5, {}, False, None, False, False),
            # 1 expected query_metrics correctly classified
            CaseResult("4", "t", "query_metrics", "query_metrics", "query_metrics", 0.9, {}, False, None, True, True),
        ]
        report = compute_metrics(results)
        self.assertEqual(report.total_cases, 4)
        self.assertEqual(report.correct_count, 3)
        self.assertEqual(report.accuracy, 0.75)

        # Matrix: expected inspect_logs -> routed query_metrics = 1
        self.assertEqual(report.confusion_matrix["inspect_logs"]["inspect_logs"], 2)
        self.assertEqual(report.confusion_matrix["inspect_logs"]["query_metrics"], 1)
        self.assertEqual(report.confusion_matrix["query_metrics"]["query_metrics"], 1)

        # inspect_logs: TP=2, FN=1, FP=0 -> precision=1.0, recall=2/3 (~0.6667)
        m_logs = report.per_action["inspect_logs"]
        self.assertEqual(m_logs.precision, 1.0)
        self.assertAlmostEqual(m_logs.recall, 2 / 3, places=3)

    def test_empty_results_raises_value_error(self):
        with self.assertRaises(ValueError):
            compute_metrics([])

    def test_format_summary_contains_all_actions_and_matrix(self):
        results = [
            CaseResult("1", "t", act, act, act, 0.6, {}, False, None, True, True)
            for act in sorted(KNOWN_ACTIONS)
        ]
        report = compute_metrics(results)
        summary = format_evaluation_summary(report)
        self.assertIn("Confusion Matrix", summary)
        for act in KNOWN_ACTIONS:
            self.assertIn(act, summary)


# ---------------------------------------------------------------------------
# 4. Evaluation runner tests (mocked Decider)
# ---------------------------------------------------------------------------


class TestEvaluationRunnerMocked(unittest.TestCase):

    def test_run_evaluation_with_mocked_decider(self):
        dataset = [
            {
                "id": "SIM-01",
                "title": "Database down",
                "service": "db",
                "severity": "P1",
                "description": "DB connection errors",
                "signals": ["500 errors"],
                "expected_action": "inspect_logs",
            },
            {
                "id": "SIM-02",
                "title": "CPU maxed",
                "service": "compute",
                "severity": "P2",
                "description": "High load",
                "signals": ["cpu 99%"],
                "expected_action": "query_metrics",
            },
        ]

        mock_decider = MagicMock()
        mock_decider.health.return_value = {"model": "test-checkpoint-2b"}
        # First call returns inspect_logs with high confidence; second returns compare_deployments with low confidence
        mock_decider.ask.side_effect = [
            {"action": {"choice": "inspect_logs", "confidence": 0.85, "probabilities": {"inspect_logs": 0.9}}},
            {"action": {"choice": "compare_deployments", "confidence": 0.20, "probabilities": {"compare_deployments": 0.3}}},
        ]

        report = run_evaluation(dataset, mock_decider, confidence_threshold=0.30)
        self.assertEqual(report.total_cases, 2)
        # First case was correct
        self.assertTrue(report.results[0].correct)
        self.assertFalse(report.results[0].fallback_applied)
        # Second case was below threshold 0.30 -> fallback to ask_for_details
        self.assertTrue(report.results[1].fallback_applied)
        self.assertEqual(report.results[1].routed_action, "ask_for_details")
        self.assertEqual(report.results[1].predicted_raw, "compare_deployments")

    def test_decider_failure_recorded_as_error(self):
        dataset = [
            {
                "id": "SIM-ERR",
                "title": "Fail case",
                "service": "fail",
                "severity": "P1",
                "description": "fail",
                "signals": [],
                "expected_action": "inspect_logs",
            }
        ]
        mock_decider = MagicMock()
        mock_decider.health.side_effect = ServerUnavailable("down")
        mock_decider.ask.side_effect = ServerUnavailable("down")

        report = run_evaluation(dataset, mock_decider)
        self.assertEqual(report.total_cases, 1)
        self.assertEqual(report.correct_count, 0)
        self.assertEqual(report.results[0].predicted_raw, "ERROR")
        self.assertEqual(report.results[0].routed_action, "ask_for_details")
        self.assertTrue(report.results[0].fallback_applied)


if __name__ == "__main__":
    unittest.main()
