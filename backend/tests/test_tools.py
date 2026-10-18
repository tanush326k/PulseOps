"""Tests for simulated diagnostic tools (Milestone 2).

Run:
    cd backend
    python -m pytest tests/test_tools.py -v
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import unittest
from pulseops import tools


BASE_INCIDENT = {
    "id": "INC-TEST-001",
    "severity": "P1",
    "service": "payments-gateway",
    "title": "High error rate",
    "description": "5xx spike after deploy",
    "signals": ["error_rate: 70%"],
}


def _check_common_fields(test, result: dict, expected_action: str):
    test.assertEqual(result["action"], expected_action)
    test.assertEqual(result["source"], "simulated")
    test.assertIn("[SIMULATED]", result["evidence"])
    test.assertIn("timestamp", result)
    test.assertIn("key_finding", result)
    test.assertIsInstance(result["evidence"], str)
    test.assertGreater(len(result["evidence"]), 100)


class TestInspectLogs(unittest.TestCase):
    def test_returns_correct_action(self):
        r = tools.inspect_logs(BASE_INCIDENT)
        _check_common_fields(self, r, "inspect_logs")

    def test_includes_service_name(self):
        r = tools.inspect_logs(BASE_INCIDENT)
        self.assertIn(BASE_INCIDENT["service"], r["evidence"])

    def test_works_with_missing_service(self):
        r = tools.inspect_logs({})
        self.assertIn("unknown-service", r["evidence"])


class TestQueryMetrics(unittest.TestCase):
    def test_returns_correct_action(self):
        r = tools.query_metrics(BASE_INCIDENT)
        _check_common_fields(self, r, "query_metrics")

    def test_includes_metric_names(self):
        r = tools.query_metrics(BASE_INCIDENT)
        self.assertIn("Error rate", r["evidence"])
        self.assertIn("latency", r["evidence"])


class TestCompareDeployments(unittest.TestCase):
    def test_returns_correct_action(self):
        r = tools.compare_deployments(BASE_INCIDENT)
        _check_common_fields(self, r, "compare_deployments")

    def test_includes_deployment_details(self):
        r = tools.compare_deployments(BASE_INCIDENT)
        self.assertIn("CHANGED", r["evidence"])
        self.assertIn("Image tag", r["evidence"])


class TestCheckDependencies(unittest.TestCase):
    def test_returns_correct_action(self):
        r = tools.check_dependencies(BASE_INCIDENT)
        _check_common_fields(self, r, "check_dependencies")

    def test_includes_health_status(self):
        r = tools.check_dependencies(BASE_INCIDENT)
        self.assertIn("HEALTHY", r["evidence"])
        self.assertIn("UNHEALTHY", r["evidence"])


class TestAskForDetails(unittest.TestCase):
    def test_returns_correct_action(self):
        r = tools.ask_for_details(BASE_INCIDENT)
        _check_common_fields(self, r, "ask_for_details")

    def test_includes_questions(self):
        r = tools.ask_for_details(BASE_INCIDENT)
        self.assertIn("deployment", r["evidence"])
        self.assertIn("rollback", r["evidence"])


class TestDeterminism(unittest.TestCase):
    """Same incident produces identical evidence (no randomness)."""

    def test_inspect_logs_is_deterministic(self):
        r1 = tools.inspect_logs(BASE_INCIDENT)
        r2 = tools.inspect_logs(BASE_INCIDENT)
        # Evidence content is identical (timestamps will differ by <1s; compare structure)
        self.assertEqual(r1["key_finding"], r2["key_finding"])

    def test_compare_deployments_is_deterministic(self):
        r1 = tools.compare_deployments(BASE_INCIDENT)
        r2 = tools.compare_deployments(BASE_INCIDENT)
        self.assertEqual(r1["key_finding"], r2["key_finding"])


if __name__ == "__main__":
    unittest.main()
