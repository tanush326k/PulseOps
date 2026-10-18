"""Tests for decision routing (Milestone 1).

All tests run without a live Decider server -- the HTTP call is mocked at the
urllib level so the routing logic is exercised independently of the model.

Run:
    cd backend
    python -m pytest tests/test_routing.py -v
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pulseops.decision import (
    KNOWN_ACTIONS,
    QUESTION_KEY,
    TRIAGE_OPTIONS,
    TRIAGE_QUESTION,
    InvalidDecisionError,
    build_state,
    route,
)


# ---------------------------------------------------------------------------
# Helper: build a fake urllib response for a given choice
# ---------------------------------------------------------------------------

def _fake_response(choice: str, probabilities: dict[str, float] | None = None):
    """Return a mock that looks like urllib.request.urlopen(...)."""
    if probabilities is None:
        # uniform-ish, choice gets highest mass
        probs = {k: 0.1 for k in TRIAGE_OPTIONS}
        probs[choice] = 0.6
    else:
        probs = probabilities
    payload = {
        "model": "strands-decider-test",
        "answers": {
            QUESTION_KEY: {
                "type": "choice",
                "choice": choice,
                "probabilities": probs,
                "confidence": 0.62,
            }
        },
        "usage": {"input_tokens": 42, "output_tokens": 1},
    }
    body = json.dumps(payload).encode()
    cm = MagicMock()
    cm.__enter__ = lambda s: SimpleNamespace(read=lambda: body)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ---------------------------------------------------------------------------
# Routing tests
# ---------------------------------------------------------------------------


class TestRouting(unittest.TestCase):
    """route() maps known action names to the correct callable."""

    def test_all_known_actions_route_successfully(self):
        for action in KNOWN_ACTIONS:
            fn = route(action)
            self.assertTrue(callable(fn), f"route({action!r}) should return a callable")

    def test_route_returns_distinct_callables(self):
        fns = {action: route(action) for action in KNOWN_ACTIONS}
        self.assertEqual(len(set(id(fn) for fn in fns.values())), len(KNOWN_ACTIONS))

    def test_invalid_action_raises_invalid_decision_error(self):
        with self.assertRaises(InvalidDecisionError):
            route("do_nothing")

    def test_invalid_action_error_message_lists_valid_options(self):
        try:
            route("hack_the_planet")
        except InvalidDecisionError as exc:
            msg = str(exc)
            for action in KNOWN_ACTIONS:
                self.assertIn(action, msg)

    def test_empty_string_raises(self):
        with self.assertRaises(InvalidDecisionError):
            route("")

    def test_case_sensitive(self):
        with self.assertRaises(InvalidDecisionError):
            route("Inspect_Logs")  # wrong case


# ---------------------------------------------------------------------------
# Decider client integration (mocked HTTP)
# ---------------------------------------------------------------------------


class TestDeciderClientMocked(unittest.TestCase):
    """Decider.ask() correctly parses the wire response."""

    def _ask_with_mock(self, choice: str) -> dict:
        from pulseops.decider_client import Decider
        decider = Decider(url="http://localhost:9999")
        fake = _fake_response(choice)
        with patch("urllib.request.urlopen", return_value=fake):
            incident = _sample_incident("deployment_change")
            state = build_state(incident)
            answers = decider.ask(state, {QUESTION_KEY: TRIAGE_QUESTION})
        return answers

    def test_returns_choice_key(self):
        answers = self._ask_with_mock("compare_deployments")
        self.assertIn(QUESTION_KEY, answers)
        self.assertEqual(answers[QUESTION_KEY]["choice"], "compare_deployments")

    def test_returns_probabilities_for_all_options(self):
        answers = self._ask_with_mock("inspect_logs")
        probs = answers[QUESTION_KEY]["probabilities"]
        for opt in TRIAGE_OPTIONS:
            self.assertIn(opt, probs)

    def test_tokens_counted(self):
        from pulseops.decider_client import Decider
        decider = Decider(url="http://localhost:9999")
        fake = _fake_response("query_metrics")
        with patch("urllib.request.urlopen", return_value=fake):
            decider.ask("state", {QUESTION_KEY: TRIAGE_QUESTION})
        self.assertEqual(decider.calls, 1)
        self.assertEqual(decider.input_tokens, 42)


# ---------------------------------------------------------------------------
# Representative incident scenarios
# ---------------------------------------------------------------------------


def _sample_incident(scenario: str) -> dict:
    SCENARIOS = {
        "deployment_change": {
            "id": "INC-001",
            "severity": "P1",
            "service": "checkout-api",
            "title": "Error rate spike immediately after deployment",
            "description": "5xx error rate jumped from <0.1% to 65% at 00:47Z. "
                           "A new image tag was pushed 3 minutes earlier.",
            "signals": [
                "error_rate: 65%",
                "deployment at 00:44Z",
                "P99 latency: 28s",
            ],
        },
        "resource_exhaustion": {
            "id": "INC-002",
            "severity": "P2",
            "service": "recommendation-engine",
            "title": "OOM kill loop",
            "description": "Pods restarting every 90 seconds with OOM exit code.",
            "signals": [
                "memory: 1.98GB / 2GB",
                "OOM kills: 12 in last 10 min",
            ],
        },
        "latency_spike": {
            "id": "INC-003",
            "severity": "P2",
            "service": "search-api",
            "title": "P99 latency degraded with no error rate change",
            "description": "P99 latency jumped from 200ms to 8s. Error rate unchanged.",
            "signals": [
                "P99_latency_ms: 8000",
                "error_rate: 0.09%",
                "cpu: 71%",
            ],
        },
        "dependency_outage": {
            "id": "INC-004",
            "severity": "P1",
            "service": "order-service",
            "title": "Payment dependency returning 503",
            "description": "All payment calls failing with 503. Other deps healthy.",
            "signals": [
                "payments_api_status: 503",
                "cart_service_status: 200",
                "auth_status: 200",
            ],
        },
        "unknown_cause": {
            "id": "INC-005",
            "severity": "P3",
            "service": "analytics-pipeline",
            "title": "Intermittent data gaps with no clear signal",
            "description": "Gaps appearing in event stream. No errors, no latency spike.",
            "signals": [
                "event_gap_pct: 3%",
                "error_rate: 0.0%",
                "cpu: 22%",
            ],
        },
    }
    return SCENARIOS[scenario]


class TestIncidentScenarios(unittest.TestCase):
    """Smoke tests: each scenario produces a valid state string and routes cleanly."""

    SCENARIOS = ["deployment_change", "resource_exhaustion", "latency_spike",
                 "dependency_outage", "unknown_cause"]

    def test_build_state_includes_all_fields(self):
        for scenario in self.SCENARIOS:
            incident = _sample_incident(scenario)
            state = build_state(incident)
            self.assertIn(incident["id"], state)
            self.assertIn(incident["service"], state)
            self.assertIn(incident["title"], state)

    def test_expected_routing_for_each_scenario(self):
        """
        For each scenario, mock the Decider to return the expected action and verify
        that route() correctly dispatches to a callable.
        """
        expected = {
            "deployment_change":  "compare_deployments",
            "resource_exhaustion": "query_metrics",
            "latency_spike":      "query_metrics",
            "dependency_outage":  "check_dependencies",
            "unknown_cause":      "ask_for_details",
        }
        for scenario, exp_action in expected.items():
            with self.subTest(scenario=scenario):
                fn = route(exp_action)
                incident = _sample_incident(scenario)
                result = fn(incident)
                self.assertEqual(result["action"], exp_action)
                self.assertEqual(result["source"], "simulated")
                self.assertIn("[SIMULATED]", result["evidence"])


class TestBuildState(unittest.TestCase):
    def test_minimal_incident(self):
        state = build_state({"id": "X-1", "severity": "P1", "service": "svc", "title": "t"})
        self.assertIn("X-1", state)
        self.assertIn("P1", state)
        self.assertIn("svc", state)

    def test_signals_included(self):
        state = build_state({
            "id": "X-2", "severity": "P2", "service": "svc", "title": "t",
            "signals": ["cpu: 90%", "errors: 50%"],
        })
        self.assertIn("cpu: 90%", state)
        self.assertIn("errors: 50%", state)


if __name__ == "__main__":
    unittest.main()
