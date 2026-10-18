"""Tests for the PulseOps FastAPI backend (Milestone 3/6).

These tests use FastAPI's TestClient (via httpx) and mock both the Decider
HTTP call and the Bedrock call, so no external services are needed.

Run:
    cd backend
    ~/strands-decider/.venv/bin/python -m pytest tests/test_api.py -v
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from pulseops.api import create_app, DEMO_INCIDENTS
from pulseops.decision import KNOWN_ACTIONS, QUESTION_KEY, TRIAGE_OPTIONS


def _make_decider_response(choice: str) -> MagicMock:
    """Mock urllib.request.urlopen returning a Decider choice answer."""
    probs = {k: 0.1 for k in TRIAGE_OPTIONS}
    probs[choice] = 0.6
    payload = {
        "model": "test-model",
        "answers": {
            QUESTION_KEY: {
                "type": "choice",
                "choice": choice,
                "probabilities": probs,
                "confidence": 0.62,
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }
    body = json.dumps(payload).encode()
    cm = MagicMock()
    cm.__enter__ = lambda s: SimpleNamespace(read=lambda: body)
    cm.__exit__ = MagicMock(return_value=False)
    return cm


def _mock_bedrock_unavailable():
    """Patch assess_incident to return unavailable response."""
    return {
        "assessment": "[Bedrock unavailable — test mode]",
        "model": None,
        "available": False,
    }


class TestHealthEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app, raise_server_exceptions=True)

    def test_health_returns_ok(self):
        # Decider will be unreachable in test; that's fine
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("known_actions", data)
        self.assertIn("decider", data)

    def test_health_lists_all_known_actions(self):
        resp = self.client.get("/api/health")
        data = resp.json()
        self.assertEqual(set(data["known_actions"]), KNOWN_ACTIONS)


class TestIncidentsEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def test_returns_all_incidents(self):
        resp = self.client.get("/api/incidents")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(len(data["incidents"]), len(DEMO_INCIDENTS))

    def test_incidents_have_required_fields(self):
        resp = self.client.get("/api/incidents")
        for inc in resp.json()["incidents"]:
            for field in ("id", "severity", "service", "title", "description"):
                self.assertIn(field, inc, f"Field {field!r} missing from {inc['id']}")


class TestRunIncidentEndpoint(unittest.TestCase):
    def setUp(self):
        self.app = create_app()
        self.client = TestClient(self.app)

    def _run(self, incident_id: str, choice: str = "compare_deployments"):
        fake = _make_decider_response(choice)
        with patch("urllib.request.urlopen", return_value=fake), \
             patch("pulseops.api.assess_incident", return_value=_mock_bedrock_unavailable()):
            return self.client.post(
                "/api/run-incident",
                json={"incident_id": incident_id},
            )

    def test_valid_incident_returns_200(self):
        resp = self._run("INC-001")
        self.assertEqual(resp.status_code, 200)

    def test_response_has_all_top_level_keys(self):
        resp = self._run("INC-001")
        data = resp.json()
        for key in ("incident", "decider", "tool_result", "assessment", "total_latency_ms"):
            self.assertIn(key, data)

    def test_decider_section_has_all_fields(self):
        resp = self._run("INC-001", "compare_deployments")
        d = resp.json()["decider"]
        self.assertEqual(d["action"], "compare_deployments")
        self.assertIn("probabilities", d)
        self.assertIn("confidence", d)
        for opt in TRIAGE_OPTIONS:
            self.assertIn(opt, d["probabilities"])

    def test_tool_result_is_simulated(self):
        resp = self._run("INC-001", "inspect_logs")
        tr = resp.json()["tool_result"]
        self.assertEqual(tr["source"], "simulated")
        self.assertIn("[SIMULATED]", tr["evidence"])

    def test_all_five_actions_route_and_produce_evidence(self):
        for action in KNOWN_ACTIONS:
            with self.subTest(action=action):
                resp = self._run("INC-001", action)
                self.assertEqual(resp.status_code, 200)
                tr = resp.json()["tool_result"]
                self.assertEqual(tr["action"], action)
                self.assertIn("[SIMULATED]", tr["evidence"])

    def test_unknown_incident_id_returns_404(self):
        resp = self._run("INC-NONEXISTENT")
        self.assertEqual(resp.status_code, 404)

    def test_missing_incident_id_returns_422(self):
        with patch("urllib.request.urlopen", return_value=_make_decider_response("inspect_logs")):
            resp = self.client.post("/api/run-incident", json={})
        self.assertEqual(resp.status_code, 422)

    def test_all_demo_incidents_return_200(self):
        for inc in DEMO_INCIDENTS:
            with self.subTest(id=inc["id"]):
                resp = self._run(inc["id"])
                self.assertEqual(resp.status_code, 200, f"Failed for {inc['id']}")

    def test_decider_unavailable_returns_503(self):
        import urllib.error
        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            resp = self.client.post("/api/run-incident", json={"incident_id": "INC-001"})
        self.assertEqual(resp.status_code, 503)


if __name__ == "__main__":
    unittest.main()
