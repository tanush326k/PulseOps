"""PulseOps FastAPI backend.

Endpoints
---------
GET  /api/health          Liveness check + Decider server reachability.
POST /api/run-incident    Run the full PulseOps pipeline for one incident.
GET  /                    Serve the frontend (index.html).

The full pipeline for /api/run-incident:
  1. Validate the request body (pydantic model).
  2. Build a plain-text state from the incident dict.
  3. Ask Strands Decider 2B for a typed choice over the 5 diagnostic actions.
  4. Validate the returned action name (InvalidDecisionError -> HTTP 422).
  5. Route to the matching simulated diagnostic tool.
  6. Ask Bedrock (via strands-agents) for an incident assessment.
  7. Return a structured JSON response to the frontend.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .bedrock_agent import assess_incident
from .decider_client import Decider, ServerUnavailable
from .decision import (
    KNOWN_ACTIONS,
    QUESTION_KEY,
    TRIAGE_OPTIONS,
    TRIAGE_QUESTION,
    InvalidDecisionError,
    build_state,
    resolve_decision,
    route,
)

# ---------------------------------------------------------------------------
# Predefined incident scenarios for the demo dropdown
# ---------------------------------------------------------------------------

DEMO_INCIDENTS: list[dict[str, Any]] = [
    {
        "id": "INC-001",
        "severity": "P1",
        "service": "checkout-api",
        "title": "Error rate spike immediately after deployment",
        "description": (
            "5xx error rate jumped from <0.1% to 65% at 00:47Z. "
            "A new image was deployed 3 minutes earlier. "
            "Customer checkout is failing."
        ),
        "signals": [
            "error_rate: 65%",
            "deployment pushed at 00:44Z (3 min before spike)",
            "P99 latency: 28s (baseline 120ms)",
            "CPU and memory normal",
        ],
    },
    {
        "id": "INC-002",
        "severity": "P2",
        "service": "recommendation-engine",
        "title": "OOM kill loop — pods restarting every 90 seconds",
        "description": (
            "Pods are restarting in a tight loop with OOM exit code. "
            "No recent deployment. "
            "Feature recommendations are unavailable."
        ),
        "signals": [
            "memory: 1.98GB / 2.0GB (99%)",
            "OOM kills: 12 in last 10 min",
            "CPU: 8% (normal)",
            "restart_count: 12",
        ],
    },
    {
        "id": "INC-003",
        "severity": "P1",
        "service": "order-service",
        "title": "Payment API returning 503 — all orders failing",
        "description": (
            "Every order placement attempt fails with payment_api 503. "
            "Auth service, database, and cache appear healthy. "
            "No recent deployment to order-service."
        ),
        "signals": [
            "payments_api_status: 503",
            "auth_service_status: 200",
            "postgres_status: healthy",
            "error_rate: 100% for /checkout endpoint",
        ],
    },
    {
        "id": "INC-004",
        "severity": "P2",
        "service": "search-api",
        "title": "P99 latency spike with no error rate change",
        "description": (
            "Search P99 latency climbed from 200ms to 8s over the past 20 minutes. "
            "Error rate is unchanged at 0.09%. "
            "CPU elevated at 71%."
        ),
        "signals": [
            "P99_latency_ms: 8000 (baseline 200ms)",
            "error_rate: 0.09% (normal)",
            "cpu: 71% (elevated from baseline 35%)",
            "no recent deployment",
        ],
    },
    {
        "id": "INC-005",
        "severity": "P3",
        "service": "analytics-pipeline",
        "title": "Intermittent data gaps with no clear signal",
        "description": (
            "3% of expected events are missing from the stream. "
            "No errors, no latency anomaly, no recent deployment. "
            "Gaps appeared 40 minutes ago."
        ),
        "signals": [
            "event_gap_pct: 3%",
            "error_rate: 0.0%",
            "cpu: 22%",
            "latency: normal",
        ],
    },
]


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class IncidentRequest(BaseModel):
    incident_id: str = Field(
        ...,
        max_length=64,
        description="Unique incident identifier, e.g. INC-001",
    )


class DeciderResult(BaseModel):
    action: str
    probabilities: dict[str, float]
    confidence: float
    raw_action: str | None = None
    fallback_applied: bool = False
    fallback_reason: str | None = None
    latency_ms: float | None = None


class RunIncidentResponse(BaseModel):
    incident: dict[str, Any]
    decider: DeciderResult
    tool_result: dict[str, Any]
    assessment: dict[str, Any]
    total_latency_ms: float


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent.parent / "frontend"


def create_app() -> FastAPI:
    app = FastAPI(
        title="PulseOps — AI Incident Commander",
        description="Strands Decider 2B · Amazon Bedrock · Incident Triage",
        version="1.0.0",
    )

    cors_origins_env = os.environ.get("PULSEOPS_CORS_ORIGINS", "").strip()
    if cors_origins_env == "*":
        allow_origins = ["*"]
        allow_origin_regex = None
    elif cors_origins_env:
        allow_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
        allow_origin_regex = None
    else:
        allow_origins = [
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
        allow_origin_regex = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_origin_regex=allow_origin_regex,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    decider = Decider()

    # --- API routes ---

    @app.get("/api/health")
    def health():
        """Liveness check. Reports whether the Decider server is reachable."""
        decider_status: dict[str, Any] = {"reachable": False}
        try:
            h = decider.health()
            decider_status = {"reachable": True, **h}
        except ServerUnavailable as exc:
            decider_status["error"] = str(exc)

        return {
            "status": "ok",
            "decider": decider_status,
            "known_actions": sorted(KNOWN_ACTIONS),
            "demo_incidents": [i["id"] for i in DEMO_INCIDENTS],
            "assessment_provider": os.environ.get("ASSESSMENT_PROVIDER", "mock"),
        }

    @app.get("/api/incidents")
    def list_incidents():
        """Return the predefined demo incident list."""
        return {"incidents": DEMO_INCIDENTS}

    @app.post("/api/run-incident", response_model=RunIncidentResponse)
    def run_incident(req: IncidentRequest):
        """Run the full PulseOps pipeline for one incident.

        Steps:
          1. Look up the incident by ID.
          2. Build a state string for Strands Decider.
          3. Ask Decider for a typed action choice.
          4. Validate and route to the diagnostic tool.
          5. Generate an incident assessment (provider controlled by ASSESSMENT_PROVIDER).
          6. Return the complete structured response.
        """
        t0 = time.monotonic()

        # Step 1: Look up incident
        incident = next(
            (i for i in DEMO_INCIDENTS if i["id"] == req.incident_id), None
        )
        if incident is None:
            raise HTTPException(
                status_code=404,
                detail=f"Unknown incident ID {req.incident_id!r}. "
                       f"Valid IDs: {[i['id'] for i in DEMO_INCIDENTS]}",
            )

        # Step 2: Build state
        state = build_state(incident)

        # Step 3: Ask Decider
        t_decider_start = time.monotonic()
        try:
            answers = decider.ask(state, {QUESTION_KEY: TRIAGE_QUESTION})
        except ServerUnavailable as exc:
            raise HTTPException(
                status_code=503,
                detail=f"Strands Decider server not reachable: {exc}\n"
                       f"Start it with: strands-decider serve <checkpoint> --port 8099",
            ) from exc
        t_decider_ms = (time.monotonic() - t_decider_start) * 1000

        answer = answers[QUESTION_KEY]
        # Step 4: Validate and route (with low-confidence fallback handling)
        try:
            decision = resolve_decision(answer)
            tool_fn = route(decision.action)
        except InvalidDecisionError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        tool_result = tool_fn(incident)

        # Step 5: Incident assessment
        assessment = assess_incident(
            incident=incident,
            action_taken=decision.action,
            evidence=tool_result["evidence"],
        )

        total_ms = (time.monotonic() - t0) * 1000

        return RunIncidentResponse(
            incident=incident,
            decider=DeciderResult(
                action=decision.action,
                raw_action=decision.raw_action,
                confidence=decision.confidence,
                probabilities=decision.probabilities,
                fallback_applied=decision.fallback_applied,
                fallback_reason=decision.fallback_reason,
                latency_ms=round(t_decider_ms, 1),
            ),
            tool_result=tool_result,
            assessment=assessment,
            total_latency_ms=round(total_ms, 1),
        )

    # --- Static frontend (must be last) ---
    if FRONTEND_DIR.exists():
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
    else:
        @app.get("/")
        def root():
            return {"message": "PulseOps backend running. Frontend not found."}

    return app


app = create_app()
