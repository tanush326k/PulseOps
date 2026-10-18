"""Simulated diagnostic tools for PulseOps.

Each function returns realistic, deterministic evidence based on the incident
dict. All output is clearly labelled with source="simulated" so it cannot be
confused with real infrastructure telemetry.

Design rules
------------
- Pure Python, no network calls, no randomness (deterministic for tests).
- Evidence is incident-specific (uses incident fields, not generic lorem ipsum).
- Each tool returns a dict with at least: action, source, evidence, timestamp.
- The [SIMULATED] prefix appears in every evidence string.
"""
from __future__ import annotations

import datetime
from typing import Any

_TS = lambda: datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

SOURCE = "simulated"
LABEL = "[SIMULATED]"


def inspect_logs(incident: dict[str, Any]) -> dict[str, Any]:
    """Return simulated recent log output for the affected service."""
    svc = incident.get("service", "unknown-service")
    ts = _TS()
    evidence = (
        f"{LABEL} Log tail for {svc} (last 50 lines, {ts}):\n\n"
        f"  2026-10-17T00:47:13Z ERROR {svc}: connection pool exhausted (pool_size=50, waiting=127)\n"
        f"  2026-10-17T00:47:14Z ERROR {svc}: upstream timeout after 30s (url=https://payments-api/charge)\n"
        f"  2026-10-17T00:47:14Z WARN  {svc}: retry attempt 3/3 failed — circuit breaker OPEN\n"
        f"  2026-10-17T00:47:15Z ERROR {svc}: HTTP 503 returned to 847 in-flight requests\n"
        f"  2026-10-17T00:47:16Z INFO  {svc}: health check endpoint /health returning 200 (local)\n"
        f"  2026-10-17T00:47:19Z ERROR {svc}: goroutine leak detected — 2,341 blocked goroutines\n\n"
        f"Pattern: errors correlate with upstream payment service calls, not local crashes.\n"
        f"First error timestamp: 2026-10-17T00:47:10Z"
    )
    return {
        "action": "inspect_logs",
        "source": SOURCE,
        "evidence": evidence,
        "timestamp": ts,
        "key_finding": "Upstream payment service calls timing out; circuit breaker open.",
    }


def query_metrics(incident: dict[str, Any]) -> dict[str, Any]:
    """Return simulated metrics for the affected service."""
    svc = incident.get("service", "unknown-service")
    ts = _TS()
    evidence = (
        f"{LABEL} Metrics snapshot for {svc} — 15-min window ending {ts}:\n\n"
        f"  CPU usage         : 23% (baseline 18%) — NORMAL\n"
        f"  Memory usage      : 1.4 GB / 2.0 GB  — NORMAL\n"
        f"  Request rate      : 1,240 req/s (baseline 1,100 req/s) — NORMAL\n"
        f"  Error rate (5xx)  : 68.4% ← CRITICAL (baseline <0.1%)\n"
        f"  P50 latency       : 312 ms (baseline 45 ms) — ELEVATED\n"
        f"  P99 latency       : 31,200 ms — CRITICAL\n"
        f"  Active connections: 8,920 (max 10,000) — HIGH\n\n"
        f"Host resources are healthy; the error spike is not resource-bound.\n"
        f"Latency distribution tail points to a downstream dependency, not CPU/memory."
    )
    return {
        "action": "query_metrics",
        "source": SOURCE,
        "evidence": evidence,
        "timestamp": ts,
        "key_finding": "Error rate 68% (P99 latency 31s). Host resources healthy — downstream issue.",
    }


def compare_deployments(incident: dict[str, Any]) -> dict[str, Any]:
    """Return a simulated deployment diff."""
    svc = incident.get("service", "unknown-service")
    ts = _TS()
    evidence = (
        f"{LABEL} Deployment diff for {svc} — current vs last-known-good ({ts}):\n\n"
        f"  Image tag         : v2.14.1  →  v2.15.0  ← CHANGED\n"
        f"  Replica count     : 12       →  12        (unchanged)\n"
        f"  CPU request       : 250m     →  250m      (unchanged)\n"
        f"  ENV PAYMENT_URL   : https://pay.internal/v1  →  https://pay.internal/v2  ← CHANGED\n"
        f"  ENV TIMEOUT_MS    : 5000     →  30000     ← CHANGED (6x increase)\n"
        f"  ENV POOL_SIZE     : 100      →  50        ← CHANGED (halved)\n"
        f"  Config map rev    : cm-443   →  cm-444    ← CHANGED\n\n"
        f"Deployment pushed at 2026-10-17T00:44:52Z (3 min before first error).\n"
        f"PAYMENT_URL v2 endpoint and halved pool size are probable causes."
    )
    return {
        "action": "compare_deployments",
        "source": SOURCE,
        "evidence": evidence,
        "timestamp": ts,
        "key_finding": "v2.15.0 deployed 3 min before incident: PAYMENT_URL changed to /v2, POOL_SIZE halved.",
    }


def check_dependencies(incident: dict[str, Any]) -> dict[str, Any]:
    """Return simulated upstream/downstream dependency health."""
    svc = incident.get("service", "unknown-service")
    ts = _TS()
    evidence = (
        f"{LABEL} Dependency health check for {svc} ({ts}):\n\n"
        f"  payments-api  /v2  → HTTP 503 (latency: timeout)   ← UNHEALTHY\n"
        f"  payments-api  /v1  → HTTP 200 (latency: 42 ms)     ← HEALTHY\n"
        f"  auth-service       → HTTP 200 (latency: 8 ms)      ← HEALTHY\n"
        f"  postgres-primary   → connection OK (lag: 0 ms)     ← HEALTHY\n"
        f"  redis-cluster      → PING OK (latency: 1 ms)       ← HEALTHY\n"
        f"  kafka-broker       → offset lag: 12 (baseline: 8)  ← NORMAL\n\n"
        f"payments-api /v2 is the only unhealthy dependency.\n"
        f"payments-api /v1 (previous version) remains healthy — rollback is viable."
    )
    return {
        "action": "check_dependencies",
        "source": SOURCE,
        "evidence": evidence,
        "timestamp": ts,
        "key_finding": "payments-api /v2 is down. /v1 healthy. All other deps OK.",
    }


def ask_for_details(incident: dict[str, Any]) -> dict[str, Any]:
    """Return a structured clarification request for the on-call engineer."""
    ts = _TS()
    evidence = (
        f"{LABEL} Clarification request generated at {ts}:\n\n"
        f"The incident commander requires additional information before proceeding:\n\n"
        f"  1. Was a deployment, config change, or feature flag flip made in the last 30 minutes?\n"
        f"  2. Are any upstream teams reporting issues with shared services (payments, auth, DB)?\n"
        f"  3. Is this affecting all regions or a specific availability zone?\n"
        f"  4. What is the blast radius — how many customers/requests are impacted?\n"
        f"  5. Has a rollback been attempted? If so, what was the result?\n\n"
        f"Please provide answers via the incident channel before the next triage cycle."
    )
    return {
        "action": "ask_for_details",
        "source": SOURCE,
        "evidence": evidence,
        "timestamp": ts,
        "key_finding": "Insufficient data for autonomous diagnosis. Human input required.",
    }
