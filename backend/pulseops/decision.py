"""Typed decision layer for PulseOps.

Builds the Strands Decider choice question for incident triage, validates the
returned action name, and routes to the correct diagnostic tool callable.

The decision is *genuine*: the action string from the model indexes into
TOOL_REGISTRY. An unknown string raises InvalidDecisionError rather than
silently defaulting -- the policy fails safe, not silently.

Integration pattern (documented for reuse)
------------------------------------------
1. Build a SystemOneRequest-compatible state string from incident data.
2. Call Decider.ask() with the TRIAGE_QUESTION dict.
3. Extract answer["action"]["choice"] -- the typed tool name.
4. Call route(choice) to get the callable; call it with the incident dict.

The question options and their descriptions live here; the model never sees
hard-coded tool names -- it reads the descriptions and picks the best fit.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from .decider_client import Decider

# ---------------------------------------------------------------------------
# The question sent to Strands Decider 2B
# ---------------------------------------------------------------------------

QUESTION_KEY = "action"

TRIAGE_OPTIONS: dict[str, str] = {
    "inspect_logs":         "Retrieve and analyse recent application and system log output for errors, exceptions, or anomalies that correlate with the incident timeline.",
    "query_metrics":        "Pull CPU, memory, request-rate, error-rate, and latency metrics for the affected service over the incident window.",
    "compare_deployments":  "Diff the current deployment against the last known-good deployment: image tags, config changes, environment variables, and replica counts.",
    "check_dependencies":   "Probe upstream and downstream service dependencies for health status, elevated latency, or circuit-breaker states.",
    "ask_for_details":      "The available information is insufficient; request additional context from the on-call engineer before proceeding.",
}

TRIAGE_QUESTION: dict[str, Any] = Decider.choice(
    instructions=(
        "You are an AI incident commander triaging a production incident. "
        "Based on the incident state below, choose the single most useful next diagnostic action."
    ),
    criteria=TRIAGE_OPTIONS,
)

# ---------------------------------------------------------------------------
# Known tool names (validated set)
# ---------------------------------------------------------------------------

KNOWN_ACTIONS: frozenset[str] = frozenset(TRIAGE_OPTIONS)

DEFAULT_CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("PULSEOPS_CONFIDENCE_THRESHOLD", "0.0")
)
FALLBACK_ACTION: str = "ask_for_details"


class InvalidDecisionError(ValueError):
    """Raised when the model returns an action name not in KNOWN_ACTIONS."""


@dataclass(frozen=True)
class RoutedDecision:
    """Represents a validated and threshold-checked decision."""

    action: str
    raw_action: str
    confidence: float
    probabilities: dict[str, float]
    fallback_applied: bool
    fallback_reason: str | None = None


def resolve_decision(
    answer: Mapping[str, Any],
    threshold: float | None = None,
    fallback_action: str = FALLBACK_ACTION,
) -> RoutedDecision:
    """Validate answer and apply conservative low-confidence fallback if needed.

    Parameters
    ----------
    answer : Mapping[str, Any]
        The dict returned by Decider for the triage question, containing
        'choice', 'confidence', and 'probabilities'.
    threshold : float | None
        Minimum confidence required to route raw choice. Defaults to
        PULSEOPS_CONFIDENCE_THRESHOLD (0.0).
    fallback_action : str
        Action to route when confidence is below threshold. Defaults to
        'ask_for_details'.

    Returns
    -------
    RoutedDecision
        Validated decision with raw and final actions clearly distinguished.

    Raises
    ------
    InvalidDecisionError
        If raw choice or fallback_action is not in KNOWN_ACTIONS.
    """
    raw_choice = answer.get("choice")
    if not isinstance(raw_choice, str) or raw_choice not in KNOWN_ACTIONS:
        raise InvalidDecisionError(
            f"Unknown action {raw_choice!r}. Valid actions: {sorted(KNOWN_ACTIONS)}"
        )

    if fallback_action not in KNOWN_ACTIONS:
        raise InvalidDecisionError(
            f"Invalid fallback action {fallback_action!r}. Valid actions: {sorted(KNOWN_ACTIONS)}"
        )

    # Validate confidence; missing/NaN/invalid is treated conservatively as 0.0
    conf_raw = answer.get("confidence")
    if conf_raw is None or not isinstance(conf_raw, (int, float)) or math.isnan(conf_raw):
        confidence = 0.0
    else:
        confidence = float(max(0.0, min(1.0, float(conf_raw))))

    # Validate probabilities
    probs_raw = answer.get("probabilities")
    if isinstance(probs_raw, dict):
        probabilities = {
            str(k): float(v)
            for k, v in probs_raw.items()
            if isinstance(v, (int, float)) and not math.isnan(v)
        }
    else:
        probabilities = {}

    target_threshold = threshold if threshold is not None else DEFAULT_CONFIDENCE_THRESHOLD
    if target_threshold > 0.0 and confidence < target_threshold:
        return RoutedDecision(
            action=fallback_action,
            raw_action=raw_choice,
            confidence=confidence,
            probabilities=probabilities,
            fallback_applied=True,
            fallback_reason=f"Confidence {confidence:.4f} is below safety threshold {target_threshold:.4f}",
        )

    return RoutedDecision(
        action=raw_choice,
        raw_action=raw_choice,
        confidence=confidence,
        probabilities=probabilities,
        fallback_applied=False,
        fallback_reason=None,
    )



# ---------------------------------------------------------------------------
# Routing table -- maps action name to tool callable
# ---------------------------------------------------------------------------
# Populated lazily to avoid a circular import; tools.py imports nothing from here.
# Using a function rather than a module-level dict means the import happens at
# call time, after all modules are initialised.

def _build_registry() -> dict[str, Callable]:
    from . import tools  # noqa: PLC0415
    return {
        "inspect_logs":        tools.inspect_logs,
        "query_metrics":       tools.query_metrics,
        "compare_deployments": tools.compare_deployments,
        "check_dependencies":  tools.check_dependencies,
        "ask_for_details":     tools.ask_for_details,
    }


_REGISTRY: dict[str, Callable] | None = None


def route(action: str) -> Callable:
    """Return the diagnostic tool callable for *action*.

    Raises
    ------
    InvalidDecisionError
        If *action* is not in KNOWN_ACTIONS.
    """
    if action not in KNOWN_ACTIONS:
        raise InvalidDecisionError(
            f"Unknown action {action!r}. Valid actions: {sorted(KNOWN_ACTIONS)}"
        )
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return _REGISTRY[action]


def build_state(incident: dict[str, Any]) -> str:
    """Render an incident dict into a plain-text state string for the Decider.

    The Decider reads this text and the question together; richer, more
    structured text generally produces better-calibrated decisions.
    """
    lines = [
        f"Incident ID   : {incident.get('id', 'N/A')}",
        f"Severity      : {incident.get('severity', 'unknown')}",
        f"Service       : {incident.get('service', 'unknown')}",
        f"Title         : {incident.get('title', '(no title)')}",
        f"Description   : {incident.get('description', '(none)')}",
    ]
    if "signals" in incident:
        lines.append("Observed signals:")
        for sig in incident["signals"]:
            lines.append(f"  - {sig}")
    if "context" in incident:
        lines.append(f"Additional context: {incident['context']}")
    return "\n".join(lines)
