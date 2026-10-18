"""Bedrock-backed incident assessment via the Strands Agents SDK.

This module provides a thin wrapper around strands-agents that:
  - Uses the Strands Agent with a Bedrock model to produce a structured
    incident assessment from diagnostic evidence.
  - Degrades gracefully if strands-agents is not installed or AWS credentials
    are absent, returning a clearly-labelled placeholder.

The Agent is stateless per-call: each invocation creates a fresh Agent so
that incident state does not leak between requests.

Environment variables (all required for real Bedrock calls):
    AWS_ACCESS_KEY_ID
    AWS_SECRET_ACCESS_KEY
    AWS_DEFAULT_REGION      (e.g. us-east-1)
    BEDROCK_MODEL_ID        (default: us.amazon.nova-lite-v1:0)
"""
from __future__ import annotations

import os
from typing import Any

BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "us.amazon.nova-lite-v1:0")

ASSESSMENT_PROMPT = """\
You are an expert SRE (Site Reliability Engineer) acting as an AI incident commander.
You have just completed an automated diagnostic action on a production incident.
Analyse the evidence below and produce a concise incident assessment with three sections:

1. SITUATION: A 2-3 sentence summary of what is happening and the probable root cause.
2. EVIDENCE: The key diagnostic findings that support your assessment (bullet points).
3. NEXT STEPS: Up to 5 prioritised, actionable remediation steps.

Keep the total response under 300 words. Be direct and operational.

Note: the diagnostic evidence is SIMULATED for demonstration purposes.
"""


def _unavailable_response(reason: str) -> dict[str, Any]:
    return {
        "assessment": (
            f"[Bedrock unavailable — {reason}]\n\n"
            "To enable real AI assessment:\n"
            "  1. Install strands-agents: pip install strands-agents\n"
            "  2. Set AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION\n"
            "  3. Set BEDROCK_MODEL_ID (e.g. us.amazon.nova-lite-v1:0)\n"
            "  4. Ensure your IAM role has bedrock:InvokeModel permission"
        ),
        "model": None,
        "available": False,
    }


def assess_incident(
    incident: dict[str, Any],
    action_taken: str,
    evidence: str,
) -> dict[str, Any]:
    """Run a Strands Agent to assess the incident and evidence.

    Parameters
    ----------
    incident:
        The raw incident dict (id, severity, service, title, description, signals).
    action_taken:
        The action name chosen by Strands Decider (e.g. 'compare_deployments').
    evidence:
        The full evidence string returned by the diagnostic tool.

    Returns
    -------
    dict with keys:
        assessment (str): The model's assessment text.
        model (str | None): The Bedrock model ID used, or None.
        available (bool): Whether Bedrock was reachable.
    """
    # --- check strands-agents is importable ---
    try:
        from strands import Agent  # noqa: PLC0415
    except ImportError:
        return _unavailable_response("strands-agents not installed")

    # --- check credentials are present ---
    if not os.environ.get("AWS_ACCESS_KEY_ID") or not os.environ.get("AWS_SECRET_ACCESS_KEY"):
        return _unavailable_response("AWS credentials not set")

    # --- build the user prompt ---
    user_msg = (
        f"INCIDENT DETAILS\n"
        f"  ID       : {incident.get('id', 'N/A')}\n"
        f"  Severity : {incident.get('severity', 'unknown')}\n"
        f"  Service  : {incident.get('service', 'unknown')}\n"
        f"  Title    : {incident.get('title', '')}\n"
        f"  Description: {incident.get('description', '')}\n\n"
        f"DIAGNOSTIC ACTION TAKEN: {action_taken}\n\n"
        f"EVIDENCE\n{evidence}\n\n"
        f"Please produce your incident assessment."
    )

    # --- call Bedrock ---
    try:
        agent = Agent(
            model=BEDROCK_MODEL_ID,
            system_prompt=ASSESSMENT_PROMPT,
        )
        result = agent(user_msg)
        # strands Agent returns an AgentResult; the text is in result.message
        assessment_text = str(result)
        return {
            "assessment": assessment_text,
            "model": BEDROCK_MODEL_ID,
            "available": True,
        }
    except Exception as exc:  # noqa: BLE001
        err_type = type(exc).__name__
        return _unavailable_response(f"{err_type}: {exc}")
