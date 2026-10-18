"""Evaluation engine for Strands Decider 2B incident triage.

Evaluates Decider action selection accuracy, confidence calibration,
confusion matrix, and low-confidence fallback behavior across the
representative evaluation dataset without invoking diagnostic tools or Bedrock.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

from .decider_client import Decider, ServerUnavailable
from .decision import (
    KNOWN_ACTIONS,
    QUESTION_KEY,
    TRIAGE_QUESTION,
    InvalidDecisionError,
    RoutedDecision,
    build_state,
    resolve_decision,
)


@dataclass
class CaseResult:
    id: str
    title: str
    expected: str
    predicted_raw: str
    routed_action: str
    confidence: float
    probabilities: dict[str, float]
    fallback_applied: bool
    fallback_reason: str | None
    correct: bool
    raw_correct: bool


@dataclass
class PerActionMetrics:
    action: str
    support: int
    predicted_count: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float


@dataclass
class EvaluationReport:
    total_cases: int
    correct_count: int
    accuracy: float
    raw_correct_count: int
    raw_accuracy: float
    confidence_threshold: float
    mean_confidence: float
    mean_correct_confidence: float
    mean_incorrect_confidence: float
    fallback_count: int
    fallback_rate: float
    per_action: dict[str, PerActionMetrics]
    confusion_matrix: dict[str, dict[str, int]]
    results: list[CaseResult]
    model_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(
    results: Sequence[CaseResult],
    confidence_threshold: float = 0.0,
    model_name: str | None = None,
) -> EvaluationReport:
    """Compute accuracy, per-action precision/recall, and confusion matrix."""
    actions = sorted(KNOWN_ACTIONS)
    total = len(results)
    if total == 0:
        raise ValueError("Cannot compute metrics on empty results list")

    correct_count = sum(1 for r in results if r.correct)
    raw_correct_count = sum(1 for r in results if r.raw_correct)
    fallback_count = sum(1 for r in results if r.fallback_applied)

    # Confusion matrix: matrix[expected][routed]
    matrix: dict[str, dict[str, int]] = {
        exp: {pred: 0 for pred in actions} for exp in actions
    }
    for r in results:
        if r.expected in matrix and r.routed_action in matrix[r.expected]:
            matrix[r.expected][r.routed_action] += 1

    # Per-action statistics
    per_action: dict[str, PerActionMetrics] = {}
    for act in actions:
        tp = matrix[act][act]
        fn = sum(matrix[act][p] for p in actions if p != act)
        fp = sum(matrix[e][act] for e in actions if e != act)
        support = sum(matrix[act].values())
        pred_count = sum(matrix[e][act] for e in actions)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        per_action[act] = PerActionMetrics(
            action=act,
            support=support,
            predicted_count=pred_count,
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
        )

    # Confidence analysis
    all_confs = [r.confidence for r in results]
    correct_confs = [r.confidence for r in results if r.correct]
    incorrect_confs = [r.confidence for r in results if not r.correct]

    mean_conf = sum(all_confs) / total if total else 0.0
    mean_corr_conf = sum(correct_confs) / len(correct_confs) if correct_confs else 0.0
    mean_incorr_conf = sum(incorrect_confs) / len(incorrect_confs) if incorrect_confs else 0.0

    return EvaluationReport(
        total_cases=total,
        correct_count=correct_count,
        accuracy=round(correct_count / total, 4),
        raw_correct_count=raw_correct_count,
        raw_accuracy=round(raw_correct_count / total, 4),
        confidence_threshold=round(confidence_threshold, 4),
        mean_confidence=round(mean_conf, 4),
        mean_correct_confidence=round(mean_corr_conf, 4),
        mean_incorrect_confidence=round(mean_incorr_conf, 4),
        fallback_count=fallback_count,
        fallback_rate=round(fallback_count / total, 4),
        per_action=per_action,
        confusion_matrix=matrix,
        results=list(results),
        model_name=model_name,
    )


def run_evaluation(
    dataset: Sequence[dict[str, Any]],
    decider: Decider,
    confidence_threshold: float = 0.0,
) -> EvaluationReport:
    """Evaluate Decider action selection against ground truth dataset."""
    results: list[CaseResult] = []

    for item in dataset:
        incident_id = item["id"]
        title = item.get("title", "")
        expected = item["expected_action"]
        state_text = build_state(item)

        try:
            answers = decider.ask(state_text, {QUESTION_KEY: TRIAGE_QUESTION})
            answer = answers[QUESTION_KEY]
            decision: RoutedDecision = resolve_decision(
                answer,
                threshold=confidence_threshold,
            )
            raw_action = decision.raw_action
            routed_action = decision.action
            confidence = decision.confidence
            probabilities = decision.probabilities
            fallback_applied = decision.fallback_applied
            fallback_reason = decision.fallback_reason
        except (ServerUnavailable, InvalidDecisionError, KeyError, Exception) as exc:
            # Failed or malformed decisions are recorded as failures
            raw_action = "ERROR"
            routed_action = "ask_for_details"
            confidence = 0.0
            probabilities = {}
            fallback_applied = True
            fallback_reason = f"Execution error: {type(exc).__name__}: {exc}"

        results.append(
            CaseResult(
                id=incident_id,
                title=title,
                expected=expected,
                predicted_raw=raw_action,
                routed_action=routed_action,
                confidence=confidence,
                probabilities=probabilities,
                fallback_applied=fallback_applied,
                fallback_reason=fallback_reason,
                correct=(routed_action == expected),
                raw_correct=(raw_action == expected),
            )
        )

    # Get model info if reachable
    model_name: str | None = None
    try:
        h = decider.health()
        model_name = h.get("model") or h.get("checkpoint")
    except Exception:
        model_name = "strands-decider-2b (offline/mocked)"

    return compute_metrics(
        results,
        confidence_threshold=confidence_threshold,
        model_name=model_name,
    )


def format_evaluation_summary(report: EvaluationReport) -> str:
    """Format the evaluation report into a clean, readable text report."""
    actions = sorted(KNOWN_ACTIONS)
    lines: list[str] = [
        "=" * 78,
        " PulseOps — Strands Decider 2B Action Selection Evaluation Report",
        "=" * 78,
        f"Model Endpoint       : {report.model_name or 'N/A'}",
        f"Total Scenarios      : {report.total_cases}",
        f"Raw Model Accuracy   : {report.raw_correct_count}/{report.total_cases} ({report.raw_accuracy * 100:.1f}%)",
        f"Routed Accuracy      : {report.correct_count}/{report.total_cases} ({report.accuracy * 100:.1f}%)",
        f"Confidence Threshold : {report.confidence_threshold:.2f}",
        f"Fallbacks Triggered  : {report.fallback_count}/{report.total_cases} ({report.fallback_rate * 100:.1f}%)",
        f"Mean Confidence      : {report.mean_confidence:.4f} (Correct: {report.mean_correct_confidence:.4f}, Incorrect: {report.mean_incorrect_confidence:.4f})",
        "",
        "-" * 78,
        " Per-Action Evaluation Breakdown",
        "-" * 78,
        f"{'Action':<22} {'Support':<8} {'Predicted':<10} {'Precision':<10} {'Recall':<8} {'F1':<6}",
        "-" * 78,
    ]

    for act in actions:
        m = report.per_action[act]
        lines.append(
            f"{act:<22} {m.support:<8} {m.predicted_count:<10} {m.precision:<10.2f} {m.recall:<8.2f} {m.f1:<6.2f}"
        )

    lines.extend([
        "",
        "-" * 78,
        " Confusion Matrix (Rows: Expected, Columns: Routed)",
        "-" * 78,
    ])

    header_cols = [f"{a[:10]:>11}" for a in actions]
    lines.append(f"{'Expected \\ Routed':<22}" + "".join(header_cols))
    for exp in actions:
        row_str = f"{exp:<22}"
        for pred in actions:
            row_str += f"{report.confusion_matrix[exp][pred]:>11}"
        lines.append(row_str)

    lines.extend(["=" * 78, ""])
    return "\n".join(lines)
