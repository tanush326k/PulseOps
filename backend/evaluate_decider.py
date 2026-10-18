#!/usr/bin/env python3
"""Run evaluation of Strands Decider 2B on the PulseOps dataset.

Usage:
    python evaluate_decider.py [--threshold 0.0] [--output eval_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pulseops.decider_client import Decider, ServerUnavailable
from pulseops.evaluation import (
    format_evaluation_summary,
    run_evaluation,
)
from pulseops.evaluation_dataset import EVALUATION_DATASET


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate Strands Decider 2B action selection on PulseOps benchmark."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:8099"),
        help="URL of the Strands Decider server (default: http://127.0.0.1:8099)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=float(os.environ.get("PULSEOPS_CONFIDENCE_THRESHOLD", "0.0")),
        help="Confidence threshold for low-confidence fallback (default: 0.0)",
    )
    parser.add_argument(
        "--output",
        default="eval_results.json",
        help="Path to save evaluation results JSON (default: eval_results.json)",
    )
    args = parser.parse_args()

    decider = Decider(url=args.url)

    print(f"Connecting to Strands Decider at {args.url}...")
    try:
        banner = decider.banner()
        print(f"Connected: {banner}")
    except ServerUnavailable as exc:
        print(f"\n[ERROR] Cannot connect to Strands Decider: {exc}")
        print("Please start the Decider server with:")
        print("    strands-decider serve <checkpoint> --port 8099")
        return 1

    print(f"\nEvaluating {len(EVALUATION_DATASET)} incidents (threshold={args.threshold:.2f})...")
    report = run_evaluation(
        dataset=EVALUATION_DATASET,
        decider=decider,
        confidence_threshold=args.threshold,
    )

    summary_text = format_evaluation_summary(report)
    print(summary_text)

    # Save JSON report
    out_path = args.output
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Full evaluation report saved to: {os.path.abspath(out_path)}")
    except OSError as exc:
        print(f"Warning: Could not save report to {out_path}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
