#!/usr/bin/env python3
"""Run latency and cost benchmark comparing Strands Decider 2B vs LLM baseline.

Usage:
    python benchmark_comparison.py [--repetitions 1] [--output benchmark_results.json]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure backend root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pulseops.benchmark import (
    format_benchmark_markdown_table,
    generate_benchmark_report,
)
from pulseops.decider_client import Decider, ServerUnavailable
from pulseops.evaluation_dataset import EVALUATION_DATASET


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run latency and cost benchmark for PulseOps Decider vs LLM baseline."
    )
    parser.add_argument(
        "--url",
        default=os.environ.get("STRANDS_DECIDER_URL", "http://127.0.0.1:8099"),
        help="URL of the Strands Decider server (default: http://127.0.0.1:8099)",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=1,
        help="Repetitions per scenario for latency statistics (default: 1)",
    )
    parser.add_argument(
        "--output",
        default="benchmark_results.json",
        help="Output path for raw JSON benchmark results (default: benchmark_results.json)",
    )
    parser.add_argument(
        "--table-output",
        default=None,
        help="Optional path to save generated Markdown comparison table",
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

    print(
        f"\nExecuting benchmark across {len(EVALUATION_DATASET)} scenarios "
        f"({args.repetitions} repetition(s) + 1 warm-up run)..."
    )

    def on_progress(idx: int, total: int, scen_id: str, elapsed_ms: float, choice: str, correct: bool) -> None:
        mark = "✓" if correct else "✗"
        print(f"  [{idx:2d}/{total:2d}] {scen_id:<12} -> {choice:<20} ({elapsed_ms:6.1f} ms) [{mark}]", flush=True)

    report = generate_benchmark_report(
        dataset=EVALUATION_DATASET,
        decider=decider,
        repetitions=args.repetitions,
        include_mock_llm=False,
        progress_callback=on_progress,
    )

    md_table = format_benchmark_markdown_table(report)
    print("\n" + md_table + "\n")

    # Save JSON results
    try:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Raw benchmark metrics saved to: {os.path.abspath(args.output)}")
    except OSError as exc:
        print(f"Warning: Could not save JSON report to {args.output}: {exc}")

    # Optionally save Markdown table
    if args.table_output:
        try:
            with open(args.table_output, "w", encoding="utf-8") as f:
                f.write(md_table + "\n")
            print(f"Comparison table saved to: {os.path.abspath(args.table_output)}")
        except OSError as exc:
            print(f"Warning: Could not save table to {args.table_output}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
