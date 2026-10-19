# PulseOps Latency and Cost Benchmark Methodology

## 1. Overview and Objective

In production incident triage, rapid and reliable diagnostic tool selection is critical. This benchmark evaluates:

- **Approach A — Strands Decider 2B**: Specialized 2B parameter decision model running on local CPU, selecting one of five diagnostic actions in a single forward pass without iterative autoregressive generation.
- **Approach B — Generative LLM Baseline**: General-purpose LLM (Amazon Bedrock Nova Lite / Claude 3.5 Sonnet) prompted with system instructions, action definitions, incident context, and JSON schema constraints.

The objective is to produce a reproducible comparison measuring latency and documenting cost implications without fabricating results.

---

## 2. Approach Definitions

### Approach A: Decider-Based Routing
- **Model**: `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` (base: `Qwen/Qwen3.5-2B-Base`).
- **Endpoint**: Local HTTP daemon listening on `http://127.0.0.1:8099/v1/systemone`.
- **Allowed Actions**:
  1. `inspect_logs`: Retrieve and analyse application/container logs.
  2. `query_metrics`: Query CPU, memory, error rate, and request rate metrics.
  3. `compare_deployments`: Diff recent deployment commits, config maps, and rollouts.
  4. `check_dependencies`: Probe database, cache, gateway, and third-party dependencies.
  5. `ask_for_details`: Solicit missing diagnostic context when input is ambiguous.
- **Timing Scope**: Measured using `time.perf_counter()` around the `Decider.ask()` call. Includes localhost loopback HTTP transfer (<1ms) and model forward pass.
- **Warm-up Policy**: The first inference request triggers model initialization and prefix cache warming. This warm-up run is recorded separately (`warmup_latency_ms`) and strictly excluded from steady-state latency statistics (mean, median, p90, p95).

### Approach B: Generative LLM Baseline
- **Prompt Structure**:
  ```text
  You are an AI SRE incident commander. Given an incident description, select the single best diagnostic action from this exact allowlist:
  - inspect_logs: Retrieve and analyse recent application and system log output.
  - query_metrics: Pull CPU, memory, request-rate, error-rate, and latency metrics.
  - compare_deployments: Diff current deployment against last known-good deployment.
  - check_dependencies: Probe upstream/downstream service dependencies.
  - ask_for_details: The available information is insufficient; request context.

  Respond strictly with a JSON object: {"action": "<action_name>"}
  ```
- **Validation**: Output is parsed as JSON and validated against the 5-item allowlist.
- **Bedrock Status Handling**: When the AWS account hits daily token quotas (`ThrottlingException: Rate exceeded`), the baseline is recorded as **blocked / unmeasured**. To uphold transparency, no fake latency, accuracy, or token figures are fabricated.

---

## 3. Cost Methodology and Economic Modeling

### AWS Pricing References (Documented 2025)

| Service / Resource | Pricing Tier | Rate | Source |
|---|---|---|---|
| **Amazon Bedrock Nova Lite** | Input Tokens | $0.06 per 1,000,000 tokens | AWS Bedrock Pricing Documentation (2025) |
| **Amazon Bedrock Nova Lite** | Output Tokens | $0.24 per 1,000,000 tokens | AWS Bedrock Pricing Documentation (2025) |
| **Amazon Bedrock Claude 3.5 Sonnet** | Input Tokens | $3.00 per 1,000,000 tokens | AWS Bedrock Pricing Documentation (2025) |
| **Amazon Bedrock Claude 3.5 Sonnet** | Output Tokens | $15.00 per 1,000,000 tokens | AWS Bedrock Pricing Documentation (2025) |
| **AWS EC2 c6i.xlarge (Compute)** | On-Demand (us-east-1) | $0.17 per hour ($4.08/day) | AWS EC2 Pricing (2025) |

### Token Accounting & Formulas

1. **Decider 2B**:
   - Average Input: ~45 tokens (incident text).
   - Average Output: 1 decision token / slot.
   - **Per-Token API Cost**: **$0.00** (self-hosted / internal compute).
   - **Compute Hosting Consideration**: On a dedicated EC2 `c6i.xlarge` instance ($0.17/hr):
     - At ~20 seconds per decision on CPU, 1 vCPU instance handles ~180 triage decisions/hour (~4,320 decisions/day).
     - Compute cost per 1,000 decisions: `(1000 / 180) * $0.17 ≈ $0.94`.

2. **Generative LLM (Nova Lite)**:
   - Average Input: ~420 tokens (system prompt + 5 tool descriptions + incident context).
   - Average Output: ~25 tokens (`{"action": "inspect_logs"}`).
   - Cost Formula:
     $$\text{Cost}_{1000} = \left(1000 \times 420 \times \frac{\$0.06}{10^6}\right) + \left(1000 \times 25 \times \frac{\$0.24}{10^6}\right) = \$0.0252 + \$0.0060 = \mathbf{\$0.0312}$$
   - Cost per 1,000 decisions: **~$0.031**.

3. **Generative LLM (Claude 3.5 Sonnet)**:
   - Cost Formula:
     $$\text{Cost}_{1000} = \left(1000 \times 420 \times \frac{\$3.00}{10^6}\right) + \left(1000 \times 25 \times \frac{\$15.00}{10^6}\right) = \$1.26 + \$0.375 = \mathbf{\$1.635}$$
   - Cost per 1,000 decisions: **~$1.64**.

---

## 4. Benchmark Execution Commands

To execute the benchmark against a running Strands Decider instance:

```bash
cd backend
~/strands-decider/.venv/bin/python benchmark_comparison.py --output benchmark_results.json
```

Output is serialized to machine-readable JSON (`benchmark_results.json`) and formatted as a Markdown table ready for documentation.
