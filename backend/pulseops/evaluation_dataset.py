"""Labelled evaluation dataset for PulseOps incident triage.

A reproducible benchmark of 20 representative production incident scenarios
covering all 5 allowed actions in KNOWN_ACTIONS:
  - inspect_logs
  - query_metrics
  - compare_deployments
  - check_dependencies
  - ask_for_details

Includes both clear-cut diagnostic scenarios and ambiguous/insufficient cases.
Ground-truth labels are established independently of model predictions.
"""
from __future__ import annotations

from typing import Any

EVALUATION_DATASET: list[dict[str, Any]] = [
    # -------------------------------------------------------------------------
    # 1. compare_deployments (4 cases: 3 clear, 1 subtle)
    # -------------------------------------------------------------------------
    {
        "id": "EVAL-001",
        "title": "5xx error spike immediately following container image deployment",
        "service": "checkout-api",
        "severity": "P1",
        "description": "5xx errors jumped from 0.05% to 64% exactly 2 minutes after deployment of release v2.4.1.",
        "signals": [
            "Deployment tag: v2.4.1 rolled out at 14:02Z",
            "Error spike started at 14:04Z",
            "CPU and memory utilisation remain baseline",
        ],
        "expected_action": "compare_deployments",
        "difficulty": "clear",
        "rationale": "Direct temporal correlation with a new release tag makes diffing the deployment changes the primary triage step.",
    },
    {
        "id": "EVAL-002",
        "title": "Configuration drift causing nil pointer exceptions across pods",
        "service": "billing-service",
        "severity": "P1",
        "description": "Pods are crash-looping after an operator synced ConfigMaps and environment variables from staging into production.",
        "signals": [
            "Helm release upgraded 5 minutes ago",
            "ConfigMap checksum changed in pod spec",
            "CrashBackOff across all replicas",
        ],
        "expected_action": "compare_deployments",
        "difficulty": "clear",
        "rationale": "Configuration changes across deployments frequently break runtime initialization; deployment diff reveals altered environment variables.",
    },
    {
        "id": "EVAL-003",
        "title": "Canary deployment diverging from baseline traffic response",
        "service": "api-gateway",
        "severity": "P2",
        "description": "Canary group receiving 10% of traffic reports 15% error rate while the 90% stable baseline reports 0.01% error rate.",
        "signals": [
            "Canary release v1.18.0 deployed to 2 instances",
            "Stable release v1.17.4 running on 18 instances",
            "Error rate isolated strictly to canary instances",
        ],
        "expected_action": "compare_deployments",
        "difficulty": "clear",
        "rationale": "Failure is strictly partitioned to canary instances; comparing canary vs stable deployment spec identifies the defect.",
    },
    {
        "id": "EVAL-004",
        "title": "Subtle regression in worker throughput after replica count reduction",
        "service": "batch-processor",
        "severity": "P2",
        "description": "Queue backlog is accumulating following a scheduled scaling policy update that modified HPA parameters.",
        "signals": [
            "HPA minReplicas updated 20m ago",
            "Worker count decreased from 24 to 8",
            "Individual worker CPU is at 92%",
        ],
        "expected_action": "compare_deployments",
        "difficulty": "ambiguous",
        "rationale": "Although CPU is elevated, the root trigger was the deployment spec change that lowered replica counts.",
    },

    # -------------------------------------------------------------------------
    # 2. inspect_logs (4 cases: 3 clear, 1 subtle)
    # -------------------------------------------------------------------------
    {
        "id": "EVAL-005",
        "title": "Unhandled runtime exception flood in payments processing route",
        "service": "payment-router",
        "severity": "P1",
        "description": "Error rate spiked to 38% on /v2/charge without any recent deployment or infrastructure changes.",
        "signals": [
            "No deployments in last 72 hours",
            "500 Internal Server Error returned to clients",
            "Log volume elevated by 400% with error level messages",
        ],
        "expected_action": "inspect_logs",
        "difficulty": "clear",
        "rationale": "An unexpected 500 error spike without deployment changes requires inspecting stack traces in application logs.",
    },
    {
        "id": "EVAL-006",
        "title": "Database connection pool exhaustion and transaction aborts",
        "service": "user-account-svc",
        "severity": "P1",
        "description": "API handlers failing with 'connection refused' or 'pool timeout' when acquiring database sessions.",
        "signals": [
            "PostgreSQL database CPU is healthy at 25%",
            "Application logs reporting pool acquisition timeouts",
            "Thread dumps indicate blocked connection waiters",
        ],
        "expected_action": "inspect_logs",
        "difficulty": "clear",
        "rationale": "Log messages contain detailed connection pool acquisition errors, query timeouts, and stack traces.",
    },
    {
        "id": "EVAL-007",
        "title": "TLS certificate validation failures on internal gRPC channel",
        "service": "auth-service",
        "severity": "P1",
        "description": "Internal microservices unable to communicate with auth-service over mutual TLS.",
        "signals": [
            "gRPC status UNAVAILABLE (code 14)",
            "mTLS handshake errors in ingress logs",
            "Certificate rotation occurred at midnight",
        ],
        "expected_action": "inspect_logs",
        "difficulty": "clear",
        "rationale": "Cryptographic and TLS handshake failures are recorded with precise certificate expiry/CN details in application logs.",
    },
    {
        "id": "EVAL-008",
        "title": "Intermittent silent data parsing failures in ingestion worker",
        "service": "telemetry-collector",
        "severity": "P3",
        "description": "Metrics show flat request rates but downstream Kafka topic message rate dropped by 20%.",
        "signals": [
            "HTTP 200 returned to clients (swallowing errors)",
            "Warn/Error log count elevated in collector pods",
            "No crash loops or OOM kills",
        ],
        "expected_action": "inspect_logs",
        "difficulty": "ambiguous",
        "rationale": "Because the HTTP endpoint returns 200, log inspection is the only way to uncover swallowed serialization warnings.",
    },

    # -------------------------------------------------------------------------
    # 3. query_metrics (4 cases: 3 clear, 1 subtle)
    # -------------------------------------------------------------------------
    {
        "id": "EVAL-009",
        "title": "Slow memory leak causing periodic OOMKilled restarts",
        "service": "search-indexer",
        "severity": "P2",
        "description": "Pods are terminated by Linux OOM-killer every 4 to 6 hours in a predictable saw-tooth pattern.",
        "signals": [
            "Container memory working set linearly increasing from 400MB to 2048MB",
            "Exit code 137 (OOMKilled) recorded across pods",
            "No errors logged prior to kill",
        ],
        "expected_action": "query_metrics",
        "difficulty": "clear",
        "rationale": "A slow leak requires examining historical timeseries memory graphs to confirm growth rates and heap allocation trends.",
    },
    {
        "id": "EVAL-010",
        "title": "P99 latency degradation due to CPU throttling under burst traffic",
        "service": "product-catalog",
        "severity": "P2",
        "description": "P99 latency degraded from 80ms to 3,200ms while P50 remains 25ms and error rate is under 0.1%.",
        "signals": [
            "P99 latency: 3,200ms",
            "P50 latency: 25ms",
            "CPU throttling percentage: 78% of container periods throttled",
            "Error rate: 0.05%",
        ],
        "expected_action": "query_metrics",
        "difficulty": "clear",
        "rationale": "Disproportionate tail latency without errors points to resource saturation; querying CPU and queue latency metrics confirms throttling.",
    },
    {
        "id": "EVAL-011",
        "title": "Redis cache eviction and hit-ratio collapse",
        "service": "session-store",
        "severity": "P2",
        "description": "Database read queries multiplied 10x following memory pressure and eviction on the shared Redis cache.",
        "signals": [
            "Cache hit ratio dropped from 94% to 31%",
            "Redis evicted_keys counter spiking",
            "Database read IOPS at 98% provisioned capacity",
        ],
        "expected_action": "query_metrics",
        "difficulty": "clear",
        "rationale": "Cache performance issues are diagnosed by querying hit/miss ratios, eviction counters, and storage metrics.",
    },
    {
        "id": "EVAL-012",
        "title": "Network socket buffer exhaustion during flash promotion",
        "service": "event-streamer",
        "severity": "P2",
        "description": "Client connections dropping intermittently; system logs show general TCP retransmission warnings.",
        "signals": [
            "Active TCP connections: 64,000",
            "TCP retrans_rate: elevated",
            "Bandwidth utilization: 94% of NIC link speed",
        ],
        "expected_action": "query_metrics",
        "difficulty": "ambiguous",
        "rationale": "Network bottleneck analysis requires cross-referencing socket count, throughput, and packet drop timeseries metrics.",
    },

    # -------------------------------------------------------------------------
    # 4. check_dependencies (4 cases: 3 clear, 1 subtle)
    # -------------------------------------------------------------------------
    {
        "id": "EVAL-013",
        "title": "Downstream payment gateway returning 504 Gateway Timeout",
        "service": "checkout-api",
        "severity": "P1",
        "description": "Checkout service is failing transactions because third-party provider Stripe/Adyen is experiencing elevated latency.",
        "signals": [
            "Calls to https://external-payments-gw/charge timing out at 30s",
            "Local checkout-api memory and CPU are normal",
            "Internal database and Redis dependencies report healthy",
        ],
        "expected_action": "check_dependencies",
        "difficulty": "clear",
        "rationale": "The failure originates outside the service boundary; probing downstream dependencies confirms the third-party outage.",
    },
    {
        "id": "EVAL-014",
        "title": "Shared Kafka cluster partition leader election causing producer backpressure",
        "service": "order-publisher",
        "severity": "P1",
        "description": "Order creation is stalling because the message bus Kafka cluster has multiple offline partitions.",
        "signals": [
            "Kafka cluster under-replicated partitions: 14",
            "Producer buffer pool filling up",
            "Local order-publisher process is idle waiting on socket writes",
        ],
        "expected_action": "check_dependencies",
        "difficulty": "clear",
        "rationale": "A shared messaging middleware failure requires probing dependency cluster health and broker availability.",
    },
    {
        "id": "EVAL-015",
        "title": "Cloud identity provider rate-limiting token verification requests",
        "service": "customer-portal",
        "severity": "P2",
        "description": "User logins are failing with HTTP 429 when the service verifies JWT tokens against external IdP /jwks endpoint.",
        "signals": [
            "IdP endpoint https://idp.example.com/oauth/keys returning 429",
            "Internal microservices report healthy",
            "Local auth cache TTL expired",
        ],
        "expected_action": "check_dependencies",
        "difficulty": "clear",
        "rationale": "The bottleneck is an external authentication provider dependency.",
    },
    {
        "id": "EVAL-016",
        "title": "Cascading timeout propagation from deep downstream recommendation service",
        "service": "frontend-web",
        "severity": "P2",
        "description": "Homepage rendering latency increased because recommendation widget call at the bottom of the page is slow.",
        "signals": [
            "frontend-web CPU normal",
            "Sub-request to recs-service takes 4.8s",
            "Core page assets (catalog, cart) return in 50ms",
        ],
        "expected_action": "check_dependencies",
        "difficulty": "ambiguous",
        "rationale": "The issue is an unisolated downstream dependency degrading the composite response.",
    },

    # -------------------------------------------------------------------------
    # 5. ask_for_details (4 cases: 3 insufficient context, 1 contradictory)
    # -------------------------------------------------------------------------
    {
        "id": "EVAL-017",
        "title": "Vague customer support escalation: 'checkout feels sluggish'",
        "service": "unknown",
        "severity": "P3",
        "description": "A support ticket states: 'A user in Europe reported checkout took too long 2 hours ago.' No user ID, URL, or error code provided.",
        "signals": [
            "No specific service identified",
            "No error logs correlate with the reported window",
            "No timestamps or request trace IDs provided",
        ],
        "expected_action": "ask_for_details",
        "difficulty": "clear",
        "rationale": "The alert lacks service names, request IDs, and specific timestamps; on-call triage cannot proceed without additional context.",
    },
    {
        "id": "EVAL-018",
        "title": "Empty alert webhook missing payload headers and metric references",
        "service": "alertmanager",
        "severity": "P3",
        "description": "An automated alert fired with description: 'Alert fired on cluster-01' without mentioning which service, threshold, or subsystem triggered it.",
        "signals": [
            "Alert rule name: GenericAnomalyDetector",
            "Payload data: null",
            "Host: unassigned",
        ],
        "expected_action": "ask_for_details",
        "difficulty": "clear",
        "rationale": "Information is wholly insufficient to select a technical diagnostic tool; the engineer must clarify the alerting condition.",
    },
    {
        "id": "EVAL-019",
        "title": "Contradictory incident report claiming simultaneous network down and 100% throughput",
        "service": "network-gateway",
        "severity": "P2",
        "description": "On-call ticket states the data center link is down, but telemetry simultaneously shows normal ingress packet counts and 0 dropped connections.",
        "signals": [
            "Ticket says: 'Primary fiber cut reported by NOC'",
            "Gateway metrics report: 1.2 Gbps continuous traffic",
            "Router health check: PASS",
        ],
        "expected_action": "ask_for_details",
        "difficulty": "ambiguous",
        "rationale": "The human report directly contradicts automated telemetry; human clarification is needed to verify which environment or link is in question.",
    },
    {
        "id": "EVAL-020",
        "title": "Ambiguous multi-service degradation without clear attribution",
        "service": "microservices-mesh",
        "severity": "P3",
        "description": "Multiple engineers report disparate symptoms across 4 different microservices without a common error signature.",
        "signals": [
            "checkout reports 1% errors",
            "search reports 2% latency increase",
            "inventory reports normal",
            "no deployment or infrastructure events logged",
        ],
        "expected_action": "ask_for_details",
        "difficulty": "ambiguous",
        "rationale": "Low-level noise across multiple services without a common pattern requires human triage clarification before executing deep diagnostic probes.",
    },
]
