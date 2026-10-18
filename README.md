# PulseOps — AI Incident Commander

> **Strands Decider 2B Hackathon** · Tags: `#tool-selection` `#strands-decider-2b`

PulseOps uses **Strands Decider 2B** to make typed, calibrated decisions during production incident triage, then routes execution to the corresponding diagnostic tool. **Amazon Bedrock** (via the Strands Agents SDK) generates a concise incident assessment from the evidence.

The decision is **genuine**: the model's `choice` output directly controls which tool runs next — it is not cosmetic.

---

## Demo (< 3 min)

1. Start the servers with `./backend/run.sh`
2. Open `http://localhost:8080` in a browser
3. Select one of the five pre-loaded incidents
4. Click **Run Incident Analysis**
5. Watch Decider pick an action → evidence appears → Bedrock assessment streams in

---

## Architecture

```
┌──────────────────────── WSL2 Ubuntu ────────────────────────────────┐
│                                                                       │
│  Strands Decider 2B (port 8099)   ←── POST /v1/systemone             │
│       ↓  typed choice answer                                          │
│  PulseOps Backend (FastAPI, port 8080)                                │
│       ↓  validate → route → tool → evidence → Bedrock                │
│       ↓  JSON response                                                │
└───────────────────────────────────────────────────────────────────────┘
         ↑↓ fetch()
┌─────────── Browser ──────────────────┐
│  PulseOps Web UI (static HTML/JS)   │
└──────────────────────────────────────┘
```

**Decision routing** — [`backend/pulseops/decision.py`](backend/pulseops/decision.py):
- Builds a `ChoiceQuestion` with 5 named options.
- Calls `Decider.ask()` → `ChoiceAnswer.choice` is the action name.
- `route(action)` indexes a `dict[str, Callable]` — unknown string → `InvalidDecisionError`.

**Graceful degradation**:
- Decider down → HTTP 503 with clear error.
- Bedrock credentials absent → labelled placeholder text; rest of demo works.

---

## Project structure

```
PulseOps/
├── backend/
│   ├── pulseops/
│   │   ├── decider_client.py   ← HTTP client for /v1/systemone (no external deps)
│   │   ├── decision.py         ← ChoiceQuestion, validator, routing table
│   │   ├── tools.py            ← 5 simulated diagnostic tools (labelled [SIMULATED])
│   │   ├── bedrock_agent.py    ← strands-agents wrapper (graceful degradation)
│   │   └── api.py              ← FastAPI app
│   ├── tests/
│   │   ├── test_routing.py     ← 13 tests: routing, invalid decision, 5 scenarios
│   │   ├── test_tools.py       ← 13 tests: each tool, determinism
│   │   └── test_api.py         ← 13 tests: health, incidents, pipeline, error cases
│   ├── requirements.txt
│   └── run.sh                  ← starts Decider + backend
├── frontend/
│   ├── index.html
│   ├── style.css               ← dark glassmorphism design
│   └── app.js
├── .env.example                ← copy to .env and fill credentials
├── .gitignore
└── LICENSE                     ← Apache-2.0
```

---

## Setup

### Prerequisites

- **WSL2** with Ubuntu and Python 3.12
- `strands-decider` repo cloned and venv at `~/strands-decider/.venv`
- Checkpoint `StrandsAgents/strands-decider-2B-qwen3.5-v1-2610` downloaded

### 1. Install additional dependencies

```bash
# Bootstrap pip if needed (first time only)
~/strands-decider/.venv/bin/python -m ensurepip

~/strands-decider/.venv/bin/python -m pip install \
  strands-agents boto3 pytest httpx
```

### 2. Configure credentials (for live Bedrock)

```bash
cp .env.example .env
# Edit .env and fill in:
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#   AWS_DEFAULT_REGION  (e.g. us-east-1)
#   BEDROCK_MODEL_ID    (e.g. us.amazon.nova-lite-v1:0)
```

Without credentials, PulseOps shows a clearly-labelled placeholder and everything else still works.

### 3. Run tests

```bash
cd backend
~/strands-decider/.venv/bin/python -m pytest tests/ -v
# Expected: 39 passed
```

### 4. Start the demo

```bash
chmod +x backend/run.sh
./backend/run.sh
```

Open `http://localhost:8080`.

---

## Decision pattern (reusable)

The typed decision pattern in [`decision.py`](backend/pulseops/decision.py) is documented for reuse:

```python
from pulseops.decider_client import Decider
from pulseops.decision import TRIAGE_QUESTION, QUESTION_KEY, route, build_state

decider = Decider()  # connects to http://127.0.0.1:8099

# 1. Build state from your incident/task data
state = build_state(incident_dict)

# 2. Ask Decider for a typed choice
answers = decider.ask(state, {QUESTION_KEY: TRIAGE_QUESTION})

# 3. Extract the chosen action (validated, fail-safe)
action = answers[QUESTION_KEY]["choice"]          # e.g. "compare_deployments"
probs  = answers[QUESTION_KEY]["probabilities"]   # per-option scores

# 4. Route to the matching tool
tool_fn = route(action)   # raises InvalidDecisionError for unknown actions
result  = tool_fn(incident_dict)
```

To adapt for a different use case: replace `TRIAGE_OPTIONS` in `decision.py` with your own option dict, and update the routing table accordingly.

---

## API reference

### `GET /api/health`
Returns backend status and Decider server reachability.

### `GET /api/incidents`
Returns the list of demo incidents.

### `POST /api/run-incident`
```json
{ "incident_id": "INC-001" }
```
Runs the full pipeline and returns:
```json
{
  "incident": { ... },
  "decider": { "action": "compare_deployments", "probabilities": {...}, "confidence": 0.7 },
  "tool_result": { "action": "...", "source": "simulated", "evidence": "...", "key_finding": "..." },
  "assessment": { "assessment": "...", "model": "...", "available": true },
  "total_latency_ms": 1240.5
}
```

---

## Licence

Apache-2.0 — see [LICENSE](LICENSE).
