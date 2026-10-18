#!/usr/bin/env bash
set -e
BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$BACKEND_DIR"

if [ -f "$BACKEND_DIR/../.env" ]; then
  set -a; source "$BACKEND_DIR/../.env"; set +a
fi

export STRANDS_DECIDER_URL="${STRANDS_DECIDER_URL:-http://127.0.0.1:8099}"
export ASSESSMENT_PROVIDER="${ASSESSMENT_PROVIDER:-mock}"
~/strands-decider/.venv/bin/python -m uvicorn pulseops.api:app --host 0.0.0.0 --port 8080 --reload


