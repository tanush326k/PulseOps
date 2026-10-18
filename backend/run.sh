#!/usr/bin/env bash
# run.sh — start both the Strands Decider server and the PulseOps backend
#
# Usage:
#   ./run.sh
#
# Environment variables (set in .env or export before running):
#   CHECKPOINT  - HuggingFace repo id or local path
#                 default: StrandsAgents/strands-decider-2B-qwen3.5-v1-2610
#   PORT        - backend port, default 8080
#   DECIDER_PORT- decider port, default 8099

set -euo pipefail

CHECKPOINT=${CHECKPOINT:-"StrandsAgents/strands-decider-2B-qwen3.5-v1-2610"}
DECIDER_PORT=${DECIDER_PORT:-8099}
PORT=${PORT:-8080}
VENV=~/strands-decider/.venv
BACKEND_DIR="$(cd "$(dirname "$0")" && pwd)"

# Load .env if present
if [ -f "$BACKEND_DIR/../.env" ]; then
  set -a; source "$BACKEND_DIR/../.env"; set +a
fi

echo "===================================================="
echo " PulseOps — AI Incident Commander"
echo "===================================================="
echo " Checkpoint : $CHECKPOINT"
echo " Decider    : http://127.0.0.1:$DECIDER_PORT"
echo " Backend    : http://127.0.0.1:$PORT"
echo "===================================================="

# Start Strands Decider server in background
echo ""
echo "[1/2] Starting Strands Decider server..."
$VENV/bin/strands-decider serve "$CHECKPOINT" --port "$DECIDER_PORT" --device cpu &
DECIDER_PID=$!
echo "      PID: $DECIDER_PID"

# Wait for Decider to be ready
echo "      Waiting for Decider to be ready..."
for i in $(seq 1 60); do
  if curl -sf "http://127.0.0.1:$DECIDER_PORT/health" > /dev/null 2>&1; then
    echo "      Decider ready after ${i}s"
    break
  fi
  sleep 1
done

# Start FastAPI backend
echo ""
echo "[2/2] Starting PulseOps backend..."
cd "$BACKEND_DIR"
STRANDS_DECIDER_URL="http://127.0.0.1:$DECIDER_PORT" \
  $VENV/bin/python -m uvicorn pulseops.api:app --host 0.0.0.0 --port "$PORT" --reload &
BACKEND_PID=$!
echo "      PID: $BACKEND_PID"

echo ""
echo "===================================================="
echo " Open http://localhost:$PORT in your browser"
echo " Press Ctrl+C to stop both processes"
echo "===================================================="

# Cleanup on exit
trap "echo 'Stopping...'; kill $DECIDER_PID $BACKEND_PID 2>/dev/null; wait" EXIT SIGINT SIGTERM
wait
