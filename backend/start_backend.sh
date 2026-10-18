#!/usr/bin/env bash
set -e
cd /mnt/d/PulseOps/backend
export STRANDS_DECIDER_URL="http://127.0.0.1:8099"
~/strands-decider/.venv/bin/python -m uvicorn pulseops.api:app --host 0.0.0.0 --port 8080
