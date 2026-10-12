#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p logs
echo "$$" > logs/run.pid
trap 'status=$?; echo "$status" > logs/run.exit' EXIT
export WORKERS
WORKERS="$(.venv/bin/python -c 'import json; print(json.load(open("logs/benchmark.json"))["chosen"]["workers"])')"
export TORCH_THREADS=1
export ALLOW_PENDING_INPUTS=1
export RUN_HOURS=8
bash explore/gpu_run/launch.sh run
