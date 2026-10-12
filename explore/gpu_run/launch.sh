#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mode="${1:-smoke}"
workers="${WORKERS:-4}"
threads="${TORCH_THREADS:-1}"
hours="${RUN_HOURS:-8}"
case "$mode" in smoke|run) ;; *) echo 'Usage: bash explore/gpu_run/launch.sh smoke|run' >&2; exit 2 ;; esac
[[ "$hours" =~ ^[0-9]+$ && "$hours" -ge 1 && "$hours" -le 10 ]] || { echo 'RUN_HOURS must be 1..10' >&2; exit 2; }
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || exit 2
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$threads"
preflight_args=(--gpu)
if [[ "${ALLOW_PENDING_INPUTS:-0}" == 1 ]]; then
    preflight_args+=(--available-inputs --checksums)
fi
.venv/bin/python explore/gpu_run/preflight.py "${preflight_args[@]}"
run_id="$(.venv/bin/python -c 'import json; print(json.load(open("run-plan.json"))["run_id"])')"
args=(coruna --source gsv --thin 0 --pano-manifest data/coruna/gsv/gpu_manifest.json
      --resume-run "data/coruna/gsv/multicategory/$run_id" --no-legacy-reuse --include-people-pets)
args+=(--torch-threads "$threads")
if [[ "${ALLOW_PENDING_INPUTS:-0}" == 1 ]]; then
    args+=(--input-wait-seconds 300)
fi
if [[ "$mode" == smoke ]]; then
    args+=(--workers 1 --max-new-views 8)
else
    args+=(--workers "$workers")
fi
mkdir -p logs
set +e
timeout --foreground --signal=TERM --kill-after=180 "$((hours * 3600))" \
    .venv/bin/python -u explore/multicategory_detect.py "${args[@]}" 2>&1 | tee -a "logs/google-$mode.log"
status="${PIPESTATUS[0]}"
set -e
.venv/bin/python explore/gpu_run/preflight.py || true
echo "Run exit status: $status. Completed views remain resumable."
echo 'Copy outputs home before destroying the rental. The time limit stops inference, not rental billing.'
exit "$status"
