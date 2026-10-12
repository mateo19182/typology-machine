#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python3 -c 'import sys; assert sys.version_info >= (3,12), "Use a Python 3.12+ CUDA container"'
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --no-cache-dir -r requirements-gpu.txt
.venv/bin/python explore/gpu_run/preflight.py --checksums --gpu
