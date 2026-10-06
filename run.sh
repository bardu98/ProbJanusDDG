#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="$PROJECT_DIR/.matplotlib"
mkdir -p "$MPLCONFIGDIR"
"${PYTHON:-python3}" -u pipeline/run.py "$@" 2>&1 | tee -a pipeline/training.log
