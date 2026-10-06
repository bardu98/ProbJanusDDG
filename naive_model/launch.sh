#!/usr/bin/env bash
set -euo pipefail
RUN_SOURCE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
exec "${PYTHON:-python}" -u "$RUN_SOURCE/run.py" "$@"
