#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "$0")"
if [[ -x .venv-runtime/bin/python ]]; then
    gesture_python=.venv-runtime/bin/python
elif [[ -x .venv/bin/python ]]; then
    gesture_python=.venv/bin/python
else
    gesture_python=python3
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
exec "$gesture_python" -m app.serve "$@"
