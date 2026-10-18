#!/usr/bin/env bash
# Compatibility wrapper; the pipeline implementation is shared with Windows.
set -euo pipefail
cd "$(dirname "$0")"
if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x .venv/bin/python ]]; then
    PYTHON=.venv/bin/python
  elif [[ -x .venv/Scripts/python.exe ]]; then
    PYTHON=.venv/Scripts/python.exe
  else
    PYTHON=python3
  fi
fi
exec "$PYTHON" run_pipeline.py "$@"
