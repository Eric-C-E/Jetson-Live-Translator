#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_EXTRA_ARGS="${PIP_EXTRA_ARGS:-}"

if [[ -d "$VENV_DIR" ]]; then
  echo "Using existing venv at $VENV_DIR"
else
  echo "Creating venv at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install -U pip setuptools wheel
python -m pip install $PIP_EXTRA_ARGS -r "$ROOT_DIR/requirements.txt"

echo "Venv ready. Activate with: source \"$VENV_DIR/bin/activate\""
