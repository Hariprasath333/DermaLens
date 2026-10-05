#!/usr/bin/env bash
# ============================================================
#  LesionIQ Research Pipeline — Linux/macOS bootstrap
# ============================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

VENV=".venv"
PYEXE="${PYTHON:-python3}"

if [ ! -x "$VENV/bin/python" ]; then
    echo "[LesionIQ] Creating virtual environment at $VENV ..."
    "$PYEXE" -m venv "$VENV"
    echo "[LesionIQ] Installing requirements... (this is one-time, ~5 minutes)"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -r requirements.txt
fi

if [ $# -eq 0 ]; then
    cat <<EOF
Usage:
  ./lesioniq.sh verify   --data-root PATH
  ./lesioniq.sh preprocess --data-root PATH --out-root PATH
  ./lesioniq.sh split    --pre-root PATH --raw-root PATH --out PATH --datasets isic2019 ...
  ./lesioniq.sh train    --variant V0 --split-dir PATH --out-dir PATH
  ./lesioniq.sh evaluate --variant V0 --checkpoint best.pt --split-dir PATH --out-dir PATH
  ./lesioniq.sh full     --config pipeline.yaml

See README.md for full options.
EOF
    exit 0
fi

exec "$VENV/bin/python" run.py "$@"
