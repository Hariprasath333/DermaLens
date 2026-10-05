#!/bin/bash
# One-shot runner for Steps 5-7 after post_training (Step 4) finishes.
# All output is teed unbuffered to a log so we see progress in real time.
#
# Usage:  PYTHONPATH=C:\\LesionIQ bash scripts/run_calibration_steps.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
export PYTHONUNBUFFERED=1
# Force UTF-8 stdout/stderr so the many Unicode chars in print statements
# (arrows, box-drawing chars, etc.) don't crash on Windows cp1252 console.
export PYTHONIOENCODING=utf-8

LOG_DIR="backend/output/reports"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo " Step 5: Dirichlet calibration on val (experimental, opt-in)"
echo "================================================================"
python -u -m experimental.calibrate_dirichlet --mode full \
    2>&1 | tee "$LOG_DIR/run_dirichlet.log"

echo ""
echo "================================================================"
echo " Step 7a: Compute effective training prior"
echo "================================================================"
python -u -m backend.classifier.compute_effective_train_prior \
    2>&1 | tee "$LOG_DIR/run_effective_train_prior.log"

echo ""
echo "================================================================"
echo " Step 6: Evaluation with ECE comparison (test set)"
echo "================================================================"
python -u backend/classifier/train_classifier.py --eval-only \
    --checkpoint backend/checkpoints/best_full.pt \
    2>&1 | tee "$LOG_DIR/run_eval.log"

echo ""
echo "================================================================"
echo " Done. Inspect:"
echo "   $LOG_DIR/run_dirichlet.log"
echo "   $LOG_DIR/run_effective_train_prior.log"
echo "   $LOG_DIR/run_eval.log"
echo "   backend/checkpoints/{per_class_temperatures,effective_train_prior}.npy"
echo "   backend/checkpoints/dirichlet_cal.npz"
echo "   backend/output/reports/{classification,calibration}_report.json"
echo "================================================================"
