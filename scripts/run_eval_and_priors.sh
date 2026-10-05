#!/bin/bash
# After per-class T + effective_train_prior are saved, run:
#   - evaluation with ECE on test set
#   - compute SLD + oracle target priors on test set
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8

LOG_DIR="backend/output/reports"
mkdir -p "$LOG_DIR"

echo "================================================================"
echo " Step 6: Evaluation with ECE comparison (test set)"
echo "================================================================"
python -u backend/classifier/train_classifier.py --eval-only \
    --checkpoint backend/checkpoints/best_full.pt \
    2>&1 | tee "$LOG_DIR/run_eval.log"

echo ""
echo "================================================================"
echo " Step 7b: Compute SLD + oracle target priors (test set)"
echo "================================================================"
python -u -m backend.classifier.compute_target_priors \
    2>&1 | tee "$LOG_DIR/run_target_priors.log"

echo ""
echo "================================================================"
echo " Done."
echo "================================================================"
