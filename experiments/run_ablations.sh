#!/usr/bin/env bash
# Wait for the main sweep to finish, then run the two memorization ablations
# (they need the baseline checkpoints + cached tau the sweep produces), and
# refresh REPORT.md.
set -e
cd "$(dirname "$0")/.."
PY=/home/akash10/miniconda3/envs/aug-spm/bin/python

echo "[ablations] waiting for main sweep to finish..."
until grep -q "\[sweep\] DONE" results/sweep.log 2>/dev/null; do
  if ! pgrep -f run_sweep.py >/dev/null && ! grep -q "\[sweep\] DONE" results/sweep.log; then
    echo "[ablations] sweep process gone without DONE — aborting"; exit 1
  fi
  sleep 20
done

echo "[ablations] recovery vs distance-to-training (correlational leak; n=7,8)"
$PY experiments/ablation_correlation.py --ns 7 8

echo "[ablations] overtraining curve (n=5)"
$PY experiments/ablation_overtraining.py --which both

echo "[ablations] collating REPORT.md"
$PY experiments/make_tables.py
echo "[ablations] DONE"
