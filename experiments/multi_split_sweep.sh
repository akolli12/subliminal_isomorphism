#!/usr/bin/env bash
# Sweep candidate MONOTONE splits: measure phi-psi-inv (functionality) + leak.
# Goal: full phi o psi^-1 at every n while keeping small-n leak strong.
set -e
cd "$(dirname "$0")/.."
PY=/home/akash10/miniconda3/envs/aug-spm/bin/python
S=30000
run(){ echo "==================== $1 ===================="; $PY experiments/multi_prover.py \
  --max-n 9 --extract-ns 4 5 6 7 --counts "$2" --tag "$1" --steps $S --extract --k1 128 --k2 128; }

run steepA_n6_2000 "200,500,2000,4000,15000,30000"
run steepB_n6_3000 "200,800,3000,6000,15000,28000"
$PY experiments/multi_prover.py --max-n 9 --extract-ns 4 5 6 7 --split factorial \
  --total 48000 --tag factorial_T48000 --steps $S --extract --k1 128 --k2 128
echo "[splitsweep] ALL DONE"
