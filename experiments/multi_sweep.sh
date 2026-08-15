#!/usr/bin/env bash
# Sweep dataset size x split strategy for the single length-generalizing prover.
# Trains on n=4..9 each time; extracts (leak) on n=4..7 (n=8,9 deferred).
set -e
cd "$(dirname "$0")/.."
PY=/home/akash10/miniconda3/envs/aug-spm/bin/python
STEPS=30000

run () {  # $1=split  $2=total
  echo "==================== split=$1 total=$2 ===================="
  $PY experiments/multi_prover.py --max-n 9 --extract-ns 4 5 6 7 \
      --total "$2" --split "$1" --steps $STEPS --extract --k1 128 --k2 128
}

# size effect (equal split)
run equal 24000
run equal 48000
# split effect (fixed total)
run linear 48000
run paper 48000
run factorial 48000

echo "[sweep] ALL DONE"
