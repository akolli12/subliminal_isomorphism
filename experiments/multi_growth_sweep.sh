#!/usr/bin/env bash
# Growth-rate spectrum at fixed total (~48000): constant, polynomial n^1..n^5, factorial.
# Measures phi-psi-inv (functionality) + leak per n. n^5 ~ extractor O(n^5) complexity.
set -e
cd "$(dirname "$0")/.."
PY=/home/akash10/miniconda3/envs/aug-spm/bin/python
S=30000
run(){ echo "==================== $1 ===================="; $PY experiments/multi_prover.py \
  --max-n 9 --extract-ns 4 5 6 7 $2 --tag "$1" --steps $S --extract --k1 128 --k2 128; }
run growth_n0_constant "--split equal --total 48000"
run growth_n1          "--split polynomial --poly-degree 1 --total 48000"
run growth_n2          "--split polynomial --poly-degree 2 --total 48000"
run growth_n3          "--split polynomial --poly-degree 3 --total 48000"
run growth_n4          "--split polynomial --poly-degree 4 --total 48000"
run growth_n5          "--split polynomial --poly-degree 5 --total 48000"
run growth_factorial   "--split factorial --total 48000"
echo "[growthsweep] ALL DONE"
