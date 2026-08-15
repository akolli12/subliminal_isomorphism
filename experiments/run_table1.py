"""Table 1: top-n witness recovery on the baseline prover, n = 4..9.

For each n: train the baseline prover if its checkpoint is missing, estimate
tau, run all six extractors over the test set, and save a per-n JSON. Random
baseline is 100/(n-1)! (top-1) and 100n/n! (top-n).

Usage:
  python experiments/run_table1.py               # all n = 4..9
  python experiments/run_table1.py --ns 4 5 6    # subset
"""

import argparse
import os

from _common import ckpt_path, save_json, RESULTS_DIR
from attack import attack_perm_prover
from train_provers import main as _unused  # noqa: F401  (kept importable)


def ensure_baseline(n):
    import subprocess
    import sys
    if os.path.exists(ckpt_path("baseline", n)):
        return
    print(f"[table1] training baseline n={n}")
    subprocess.check_call([sys.executable,
                           os.path.join(os.path.dirname(__file__), "train_provers.py"),
                           "--tag", "baseline", "--n", str(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9])
    ap.add_argument("--force-tau", action="store_true")
    args = ap.parse_args()

    for n in args.ns:
        ensure_baseline(n)
        print(f"[table1] attacking baseline n={n}")
        res = attack_perm_prover("baseline", n, force_tau=args.force_tau)
        save_json(res, os.path.join(RESULTS_DIR, f"table1_baseline_n{n}.json"))


if __name__ == "__main__":
    main()
