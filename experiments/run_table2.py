"""Table 2: top-n witness recovery on the two defended provers, n = 4..6.

Simulator-aligned defense (Section 4.1): a separate prover trained with the
uniform-KL psi loss. Witness-masked defense (Section 4.2): the baseline prover
with the phi block zeroed while sampling psi (no separate training). Both
should collapse recovery to the random baseline.

Usage:
  python experiments/run_table2.py            # all n = 4..6
"""

import argparse
import os
import subprocess
import sys

from _common import ckpt_path, save_json, RESULTS_DIR
from attack import attack_perm_prover


def ensure(tag, n):
    if os.path.exists(ckpt_path(tag, n)):
        return
    print(f"[table2] training {tag} n={n}")
    subprocess.check_call([sys.executable,
                           os.path.join(os.path.dirname(__file__), "train_provers.py"),
                           "--tag", tag, "--n", str(n)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 5, 6])
    args = ap.parse_args()

    for n in args.ns:
        # simulator-aligned prover
        ensure("sim-aligned", n)
        print(f"[table2] attacking sim-aligned n={n}")
        res = attack_perm_prover("sim-aligned", n)
        save_json(res, os.path.join(RESULTS_DIR, f"table2_sim-aligned_n{n}.json"))

        # witness-masked = baseline prover, phi zeroed at inference
        ensure("baseline", n)
        print(f"[table2] attacking witness-masked (baseline+maskphi) n={n}")
        res = attack_perm_prover("baseline", n, zero_blocks_name="phi")
        res["tag"] = "witness-masked"
        save_json(res, os.path.join(RESULTS_DIR, f"table2_witness-masked_n{n}.json"))


if __name__ == "__main__":
    main()
