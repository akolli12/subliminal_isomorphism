"""Train every checkpoint the experiments need, skipping any that exist.

  baseline       n = 4..9   (Tables 1, 2 witness-masked, 5)
  sim-aligned    n = 4..6   (Table 2)
  graph-given    n = 4, 5   (Table 3)
  graph-learned  n = 4, 5, 6 (non-abstracted experiment)

Run in the background and tail the log:
  python experiments/train_all.py > results/train_all.log 2>&1
"""

import os
import subprocess
import sys

from _common import ckpt_path

HERE = os.path.dirname(os.path.abspath(__file__))

JOBS = (
    [("baseline", n) for n in (4, 5, 6, 7, 8, 9)]
    + [("sim-aligned", n) for n in (4, 5, 6)]
    + [("graph-given", n) for n in (4, 5)]
    + [("graph-learned", n) for n in (4, 5, 6)]
)


def main():
    for tag, n in JOBS:
        if os.path.exists(ckpt_path(tag, n)):
            print(f"[skip] {tag} n={n} (checkpoint exists)", flush=True)
            continue
        print(f"[run] {tag} n={n}", flush=True)
        subprocess.check_call(
            [sys.executable, os.path.join(HERE, "train_provers.py"),
             "--tag", tag, "--n", str(n)])
    print("[done] all checkpoints trained", flush=True)


if __name__ == "__main__":
    main()
