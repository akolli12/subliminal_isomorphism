"""Table 5: output-validity diagnostics on the baseline prover, n = 4..9.

  psi_valid            : fraction of 5000 unconstrained samples that are valid
                         permutations.
  psi_inv_correct_psi  : fraction of 1000 teacher-forced instances whose psi_inv
                         argmax equals argsort(psi) at every position.
"""

import argparse
import os

import torch

from _common import ckpt_path, save_json, RESULTS_DIR
from subliminal.configs import CONFIGS, SEED_EVAL
from subliminal.data import make_perm_dataset, build_perm_sequences, rand_perms
from subliminal.diagnostics import psi_valid_diag, psi_inv_correct_diag
from subliminal.layout import perm_layout
from subliminal.seeding import set_seed
from subliminal.train import load_prover


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9])
    args = ap.parse_args()

    results = {}
    for n in args.ns:
        cfg = CONFIGS[n]
        layout = perm_layout(n)
        model = load_prover(ckpt_path("baseline", n), layout,
                            d_model=cfg.d_model, n_heads=cfg.n_heads,
                            n_layers=cfg.n_layers)
        set_seed(SEED_EVAL)
        g = torch.Generator().manual_seed(SEED_EVAL + 100)
        contexts = rand_perms(5000, n, g)                    # 5000 phis
        valid = psi_valid_diag(model, contexts, layout)

        phi, psi = make_perm_dataset(1000, n, SEED_EVAL + 200)
        seqs = build_perm_sequences(phi, psi)
        correct = psi_inv_correct_diag(model, seqs, layout)

        results[n] = {"psi_valid_pct": 100.0 * valid,
                      "psi_inv_correct_pct": 100.0 * correct}
        print(f"[table5] n={n}  psi_valid={100*valid:.2f}%  "
              f"psi_inv_correct={100*correct:.2f}%")
    save_json(results, os.path.join(RESULTS_DIR, "table5_diagnostics.json"))


if __name__ == "__main__":
    main()
