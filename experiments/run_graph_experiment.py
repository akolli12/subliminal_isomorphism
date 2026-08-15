"""Non-abstracted graph experiment (the paper's new numbers).

The graph-learned prover is NOT handed the witness phi; it must compute the
isomorphism from (G0, G1) itself, then commit psi. For n = 4, 5, 6 we report:

  1. Witness-finding accuracy, both directions:
       forward (G0,G1) -> phi  with phi(G0)=G1
       reverse (G1,G0) -> phi' with phi'(G1)=G0
     (exact-match to the generating permutation, and valid-isomorphism rate).

  2. Witness-extraction recovery under two conditionings:
       true-phi : context [G0|G1|phi_true], sample psi.  Isolated leakage.
       self-phi : prefix [G0|G1]; the model generates its own phi, then psi.
                  Full self-witnessing pipeline (Assumption 3.2).
"""

import argparse
import math
import os

from _common import save_json, ckpt_path, RESULTS_DIR
from graph_attack import (
    witness_finding_accuracy, tau_true_phi, tau_self_phi,
    build_graph_test_set, run_graph_extraction,
)
from subliminal.configs import CONFIGS, SEED_EVAL, SEED_TAU
from subliminal.layout import graph_layout
from subliminal.tau import ExtractorBank
from subliminal.train import load_prover

# Test-set sizes: full S_n where cheap, capped for n=6.
NUM_TEST = {4: 24, 5: 120, 6: 720}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 5, 6])
    args = ap.parse_args()

    out = {}
    for n in args.ns:
        cfg = CONFIGS[n]
        layout = graph_layout(n)
        model = load_prover(ckpt_path("graph-learned", n), layout,
                            d_model=cfg.d_model, n_heads=cfg.n_heads,
                            n_layers=cfg.n_layers)

        print(f"[graph-exp] n={n}: witness-finding accuracy")
        fwd = witness_finding_accuracy(model, layout, num=2000, seed=SEED_EVAL,
                                       reverse=False)
        rev = witness_finding_accuracy(model, layout, num=2000, seed=SEED_EVAL,
                                       reverse=True)

        test = build_graph_test_set(n, NUM_TEST[n], seed=SEED_EVAL)

        print(f"[graph-exp] n={n}: tau (true-phi)")
        tr, tl = tau_true_phi(model, layout, k1=cfg.k1, k2=cfg.k2, seed=SEED_TAU)
        bank_true = ExtractorBank(tr, tl)
        rec_true = run_graph_extraction(model, layout, bank_true, test,
                                        conditioning="true-phi", k2=cfg.k2,
                                        seed=SEED_EVAL)

        print(f"[graph-exp] n={n}: tau (self-phi)")
        trs, tls = tau_self_phi(model, layout, k1=cfg.k1, k2=cfg.k2, seed=SEED_TAU)
        bank_self = ExtractorBank(trs, tls)
        rec_self = run_graph_extraction(model, layout, bank_self, test,
                                        conditioning="self-phi", k2=cfg.k2,
                                        seed=SEED_EVAL)

        out[n] = {
            "witness_finding": {"forward": fwd, "reverse": rev},
            "recovery_true_phi": rec_true,
            "recovery_self_phi": rec_self,
        }
        print(f"[graph-exp] n={n}  wf_fwd_exact={fwd['exact_match']:.3f} "
              f"wf_rev_exact={rev['exact_match']:.3f}  "
              f"rec_true_union_topn={rec_true['union']['topn_pct']:.2f}  "
              f"rec_self_union_topn={rec_self['union']['topn_pct']:.2f}")
        save_json(out, os.path.join(RESULTS_DIR, "graph_experiment.json"))


if __name__ == "__main__":
    main()
