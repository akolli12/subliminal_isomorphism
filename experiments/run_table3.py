"""Table 3: recovery with vs without (G0, G1) conditioning, n = 4, 5.

'Without' is the perm-only baseline prover (witness-only input, used in the
main paper). 'With' is the graph-given prover (full GMW-GI input). The metric
is the top-n recovery rate of the best single-witness extractor. Differences
within sampling noise indicate the (G0,G1)-abstraction does not change the
leakage measured by the extractors.
"""

import argparse
import os

import torch

from _common import ckpt_path, save_json, full_or_sample_perms, RESULTS_DIR
from attack import attack_perm_prover
from graph_attack import (
    tau_true_phi, build_graph_test_set, run_graph_extraction,
)
from subliminal.configs import CONFIGS, SEED_EVAL, SEED_TAU
from subliminal.layout import graph_layout
from subliminal.tau import ExtractorBank
from subliminal.train import load_prover

SINGLE = ["single-max-spread raw", "single-max-spread log"]


def best_single(res):
    return max(res["extractors"][m]["topn_pct"] for m in SINGLE)


def with_conditioning(n):
    cfg = CONFIGS[n]
    layout = graph_layout(n)
    model = load_prover(ckpt_path("graph-given", n), layout,
                        d_model=cfg.d_model, n_heads=cfg.n_heads,
                        n_layers=cfg.n_layers)
    tau_raw, tau_log = tau_true_phi(model, layout, k1=cfg.k1, k2=cfg.k2,
                                    seed=SEED_TAU)
    bank = ExtractorBank(tau_raw, tau_log)
    test = build_graph_test_set(n, num=min(720, _full(n)), seed=SEED_EVAL)
    return run_graph_extraction(model, layout, bank, test,
                                conditioning="true-phi", k2=cfg.k2, seed=SEED_EVAL)


def _full(n):
    import math
    return math.factorial(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 5])
    args = ap.parse_args()

    out = {}
    for n in args.ns:
        print(f"[table3] n={n}: without-conditioning (baseline)")
        without = attack_perm_prover("baseline", n)
        print(f"[table3] n={n}: with-conditioning (graph-given)")
        with_ = with_conditioning(n)
        out[n] = {
            "with_conditioning_best_single_topn_pct": best_single(with_),
            "without_conditioning_best_single_topn_pct": best_single(without),
            "with_full": with_, "without_full": without,
        }
        print(f"[table3] n={n}  with={out[n]['with_conditioning_best_single_topn_pct']:.2f}  "
              f"without={out[n]['without_conditioning_best_single_topn_pct']:.2f}")
    save_json(out, os.path.join(RESULTS_DIR, "table3_conditioning.json"))


if __name__ == "__main__":
    main()
