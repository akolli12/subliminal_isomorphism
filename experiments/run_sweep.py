"""n-major sweep: for each n, run Tables 1, 2, 3 (and the graph experiment)
end-to-end, then regenerate results/REPORT.md.

Running n-major (all tables for n=4, then n=5, ...) validates the whole
pipeline on small, fast n before committing compute to n=8, 9. Every stage
saves its JSON immediately, so a failure part-way still leaves a coherent,
inspectable partial report. Missing checkpoints are trained on demand.

Usage:
  python experiments/run_sweep.py                 # n = 4..9
  python experiments/run_sweep.py --ns 4 5        # subset
  python experiments/run_sweep.py --skip-graph    # tables only
"""

import argparse
import json
import os
import subprocess
import sys

from _common import ckpt_path, save_json, RESULTS_DIR
from attack import attack_perm_prover
import make_tables

HERE = os.path.dirname(os.path.abspath(__file__))

# Graph provers exist only for these n.
TABLE3_NS = (4, 5)          # graph-given (phi input) vs baseline
GRAPH_EXP_NS = (4, 5, 6)    # graph-learned (self-witnessing)


def ensure(tag, n):
    if os.path.exists(ckpt_path(tag, n)):
        return
    print(f"[sweep] training {tag} n={n}", flush=True)
    subprocess.check_call([sys.executable, os.path.join(HERE, "train_provers.py"),
                           "--tag", tag, "--n", str(n)])


def merge_json(path, key, value):
    data = {}
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data[str(key)] = value
    save_json(data, path)


def do_table1(n):
    ensure("baseline", n)
    print(f"[sweep] Table 1  baseline n={n}", flush=True)
    res = attack_perm_prover("baseline", n)
    save_json(res, os.path.join(RESULTS_DIR, f"table1_baseline_n{n}.json"))


def do_table2(n):
    """Controlled defense experiment: baseline-control, simulator-aligned, and
    witness-masked provers ALL trained at the same (converged) config, so the
    only difference is the defense. The control still leaks; the defenses drop
    to the random floor."""
    from subliminal.configs import defense_uses_default
    control_tag = "baseline" if defense_uses_default(n) else "baseline-dc"

    ensure(control_tag, n)
    print(f"[sweep] Table 2  baseline-control ({control_tag}) n={n}", flush=True)
    res = attack_perm_prover(control_tag, n)
    res["tag"] = "baseline-control"
    save_json(res, os.path.join(RESULTS_DIR, f"table2_baseline-control_n{n}.json"))

    ensure("sim-aligned", n)
    print(f"[sweep] Table 2  sim-aligned n={n}", flush=True)
    res = attack_perm_prover("sim-aligned", n)
    save_json(res, os.path.join(RESULTS_DIR, f"table2_sim-aligned_n{n}.json"))

    print(f"[sweep] Table 2  witness-masked n={n}", flush=True)
    res = attack_perm_prover(control_tag, n, zero_blocks_name="phi")
    res["tag"] = "witness-masked"
    save_json(res, os.path.join(RESULTS_DIR, f"table2_witness-masked_n{n}.json"))


def do_table3(n):
    from run_table3 import with_conditioning, best_single
    ensure("graph-given", n)
    print(f"[sweep] Table 3  conditioning n={n}", flush=True)
    without = attack_perm_prover("baseline", n)
    with_ = with_conditioning(n)
    entry = {
        "with_conditioning_best_single_topn_pct": best_single(with_),
        "without_conditioning_best_single_topn_pct": best_single(without),
        "with_full": with_, "without_full": without,
    }
    merge_json(os.path.join(RESULTS_DIR, "table3_conditioning.json"), n, entry)


def do_graph_experiment(n):
    from graph_attack import (
        witness_finding_accuracy, tau_true_phi, tau_self_phi,
        build_graph_test_set, run_graph_extraction,
    )
    from subliminal.configs import CONFIGS, SEED_EVAL, SEED_TAU
    from subliminal.layout import graph_layout
    from subliminal.tau import ExtractorBank
    from subliminal.train import load_prover
    NUM_TEST = {4: 24, 5: 120, 6: 720}

    ensure("graph-learned", n)
    print(f"[sweep] Graph experiment n={n}", flush=True)
    cfg = CONFIGS[n]
    layout = graph_layout(n)
    model = load_prover(ckpt_path("graph-learned", n), layout,
                        d_model=cfg.d_model, n_heads=cfg.n_heads,
                        n_layers=cfg.n_layers)
    fwd = witness_finding_accuracy(model, layout, num=2000, seed=SEED_EVAL,
                                   reverse=False)
    rev = witness_finding_accuracy(model, layout, num=2000, seed=SEED_EVAL,
                                   reverse=True)
    test = build_graph_test_set(n, NUM_TEST[n], seed=SEED_EVAL)

    tr, tl = tau_true_phi(model, layout, k1=cfg.k1, k2=cfg.k2, seed=SEED_TAU)
    rec_true = run_graph_extraction(model, layout, ExtractorBank(tr, tl), test,
                                    conditioning="true-phi", k2=cfg.k2, seed=SEED_EVAL)
    trs, tls = tau_self_phi(model, layout, k1=cfg.k1, k2=cfg.k2, seed=SEED_TAU)
    rec_self = run_graph_extraction(model, layout, ExtractorBank(trs, tls), test,
                                    conditioning="self-phi", k2=cfg.k2, seed=SEED_EVAL)
    entry = {"witness_finding": {"forward": fwd, "reverse": rev},
             "recovery_true_phi": rec_true, "recovery_self_phi": rec_self}
    merge_json(os.path.join(RESULTS_DIR, "graph_experiment.json"), n, entry)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 5, 6, 7, 8, 9])
    ap.add_argument("--skip-graph", action="store_true")
    args = ap.parse_args()

    for n in args.ns:
        print(f"\n{'='*70}\n[sweep] n = {n}\n{'='*70}", flush=True)
        do_table1(n)
        do_table2(n)
        if n in TABLE3_NS and not args.skip_graph:
            do_table3(n)
        if n in GRAPH_EXP_NS and not args.skip_graph:
            do_graph_experiment(n)
        make_tables.main()          # refresh REPORT.md after every n
        print(f"[sweep] n={n} complete; REPORT.md refreshed", flush=True)
    print("[sweep] DONE", flush=True)


if __name__ == "__main__":
    main()
