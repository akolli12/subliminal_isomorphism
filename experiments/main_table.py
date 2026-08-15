"""MAIN RESULTS TABLE orchestrator (one seed at a time).

Grid: {2 splits} x {seed} x {perm-only, graph-fixG0, graph-fixG1}
      x {baseline, simulator-aligned, witness-masked} x {n=4..9}
      x {6 extractors + union}, plus functionality diagnostics per model.

Per split it trains 4 models: perm-baseline, perm-sim-aligned, graph-baseline,
graph-sim-aligned (witness-masked reuses each baseline with phi masked). Then:
  - perm-only extraction: global tau, up to PERM_TESTS test witnesses.
  - graph extraction: per-instance tau (fix-G0 and fix-G1), GRAPH_INSTANCES
    instances (the expensive, faithful non-abstracted attack).
Results are saved incrementally to results/main_table_seed{seed}_{split}.json.

Usage:
  python experiments/main_table.py --seed 0 --splits exponential flat
  python experiments/main_table.py --seed 0 --splits exponential --smoke   # quick check
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from _common import ckpt_path, save_json, RESULTS_DIR
from multi_prover import (per_n_counts, train as perm_train,
                          diagnostics as perm_diag, extract as perm_extract,
                          D_MODEL, N_HEADS, N_LAYERS)
from multi_defenses import train_soft as perm_train_soft
from multi_graph import (train as graph_train, diagnostics as graph_diag)
from graph_perinstance import attack_graph_perinstance
from subliminal.configs import SEED_DATA, SEED_TRAIN, SEED_EVAL
from subliminal.data import rand_perms, rand_graphs
from subliminal.model import TinyTransformer
from subliminal.multi import specials, IGNORE
from subliminal.multi_graph import graph_seq_len, build_graph_multi_batch_soft
from subliminal.seeding import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_N = 9
NS = [4, 5, 6, 7, 8, 9]


def graph_train_soft(ns, counts, max_n, steps, batch, lr, seed, path, psi_mode):
    """Train a graph-conditioned prover with soft targets (uniform psi = defense)."""
    set_seed(seed)
    g = torch.Generator().manual_seed(SEED_DATA)
    nlist, g0s, phis, psis = [], [], [], []
    for n in ns:
        c = counts[n]
        g0s += list(rand_graphs(c, n, g)); phis += list(rand_perms(c, n, g))
        psis += list(rand_perms(c, n, g)); nlist += [n] * c
    toks, tgts, masks = build_graph_multi_batch_soft(nlist, g0s, phis, psis,
                                                     max_n, psi_mode)
    toks, tgts, masks = toks.to(DEVICE), tgts.to(DEVICE), masks.to(DEVICE)
    model = TinyTransformer(specials(max_n)["vocab"], graph_seq_len(max_n),
                            D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        idx = torch.randint(0, toks.shape[0], (batch,), device=DEVICE)
        logp = F.log_softmax(model(toks[idx]), dim=-1)
        m = masks[idx]
        loss = (-(tgts[idx] * logp).sum(-1))[m].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0 or step == steps - 1:
            print(f"    [graph-{psi_mode}] step {step} loss {loss.item():.4f}", flush=True)
    torch.save(model.state_dict(), path)
    return model


def run_split(split, seed, steps, graph_instances, perm_tests, k1, k2, gk1, gk2,
              skip_graph=False):
    counts = per_n_counts(NS, 48000, split, base=2.6)
    tag = f"seed{seed}_{split}"
    outpath = os.path.join(RESULTS_DIR, f"main_table_{tag}.json")
    out = {"split": split, "seed": seed, "counts": counts, "ns": NS,
           "perm_tests": perm_tests, "graph_instances": graph_instances,
           "diagnostics": {}, "perm": {}, "graph": {}}

    def flush():
        save_json(out, outpath)

    print(f"\n===== split={split} seed={seed} counts={counts} =====", flush=True)

    # ---- train ----
    loss_ce = {"psi": "ce", "psi_inv": "ce", "phi_psi_inv": "ce"}
    print("[main] train perm baseline", flush=True)
    pb = perm_train(NS, counts, MAX_N, steps, 64, 3e-4, SEED_TRAIN + seed,
                    ckpt_path(f"mt_perm_base_{tag}", MAX_N))
    print("[main] train perm sim-aligned", flush=True)
    ps = perm_train_soft(NS, counts, MAX_N, steps, 64, 3e-4, SEED_TRAIN + seed,
                         ckpt_path(f"mt_perm_sim_{tag}", MAX_N), psi_mode="uniform")
    print("[main] train graph baseline", flush=True)
    gb = graph_train(NS, counts, MAX_N, steps, 64, 3e-4, SEED_TRAIN + seed,
                     ckpt_path(f"mt_graph_base_{tag}", MAX_N))
    print("[main] train graph sim-aligned", flush=True)
    gs = graph_train_soft(NS, counts, MAX_N, steps, 64, 3e-4, SEED_TRAIN + seed,
                          ckpt_path(f"mt_graph_sim_{tag}", MAX_N), psi_mode="uniform")

    # ---- functionality diagnostics (completeness of every prover) ----
    print("[main] diagnostics", flush=True)
    out["diagnostics"]["perm_baseline"] = perm_diag(pb, NS, MAX_N, SEED_EVAL)
    out["diagnostics"]["perm_sim"] = perm_diag(ps, NS, MAX_N, SEED_EVAL)
    out["diagnostics"]["graph_baseline"] = graph_diag(gb, NS, MAX_N, SEED_EVAL)
    out["diagnostics"]["graph_sim"] = graph_diag(gs, NS, MAX_N, SEED_EVAL)
    flush()

    # ---- perm-only extraction (global tau; baseline / sim / masked) ----
    for prover, model, mask in [("baseline", pb, False), ("sim_aligned", ps, False),
                                ("witness_masked", pb, True)]:
        print(f"[main] perm extract: {prover}", flush=True)
        out["perm"][prover] = perm_extract(model, NS, MAX_N, k1, k2,
                                           mask_phi=mask, test_cap=perm_tests)
        flush()

    # ---- graph extraction (per-instance tau; fix-G0 and fix-G1) ----
    if skip_graph:
        print("[main] skipping slow graph extraction (--skip-graph); use "
              "graph_reextract.py for the batched 100-instance version", flush=True)
        print(f"[main] DONE (train+perm only) split={split} seed={seed}", flush=True)
        return
    for prover, model, mask in [("baseline", gb, False), ("sim_aligned", gs, False),
                                ("witness_masked", gb, True)]:
        for fix in ["G0", "G1"]:
            key = f"{prover}_fix{fix}"
            out["graph"][key] = {}
            for n in NS:
                print(f"[main] graph extract: {key} n={n}", flush=True)
                out["graph"][key][n] = attack_graph_perinstance(
                    model, n, MAX_N, fix=fix, num_instances=graph_instances,
                    k1=gk1, k2=gk2, seed=SEED_EVAL, mask_phi=mask)
                flush()
    print(f"[main] DONE split={split} seed={seed}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--splits", nargs="+", default=["exponential", "flat"])
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--graph-instances", type=int, default=60)
    ap.add_argument("--perm-tests", type=int, default=2000)
    ap.add_argument("--k1", type=int, default=128)
    ap.add_argument("--k2", type=int, default=128)
    ap.add_argument("--gk1", type=int, default=48)
    ap.add_argument("--gk2", type=int, default=48)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-graph", action="store_true",
                    help="train+perm only; graph via graph_reextract.py (batched)")
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.graph_instances, args.perm_tests = 300, 4, 50
        args.k1 = args.k2 = args.gk1 = args.gk2 = 16
        global NS
        NS = [4, 5]
    split_map = {"flat": "equal", "exponential": "exponential"}
    for s in args.splits:
        run_split(split_map.get(s, s), args.seed, args.steps,
                  args.graph_instances, args.perm_tests, args.k1, args.k2,
                  args.gk1, args.gk2, skip_graph=args.skip_graph)


if __name__ == "__main__":
    main()
