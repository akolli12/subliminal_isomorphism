"""Graph-conditioned length-generalizing prover (G0,G1 in the input).

One shared model over n=4..MAX_N with the full GMW-GI sequence (delimiters),
paper-proportional split. Trains (CE, phi given), checks usability per n, and
measures top-n recovery at n=4..7 — the "with (G0,G1) conditioning" side of the
Table-3 comparison. The "without" side is the perm-only baseline shared model
(multi_paper_T48000), loaded for reference.
"""

import argparse
import itertools
import json
import os
import random

import torch
import torch.nn.functional as F

from _common import ckpt_path, save_json, RESULTS_DIR
from multi_prover import per_n_counts, D_MODEL, N_HEADS, N_LAYERS, N7_TEST_CAP
from subliminal.configs import SEED_DATA, SEED_TRAIN, SEED_TAU, SEED_EVAL
from subliminal.data import rand_perms, rand_graphs
from subliminal.diagnostics import psi_valid_diag, psi_inv_correct_diag
from subliminal.model import TinyTransformer
from subliminal.multi import specials, IGNORE
from subliminal.multi_graph import (graph_seq_len, graph_multi_layout,
                                    build_graph_multi_batch, GraphMultiContext)
from subliminal.seeding import set_seed
from subliminal.tau import estimate_tau, ExtractorBank, EXTRACTORS
from subliminal.extract import run_extraction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SINGLE = ["single-max-spread raw", "single-max-spread log"]


def make_dataset(ns, counts, max_n, seed):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    nlist, g0s, phis, psis = [], [], [], []
    for n in ns:
        c = counts[n]
        g0 = rand_graphs(c, n, g)
        p = rand_perms(c, n, g)
        q = rand_perms(c, n, g)
        nlist += [n] * c
        g0s += list(g0); phis += list(p); psis += list(q)
    return build_graph_multi_batch(nlist, g0s, phis, psis, max_n)


def train(ns, counts, max_n, steps, batch, lr, seed, path):
    set_seed(seed)
    toks, labels = make_dataset(ns, counts, max_n, SEED_DATA)
    toks, labels = toks.to(DEVICE), labels.to(DEVICE)
    print(f"[graph] dataset {toks.shape[0]} ex, seq_len {toks.shape[1]}, "
          f"counts {counts}", flush=True)
    model = TinyTransformer(specials(max_n)["vocab"], graph_seq_len(max_n),
                            D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        idx = torch.randint(0, toks.shape[0], (batch,), device=DEVICE)
        logits = model(toks[idx])
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               labels[idx].reshape(-1), ignore_index=IGNORE)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0 or step == steps - 1:
            print(f"  step {step:6d}  loss={loss.item():.4f}", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  saved {path}", flush=True)
    return model


def diagnostics(model, ns, max_n, seed):
    g = torch.Generator().manual_seed(seed + 100)
    out = {}
    for n in ns:
        lay = graph_multi_layout(n, max_n)
        ctx = GraphMultiContext(n, max_n, seed)(rand_perms(1000, n, g))
        valid = psi_valid_diag(model, ctx, lay)
        pe, qe = rand_perms(1000, n, g), rand_perms(1000, n, g)
        g0e = rand_graphs(1000, n, g)
        seqs, _ = build_graph_multi_batch([n] * 1000, list(g0e), list(pe),
                                          list(qe), max_n)
        pinv = psi_inv_correct_diag(model, seqs, lay)
        out[n] = {"psi_valid_pct": 100 * valid, "psi_inv_acc_pct": 100 * pinv}
        print(f"  n={n}: psi_valid={100*valid:5.1f}%  psi_inv={100*pinv:5.1f}%",
              flush=True)
    return out


def extract(model, ns, max_n, k1, k2):
    out = {}
    for n in ns:
        lay = graph_multi_layout(n, max_n)
        ctxfn = GraphMultiContext(n, max_n, SEED_TAU)
        tr, tl = estimate_tau(model, lay, k1=k1, k2=k2, seed=SEED_TAU,
                              context_fn=ctxfn)
        bank = ExtractorBank(tr, tl)
        perms = list(itertools.permutations(range(n)))
        if len(perms) > N7_TEST_CAP:
            random.Random(SEED_EVAL).shuffle(perms)
            perms = perms[:N7_TEST_CAP]
        testctx = GraphMultiContext(n, max_n, SEED_EVAL)
        contexts = [testctx(torch.tensor(p).unsqueeze(0))[0] for p in perms]
        res = run_extraction(model, lay, bank, test_contexts=contexts,
                             true_witnesses=perms, k2=k2, chunk=1 << 15,
                             seed=SEED_EVAL)
        best = max(res["extractors"][m]["topn_pct"] for m in EXTRACTORS)
        best_single = max(res["extractors"][m]["topn_pct"] for m in SINGLE)
        out[n] = {"union_topn_pct": res["union"]["topn_pct"],   # ref only (biased)
                  "best_topn_pct": best, "best_single_topn_pct": best_single,
                  "random_topn_pct": res["random_topn_pct"],
                  "num_test": res["num_test"],
                  "extractors": {m: {"topn_pct": res["extractors"][m]["topn_pct"],
                                     "top1_pct": res["extractors"][m]["top1_pct"]}
                                 for m in EXTRACTORS}}
        print(f"  n={n}: union={res['union']['topn_pct']:.1f}%  best={best:.1f}%  "
              f"best-single={best_single:.1f}%  (random "
              f"{res['random_topn_pct']:.3g}%)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=9)
    ap.add_argument("--extract-ns", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--total", type=int, default=48000)
    ap.add_argument("--split", default="exponential")
    ap.add_argument("--exp-base", type=float, default=2.6)
    ap.add_argument("--baseline-json", default="exp2p6_T48000.json")
    ap.add_argument("--tag", default="exp2p6")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k1", type=int, default=128)
    ap.add_argument("--k2", type=int, default=128)
    args = ap.parse_args()

    ns = list(range(4, args.max_n + 1))
    counts = per_n_counts(ns, args.total, args.split, base=args.exp_base)
    path = ckpt_path(f"multi_graph_{args.tag}", args.max_n)
    print(f"[graph] split={args.split} total={args.total} counts={counts}", flush=True)

    model = train(ns, counts, args.max_n, args.steps, args.batch, args.lr,
                  SEED_TRAIN, path)
    print("[graph] diagnostics:", flush=True)
    diag = diagnostics(model, ns, args.max_n, SEED_EVAL)
    print("[graph] extraction (with G0,G1 conditioning):", flush=True)
    ext = extract(model, args.extract_ns, args.max_n, args.k1, args.k2)

    # reference: without conditioning (perm-only baseline shared model)
    without = None
    bpath = os.path.join(RESULTS_DIR, args.baseline_json)
    if os.path.exists(bpath):
        without = json.load(open(bpath)).get("extraction")

    out = {"total": args.total, "split": args.split, "counts": counts,
           "diagnostics": diag, "with_conditioning": ext,
           "without_conditioning": without}
    save_json(out, os.path.join(RESULTS_DIR, f"multi_graph_{args.tag}.json"))
    print("[graph] DONE", flush=True)


if __name__ == "__main__":
    main()
