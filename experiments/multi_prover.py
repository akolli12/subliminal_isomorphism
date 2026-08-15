"""One length-generalizing prover over n=4..MAX_N (delimiter tokens).

Trains a single model on a mixed-n dataset, then (per n) checks it is a usable
prover — ψ-valid rate and ψ⁻¹ accuracy — and measures top-n witness recovery by
the polynomial-time extractor. The model always TRAINS on all of n=4..MAX_N;
extraction is only run on --extract-ns (n=8,9 are slow and deferred).

Dataset budget is split across n by --split:
  equal      : same count per n
  linear     : proportional to n
  paper      : proportional to the paper's per-n Table-4 sizes (grows with n)
  factorial  : proportional to n! (heavily favors large n; starves small n)

Usage:
  python experiments/multi_prover.py --total 48000 --split equal --steps 40000 --extract
  python experiments/multi_prover.py --total 48000 --split paper  --steps 40000 --extract
"""

import argparse
import itertools
import math
import os
import random

import torch
import torch.nn.functional as F

from _common import ckpt_path, save_json, RESULTS_DIR
from subliminal.configs import SEED_DATA, SEED_TRAIN, SEED_TAU, SEED_EVAL
from subliminal.data import rand_perms
from subliminal.diagnostics import psi_valid_diag, psi_inv_correct_diag
from subliminal.model import TinyTransformer
from subliminal.multi import (multi_layout, multi_seq_len, specials,
                              build_multi_batch, MultiContext, IGNORE)
from subliminal.seeding import set_seed
from subliminal.tau import estimate_tau, ExtractorBank, EXTRACTORS
from subliminal.extract import run_extraction

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D_MODEL, N_HEADS, N_LAYERS = 256, 4, 8
PAPER_SIZE = {4: 100, 5: 100, 6: 200, 7: 500, 8: 10000, 9: 20000}
N7_TEST_CAP = 720


def per_n_counts(ns, total, split, floor=200, base=2.0, poly_degree=1):
    if split == "equal":
        w = [1] * len(ns)
    elif split == "linear":
        w = list(ns)
    elif split == "paper":
        w = [PAPER_SIZE[n] for n in ns]
    elif split == "factorial":
        w = [math.factorial(n) for n in ns]
    elif split == "exponential":
        w = [base ** n for n in ns]     # geometric; larger base => small n lower
    elif split == "polynomial":
        w = [n ** poly_degree for n in ns]   # n^d; d=5 ~ extractor O(n^5) cost
    else:
        raise ValueError(split)
    s = sum(w)
    return {n: max(floor, round(total * wi / s)) for n, wi in zip(ns, w)}


def make_mixed_dataset(ns, counts, max_n, seed):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    nlist, phis, psis = [], [], []
    for n in ns:
        p = rand_perms(counts[n], n, g)
        q = rand_perms(counts[n], n, g)
        nlist += [n] * counts[n]
        phis += list(p)
        psis += list(q)
    return build_multi_batch(nlist, phis, psis, max_n)


def train(ns, counts, max_n, steps, batch, lr, seed, path):
    set_seed(seed)
    toks, labels = make_mixed_dataset(ns, counts, max_n, SEED_DATA)
    toks, labels = toks.to(DEVICE), labels.to(DEVICE)
    print(f"[multi] dataset {toks.shape[0]} ex, seq_len {toks.shape[1]}, "
          f"vocab {specials(max_n)['vocab']}, counts {counts}", flush=True)
    model = TinyTransformer(specials(max_n)["vocab"], multi_seq_len(max_n),
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


@torch.no_grad()
def _block_acc(model, seqs, lay, block, max_n):
    blk = lay[block]
    logits = model(seqs.to(DEVICE))
    pred = logits[:, blk.start - 1:blk.stop - 1, :lay.n].argmax(-1)
    return (pred == seqs[:, blk].to(DEVICE)).all(dim=1).float().mean().item()


def diagnostics(model, ns, max_n, seed):
    g = torch.Generator().manual_seed(seed + 100)
    out = {}
    for n in ns:
        lay = multi_layout(n, max_n)
        ctx = MultiContext(n, max_n)(rand_perms(2000, n, g))
        valid = psi_valid_diag(model, ctx, lay)
        pe, qe = rand_perms(1000, n, g), rand_perms(1000, n, g)
        seqs, _ = build_multi_batch([n] * 1000, list(pe), list(qe), max_n)
        pinv = psi_inv_correct_diag(model, seqs, lay)
        ppi = _block_acc(model, seqs, lay, "phi_psi_inv", max_n)   # c=1 response
        out[n] = {"psi_valid_pct": 100 * valid, "psi_inv_acc_pct": 100 * pinv,
                  "phi_psi_inv_acc_pct": 100 * ppi}
        print(f"  n={n}: psi_valid={100*valid:5.1f}%  psi_inv={100*pinv:5.1f}%  "
              f"phi_psi_inv={100*ppi:5.1f}%", flush=True)
    return out


def extract(model, ns, max_n, k1, k2, mask_phi=False, test_cap=N7_TEST_CAP):
    out = {}
    for n in ns:
        lay = multi_layout(n, max_n)
        ctxfn = MultiContext(n, max_n)
        zb = (lay["phi"],) if mask_phi else ()      # witness-masking defense (§4.2)
        tr, tl = estimate_tau(model, lay, k1=k1, k2=k2, seed=SEED_TAU,
                              context_fn=ctxfn, zero_blocks=zb)
        bank = ExtractorBank(tr, tl)
        perms = list(itertools.permutations(range(n)))
        if len(perms) > test_cap:
            random.Random(SEED_EVAL).shuffle(perms)
            perms = perms[:test_cap]
        contexts = [ctxfn(torch.tensor(p).unsqueeze(0))[0] for p in perms]
        res = run_extraction(model, lay, bank, test_contexts=contexts,
                             true_witnesses=perms, k2=k2, chunk=1 << 15,
                             seed=SEED_EVAL, zero_blocks=zb)
        best = max(res["extractors"][m]["topn_pct"] for m in EXTRACTORS)
        out[n] = {"union_topn_pct": res["union"]["topn_pct"],   # ref only (biased)
                  "best_topn_pct": best,
                  "random_topn_pct": res["random_topn_pct"],
                  "num_test": res["num_test"],
                  # per-extractor top-n AND top-1 (the correct, unbiased metric)
                  "extractors": {m: {"topn_pct": res["extractors"][m]["topn_pct"],
                                     "top1_pct": res["extractors"][m]["top1_pct"]}
                                 for m in EXTRACTORS}}
        print(f"  n={n}: union={res['union']['topn_pct']:.1f}%  best={best:.1f}%  "
              f"(random {res['random_topn_pct']:.3g}%, "
              f"{res['union']['topn_pct']/res['random_topn_pct']:.0f}x)", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=9)
    ap.add_argument("--extract-ns", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--total", type=int, default=48000)
    ap.add_argument("--split", choices=["equal", "linear", "paper", "factorial", "exponential"],
                    default="equal")
    ap.add_argument("--min-per-n", type=int, default=200,
                    help="floor on examples per n (anchors starved mid-range n)")
    ap.add_argument("--exp-base", type=float, default=2.0,
                    help="base for --split exponential (larger => small n lower)")
    ap.add_argument("--counts", type=str, default=None,
                    help="explicit per-n counts, comma-separated for n=4..max_n "
                         "(overrides --split), e.g. 200,500,3000,5000,15000,30000")
    ap.add_argument("--tag", type=str, default=None, help="checkpoint/result tag")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k1", type=int, default=128)
    ap.add_argument("--k2", type=int, default=128)
    ap.add_argument("--extract", action="store_true")
    args = ap.parse_args()

    ns = list(range(4, args.max_n + 1))
    if args.counts:
        vals = [int(x) for x in args.counts.split(",")]
        assert len(vals) == len(ns), f"need {len(ns)} counts for n={ns}"
        counts = dict(zip(ns, vals))
    else:
        counts = per_n_counts(ns, args.total, args.split, floor=args.min_per_n, base=args.exp_base)
    tag = args.tag or (
        f"multi_{args.split}_T{args.total}"
        + (f"_floor{args.min_per_n}" if args.min_per_n != 200 else ""))
    path = ckpt_path(tag, args.max_n)

    print(f"[multi] split={args.split} total={args.total} counts={counts}", flush=True)
    model = train(ns, counts, args.max_n, args.steps, args.batch, args.lr,
                  SEED_TRAIN, path)

    print("[multi] functionality diagnostics (all trained n):", flush=True)
    diag = diagnostics(model, ns, args.max_n, SEED_EVAL)
    out = {"tag": tag, "split": args.split, "total": args.total,
           "counts": counts, "train_ns": ns, "diagnostics": diag}
    if args.extract:
        print(f"[multi] extraction (n={args.extract_ns}):", flush=True)
        out["extraction"] = extract(model, args.extract_ns, args.max_n,
                                    args.k1, args.k2)
    save_json(out, os.path.join(RESULTS_DIR, f"{tag}.json"))
    print(f"[multi] DONE {tag}", flush=True)


if __name__ == "__main__":
    main()
