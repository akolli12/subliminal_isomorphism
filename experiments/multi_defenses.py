"""The two defenses on the single length-generalizing prover (paper split).

Same delimiter-token recipe as the baseline shared model; only training /
inference changes:

  simulator-aligned (§4.1) : ONE new shared model whose psi block is trained to
                             uniform-over-remaining (soft-CE) instead of CE.
  witness-masked   (§4.2)  : REUSE the baseline shared model; zero the phi block
                             at inference so P[psi | x, phi] = P[psi].

Both are trained/evaluated on n=4..9 and attacked on n=4..7 (paper split),
exactly like multi_prover.py. The baseline reference is loaded from the
existing multi_paper_T48000 run.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

from _common import ckpt_path, save_json, RESULTS_DIR
from multi_prover import (per_n_counts, diagnostics, extract, D_MODEL, N_HEADS,
                          N_LAYERS, N7_TEST_CAP)
from subliminal.configs import SEED_DATA, SEED_TRAIN, SEED_EVAL
from subliminal.data import rand_perms
from subliminal.model import TinyTransformer
from subliminal.multi import (multi_seq_len, specials, build_multi_batch_soft)
from subliminal.seeding import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def make_soft_dataset(ns, counts, max_n, seed, psi_mode):
    set_seed(seed)
    g = torch.Generator().manual_seed(seed)
    nlist, phis, psis = [], [], []
    for n in ns:
        p = rand_perms(counts[n], n, g)
        q = rand_perms(counts[n], n, g)
        nlist += [n] * counts[n]
        phis += list(p)
        psis += list(q)
    return build_multi_batch_soft(nlist, phis, psis, max_n, psi_mode)


def train_soft(ns, counts, max_n, steps, batch, lr, seed, path, psi_mode):
    set_seed(seed)
    toks, tgts, masks = make_soft_dataset(ns, counts, max_n, SEED_DATA, psi_mode)
    toks, tgts, masks = toks.to(DEVICE), tgts.to(DEVICE), masks.to(DEVICE)
    print(f"[def] psi_mode={psi_mode} dataset {toks.shape[0]} ex, "
          f"seq_len {toks.shape[1]}", flush=True)
    model = TinyTransformer(specials(max_n)["vocab"], multi_seq_len(max_n),
                            D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        idx = torch.randint(0, toks.shape[0], (batch,), device=DEVICE)
        logits = model(toks[idx])
        logp = F.log_softmax(logits, dim=-1)
        m = masks[idx]
        # soft cross-entropy over loss positions
        per_pos = -(tgts[idx] * logp).sum(-1)
        loss = per_pos[m].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 2000 == 0 or step == steps - 1:
            print(f"  step {step:6d}  loss={loss.item():.4f}", flush=True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  saved {path}", flush=True)
    return model


def load_baseline(max_n, tag="exp2p6_T48000"):
    path = ckpt_path(tag, max_n)
    model = TinyTransformer(specials(max_n)["vocab"], multi_seq_len(max_n),
                            D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-n", type=int, default=9)
    ap.add_argument("--extract-ns", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--total", type=int, default=48000)
    ap.add_argument("--split", default="exponential")
    ap.add_argument("--exp-base", type=float, default=2.6)
    ap.add_argument("--baseline-tag", default="exp2p6_T48000")
    ap.add_argument("--tag", default="exp2p6")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--k1", type=int, default=128)
    ap.add_argument("--k2", type=int, default=128)
    args = ap.parse_args()

    ns = list(range(4, args.max_n + 1))
    counts = per_n_counts(ns, args.total, args.split, base=args.exp_base)
    out = {"total": args.total, "split": args.split, "counts": counts}

    # --- simulator-aligned: new shared model, uniform psi ---
    sa_path = ckpt_path(f"multi_simaligned_{args.tag}", args.max_n)
    print("[def] === simulator-aligned (uniform psi) ===", flush=True)
    sa = train_soft(ns, counts, args.max_n, args.steps, args.batch, args.lr,
                    SEED_TRAIN, sa_path, psi_mode="uniform")
    print("[def] sim-aligned diagnostics:", flush=True)
    out["sim_aligned_diag"] = diagnostics(sa, ns, args.max_n, SEED_EVAL)
    print("[def] sim-aligned extraction:", flush=True)
    out["sim_aligned"] = extract(sa, args.extract_ns, args.max_n, args.k1, args.k2)

    # --- witness-masked: reuse baseline, mask phi at inference ---
    print("[def] === witness-masked (reuse baseline, mask phi) ===", flush=True)
    base = load_baseline(args.max_n, args.baseline_tag)
    out["witness_masked"] = extract(base, args.extract_ns, args.max_n,
                                    args.k1, args.k2, mask_phi=True)

    save_json(out, os.path.join(RESULTS_DIR, f"multi_defenses_{args.tag}.json"))
    print("[def] DONE", flush=True)


if __name__ == "__main__":
    main()
