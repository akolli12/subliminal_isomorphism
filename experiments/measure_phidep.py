"""Measure the phi-dependence and non-uniformity of psi for the shared models.

phi-dependence = max over (i,v) of the spread of P[psi(i)=v | phi] across several
phi. It is the artifact-free leak metric (the paper's own, §3.5): the extractor
can only recover phi if this is nonzero. Non-uniformity = max|P - 1/n|.

Writes results/multi_phidep.json:
  baseline (leak)   : phi-dep and non-uniformity both large,
  simulator-aligned : both ~ sampling-noise floor (psi ~ uniform, phi-independent),
  witness-masked    : phi-dep ~ floor (channel closed) but non-uniformity > 0 (biased).
"""

import argparse
import os

import torch

from _common import ckpt_path, save_json, RESULTS_DIR
from multi_prover import D_MODEL, N_HEADS, N_LAYERS
from subliminal.data import rand_perms
from subliminal.model import TinyTransformer
from subliminal.multi import specials, multi_seq_len, multi_layout, MultiContext
from subliminal.sample import sample_psi, marginal_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_N = 9


def load(tag):
    m = TinyTransformer(specials(MAX_N)["vocab"], multi_seq_len(MAX_N),
                        D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path(tag, MAX_N), map_location=DEVICE))
    m.eval()
    return m


@torch.no_grad()
def stats(model, n, K, n_phi, zero_blocks=()):
    g = torch.Generator().manual_seed(1)
    Ms = []
    for p in rand_perms(n_phi, n, g):
        ctx = MultiContext(n, MAX_N)(p.unsqueeze(0).repeat(K, 1))
        Ms.append(marginal_matrix(sample_psi(model, ctx, multi_layout(n, MAX_N),
                                             valid=True, zero_blocks=zero_blocks), n))
    Ms = torch.stack(Ms)
    phidep = (Ms.max(0).values - Ms.min(0).values).abs().max().item()
    nonunif = (Ms - 1.0 / n).abs().max().item()
    return {"phi_dep": phidep, "nonunif": nonunif}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[4, 5, 6, 7])
    ap.add_argument("--K", type=int, default=6000)
    ap.add_argument("--n-phi", type=int, default=6)
    args = ap.parse_args()

    base = load("multi_paper_T48000")
    sa = load("multi_simaligned_paper")
    out = {"K": args.K, "n_phi": args.n_phi, "by_n": {}}
    for n in args.ns:
        phiblk = multi_layout(n, MAX_N)["phi"]
        out["by_n"][n] = {
            "baseline": stats(base, n, args.K, args.n_phi),
            "sim_aligned": stats(sa, n, args.K, args.n_phi),
            "witness_masked": stats(base, n, args.K, args.n_phi, (phiblk,)),
        }
        r = out["by_n"][n]
        print(f"n={n}: base phi-dep {r['baseline']['phi_dep']:.3f} | "
              f"sim {r['sim_aligned']['phi_dep']:.3f} | "
              f"mask {r['witness_masked']['phi_dep']:.3f} "
              f"(mask nonunif {r['witness_masked']['nonunif']:.3f})", flush=True)
    save_json(out, os.path.join(RESULTS_DIR, "multi_phidep.json"))


if __name__ == "__main__":
    main()
