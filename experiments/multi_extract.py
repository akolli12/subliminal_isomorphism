"""Extraction-only pass for the trained shared models at chosen n (e.g. n=8).

Loads the already-trained checkpoints and runs the polynomial-time extractor:
  baseline (leak), simulator-aligned, witness-masked (baseline + mask phi),
  and graph-conditioned. n=9 is intentionally separate (overnight).

Usage:
  python experiments/multi_extract.py --ns 8
  python experiments/multi_extract.py --ns 9        # overnight
"""

import argparse
import os

import torch

from _common import ckpt_path, save_json, RESULTS_DIR
from multi_prover import extract as perm_extract, D_MODEL, N_HEADS, N_LAYERS
from multi_graph import extract as graph_extract
from subliminal.model import TinyTransformer
from subliminal.multi import specials, multi_seq_len
from subliminal.multi_graph import graph_seq_len

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_N = 9


def load(tag, seq_len):
    model = TinyTransformer(specials(MAX_N)["vocab"], seq_len,
                            D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path(tag, MAX_N), map_location=DEVICE))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[8])
    ap.add_argument("--k1", type=int, default=128)
    ap.add_argument("--k2", type=int, default=128)
    args = ap.parse_args()

    perm_len = multi_seq_len(MAX_N)
    out = {"ns": args.ns, "k1": args.k1, "k2": args.k2}

    print("[n-extract] baseline (leak)", flush=True)
    base = load("multi_paper_T48000", perm_len)
    out["baseline"] = perm_extract(base, args.ns, MAX_N, args.k1, args.k2)

    print("[n-extract] simulator-aligned", flush=True)
    sa = load("multi_simaligned_paper", perm_len)
    out["sim_aligned"] = perm_extract(sa, args.ns, MAX_N, args.k1, args.k2)

    print("[n-extract] witness-masked (baseline + mask phi)", flush=True)
    out["witness_masked"] = perm_extract(base, args.ns, MAX_N, args.k1, args.k2,
                                         mask_phi=True)

    print("[n-extract] graph-conditioned", flush=True)
    gm = load("multi_graph_paper", graph_seq_len(MAX_N))
    out["graph_with_conditioning"] = graph_extract(gm, args.ns, MAX_N,
                                                   args.k1, args.k2)

    tag = "_".join(str(n) for n in args.ns)
    save_json(out, os.path.join(RESULTS_DIR, f"multi_extract_n{tag}.json"))
    print("[n-extract] DONE", flush=True)


if __name__ == "__main__":
    main()
