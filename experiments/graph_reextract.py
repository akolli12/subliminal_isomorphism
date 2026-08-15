"""Re-run ONLY the graph extraction with more instances + the batched attack,
using the already-trained checkpoints. Overwrites the graph cells in the
existing main_table_seed0_{split}.json (incrementally).
"""
import argparse
import json
import os

import torch

from _common import ckpt_path, RESULTS_DIR
from multi_prover import D_MODEL, N_HEADS, N_LAYERS
from graph_perinstance import attack_graph_perinstance_batched
from subliminal.model import TinyTransformer
from subliminal.multi import specials
from subliminal.multi_graph import graph_seq_len

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_N = 9
NS = [4, 5, 6, 7, 8, 9]


def load(tag):
    m = TinyTransformer(specials(MAX_N)["vocab"], graph_seq_len(MAX_N),
                        D_MODEL, N_HEADS, N_LAYERS).to(DEVICE)
    m.load_state_dict(torch.load(ckpt_path(tag, MAX_N), map_location=DEVICE))
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["exponential", "equal"])
    ap.add_argument("--instances", type=int, default=100)
    ap.add_argument("--k1", type=int, default=48)
    ap.add_argument("--k2", type=int, default=48)
    ap.add_argument("--ns", type=int, nargs="+", default=NS)
    ap.add_argument("--provers", nargs="+",
                    default=["baseline", "sim_aligned", "witness_masked"],
                    choices=["baseline", "sim_aligned", "witness_masked"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Load both splits' checkpoints + json up front.
    splits = {}
    for split in args.splits:
        tag = f"seed{args.seed}_{split}"
        jf = os.path.join(RESULTS_DIR, f"main_table_{tag}.json")
        d = json.load(open(jf))
        d["graph_instances"] = args.instances
        d["graph_batched"] = True
        splits[split] = dict(tag=tag, jf=jf, d=d,
                             gb=load(f"mt_graph_base_{tag}"),
                             gs=load(f"mt_graph_sim_{tag}"))

    # Prover-outer ordering: the headline "baseline" leak finishes for BOTH
    # splits before the two defenses, so the most-important numbers land first.
    provers = [("baseline", "gb", False),
               ("sim_aligned", "gs", False),
               ("witness_masked", "gb", True)]
    provers = [p for p in provers if p[0] in args.provers]
    for prover, model_key, mask in provers:
        for split in args.splits:
            S = splits[split]
            model, d, jf = S[model_key], S["d"], S["jf"]
            for fix in ["G0", "G1"]:
                key = f"{prover}_fix{fix}"
                d["graph"].setdefault(key, {})
                for n in args.ns:
                    print(f"[reextract] {split} {key} n={n} (N={args.instances})",
                          flush=True)
                    d["graph"][key][str(n)] = attack_graph_perinstance_batched(
                        model, n, MAX_N, fix=fix, num_instances=args.instances,
                        k1=args.k1, k2=args.k2, seed=args.seed, mask_phi=mask)
                    json.dump(d, open(jf, "w"), indent=2)      # incremental save
            print(f"[reextract] done {prover} {split}", flush=True)
    print("[reextract] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
