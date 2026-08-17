#!/usr/bin/env python3
"""Load graph checkpoints and append all 36 batched graph-table cells."""

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import CONFIG, checkpoint, result_path
from src.graph_attack import graph_attack
from src.layouts import graph_seq_len, specials
from src.model import TinyTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CONDITIONS = (
    # result name, checkpoint model, mask explicit phi tokens
    ("baseline", "baseline", False),
    ("sim_aligned", "sim_aligned", False),
    ("witness_masked", "baseline", True),
)


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)


def load_model(kind, seed):
    model = TinyTransformer(specials(CONFIG.max_n)["vocab"], graph_seq_len(CONFIG.max_n),
                            CONFIG.d_model, CONFIG.n_heads, CONFIG.n_layers).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint(kind, seed), map_location=DEVICE))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--instances", type=int, default=CONFIG.graph_tests)
    parser.add_argument("--k1", type=int, default=CONFIG.graph_k1)
    parser.add_argument("--k2", type=int, default=CONFIG.graph_k2)
    args = parser.parse_args()

    output_path = result_path(args.seed)
    with output_path.open() as handle:
        output = json.load(handle)
    output["graph_instances"] = args.instances
    output["graph_batched"] = True
    models = {"baseline": load_model("graph_base", args.seed),
              "sim_aligned": load_model("graph_sim", args.seed)}
    for condition, model_name, masked in CONDITIONS:
        model = models[model_name]
        for fixed in ("G0", "G1"):
            key = f"{condition}_fix{fixed}"
            output["graph"].setdefault(key, {})
            for n in CONFIG.ns:
                print(f"[graph] {key} n={n}", flush=True)
                output["graph"][key][str(n)] = graph_attack(
                    model, n, CONFIG.max_n, fixed, args.instances, args.k1, args.k2,
                    args.seed, mask_phi=masked, sample_batch=CONFIG.graph_sample_batch,
                )
                save_json(output, output_path)
    print("[graph] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
