#!/usr/bin/env python3
"""Train the four shared models and produce diagnostics + phi-only results."""

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json

from src.config import CONFIG, checkpoint, result_path
from src.extraction import permutation_attack
from src.layouts import PermutationContext, perm_layout
from src.training import diagnostics, train


def save_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)


MODEL_SPECS = (
    # name, graph-conditioned, simulator-aligned
    ("perm_base", False, False),
    ("perm_sim", False, True),
    ("graph_base", True, False),
    ("graph_sim", True, True),
)

CONDITIONS = (
    # result name, checkpoint model, mask explicit phi tokens
    ("baseline", "perm_base", False),
    ("sim_aligned", "perm_sim", False),
    ("witness_masked", "perm_base", True),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=CONFIG.steps)
    args = parser.parse_args()
    config = replace(CONFIG, steps=args.steps)

    models = {}
    for name, graph_conditioned, simulator_aligned in MODEL_SPECS:
        print(f"[main] training {name}", flush=True)
        models[name] = train(
            config, graph_conditioned, simulator_aligned, seed=args.seed,
            checkpoint=str(checkpoint(name, args.seed)),
            preserve_training_seed=(name == "graph_sim"),
        )

    diagnostics = {
        "perm_baseline": diagnostics(models["perm_base"], config, False, config.eval_seed),
        "perm_sim": diagnostics(models["perm_sim"], config, False, config.eval_seed),
        "graph_baseline": diagnostics(models["graph_base"], config, True, config.eval_seed),
        "graph_sim": diagnostics(models["graph_sim"], config, True, config.eval_seed),
    }

    permutation_results = {}
    for condition, model_name, masked in CONDITIONS:
        model = models[model_name]
        permutation_results[condition] = {}
        for n in config.ns:
            print(f"[main] permutation attack {condition} n={n}", flush=True)
            permutation_results[condition][n] = permutation_attack(
                model, perm_layout(n, config.max_n), PermutationContext(config.max_n),
                config.perm_k1, config.perm_k2, config.perm_tests,
                config.tau_seed, config.eval_seed, mask_phi=masked,
            )
        save_json({
            "split": "equal", "seed": args.seed, "counts": config.counts,
            "ns": list(config.ns), "perm_tests": config.perm_tests,
            "graph_instances": config.graph_tests, "diagnostics": diagnostics,
            "perm": permutation_results, "graph": {},
        }, result_path(args.seed))


if __name__ == "__main__":
    main()
