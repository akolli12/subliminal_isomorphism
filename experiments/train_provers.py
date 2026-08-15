"""Train prover checkpoints.

Prover tags:
  baseline       perm-only, standard psi loss                 (Tables 1, 2, 5)
  sim-aligned    perm-only, uniform-KL psi loss (defense 4.1) (Table 2)
  graph-given    graph-conditioned, phi input-only            (Table 3)
  graph-learned  graph-conditioned, phi under loss            (new experiment)

The witness-masked defense (4.2) needs no separate checkpoint: it reuses the
baseline prover with the phi block zeroed at inference (see run_table2.py).

Usage:
  python experiments/train_provers.py --tag baseline --n 4
  python experiments/train_provers.py --tag sim-aligned --n 5
  python experiments/train_provers.py --tag graph-learned --n 4
"""

import argparse

import torch

from _common import ckpt_path, DATA_DIR
from subliminal.configs import CONFIGS, VAL_SIZE, SEED_DATA, SEED_TRAIN
from subliminal.data import (
    make_perm_dataset, make_graph_dataset,
    build_perm_sequences, build_graph_sequences,
)
from subliminal.layout import perm_layout, graph_layout
from subliminal.train import train_prover

PERM_TAGS = {
    "baseline":    {"psi": "ce", "psi_inv": "ce", "phi_psi_inv": "ce"},
    "sim-aligned": {"psi": "uniform", "psi_inv": "ce", "phi_psi_inv": "ce"},
    # baseline loss trained at the DEFENSE config — the Table-2 control that
    # shows the same-config baseline still leaks (so the loss, not the data,
    # closes the leak). Same loss as "baseline", different training budget.
    "baseline-dc": {"psi": "ce", "psi_inv": "ce", "phi_psi_inv": "ce"},
}
GRAPH_TAGS = {
    "graph-given":   {"psi": "ce", "psi_inv": "ce", "phi_psi_inv": "ce"},
    "graph-learned": {"phi": "ce", "psi": "ce", "psi_inv": "ce",
                      "phi_psi_inv": "ce"},
}


def build_perm_data(n, size):
    phi, psi = make_perm_dataset(size, n, SEED_DATA)
    vphi, vpsi = make_perm_dataset(VAL_SIZE, n, SEED_DATA + 1)
    return build_perm_sequences(phi, psi), build_perm_sequences(vphi, vpsi)


def build_graph_data(n, size):
    g0, phi, psi = make_graph_dataset(size, n, SEED_DATA)
    vg0, vphi, vpsi = make_graph_dataset(VAL_SIZE, n, SEED_DATA + 1)
    return (build_graph_sequences(g0, phi, psi, n),
            build_graph_sequences(vg0, vphi, vpsi, n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    choices=list(PERM_TAGS) + list(GRAPH_TAGS))
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--steps", type=int, default=None,
                    help="override Table 4 step count (for smoke tests)")
    ap.add_argument("--dataset-size", type=int, default=None,
                    help="override training-set size")
    args = ap.parse_args()

    cfg = CONFIGS[args.n]
    is_graph = args.tag in GRAPH_TAGS

    # The graph-learned prover must actually learn to SOLVE graph isomorphism
    # from (G0, G1), a much harder task than the phi-given provers. It needs a
    # larger training set and more updates to generalize witness-finding beyond
    # memorization; the phi-given / perm-only provers use the Table 4 config.
    GRAPH_LEARNED = {
        4: (5000, 20000), 5: (10000, 30000), 6: (20000, 50000),
    }
    if args.tag == "graph-learned":
        default_size, default_steps = GRAPH_LEARNED[args.n]
    elif args.tag in ("sim-aligned", "baseline-dc"):
        from subliminal.configs import defense_train_config
        default_size, default_steps = defense_train_config(args.n)
    else:
        default_size, default_steps = cfg.dataset_size, cfg.steps
    size = args.dataset_size if args.dataset_size is not None else default_size
    steps = args.steps if args.steps is not None else default_steps

    if is_graph:
        layout = graph_layout(args.n)
        loss_spec = GRAPH_TAGS[args.tag]
        train_seqs, val_seqs = build_graph_data(args.n, size)
    else:
        layout = perm_layout(args.n)
        loss_spec = PERM_TAGS[args.tag]
        train_seqs, val_seqs = build_perm_data(args.n, size)

    print(f"[train] tag={args.tag} n={args.n} steps={steps} "
          f"dataset={size} batch={cfg.batch} loss={loss_spec}")
    train_prover(
        layout, loss_spec, train_seqs, val_seqs,
        steps=steps, batch=cfg.batch, lr=cfg.lr, seed=SEED_TRAIN,
        ckpt_path=ckpt_path(args.tag, args.n),
        d_model=cfg.d_model, n_heads=cfg.n_heads, n_layers=cfg.n_layers,
    )


if __name__ == "__main__":
    main()
