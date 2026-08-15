"""Ablation A: the leak is CORRELATIONAL, not exact memorization.

Success metric: top-n recovery by the polynomial-time extractor
(Hungarian/Murty). The extractor's tau table is estimated by AVERAGING the
coordinate marginals P[psi(i)=v | phi(j)=u] over RANDOM witnesses — so it can
only recover a witness if that per-coordinate structure is *shared* across
witnesses. That is exactly what "correlational" means.

The clean test: does the extractor recover witnesses the prover NEVER saw?
  seen   : witnesses in the prover's training set.
  unseen : fresh witnesses from the same distribution, absent from training.
If the leak were exact per-instance memorization, only `seen` would recover and
`unseen` would sit at the random baseline. Instead both recover far above
random (and unseen is comparable to seen) — the leak reaches witnesses the model
never trained on, because they share coordinate structure with the training
witnesses. We also report a Hamming-distance breakdown to show recovery does not
require closeness to a specific training witness.

Uses the usable baseline provers (n=6, n=7: valid psi + correct psi^{-1}) and
their cached tau; no retraining.
"""

import argparse
import os

import torch

from _common import ckpt_path, save_json, RESULTS_DIR
from attack import get_or_build_tau, attack_on_perms
from subliminal.configs import CONFIGS, SEED_DATA, SEED_EVAL
from subliminal.contexts import PermContext
from subliminal.data import make_perm_dataset, rand_perms
from subliminal.layout import perm_layout
from subliminal.train import load_prover

NUM = 300           # test witnesses per group


def min_hamming(cands, train):
    return (cands.unsqueeze(1) != train.unsqueeze(0)).sum(2).min(1).values


def seen_unseen(n, num, seed):
    cfg = CONFIGS[n]
    phi_train, _ = make_perm_dataset(cfg.dataset_size, n, SEED_DATA)
    train = torch.unique(phi_train, dim=0)
    train_set = {tuple(r.tolist()) for r in train}

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(train.shape[0], generator=g)[:num]
    seen = [tuple(train[i].tolist()) for i in idx]

    unseen, gg = [], torch.Generator().manual_seed(seed + 999)
    while len(unseen) < num:
        p = tuple(rand_perms(1, n, gg)[0].tolist())
        if p not in train_set:
            unseen.append(p)
    return train, seen, unseen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[6, 7])
    args = ap.parse_args()

    out = {}
    for n in args.ns:
        cfg = CONFIGS[n]
        layout = perm_layout(n)
        model = load_prover(ckpt_path("baseline", n), layout,
                            d_model=cfg.d_model, n_heads=cfg.n_heads,
                            n_layers=cfg.n_layers)
        tau_raw, tau_log = get_or_build_tau(model, layout, "baseline", n,
                                            PermContext(layout))
        train, seen, unseen = seen_unseen(n, NUM, SEED_EVAL)

        res_seen = attack_on_perms(model, layout, tau_raw, tau_log, seen,
                                   k2=cfg.k2, seed=SEED_EVAL)
        res_unseen = attack_on_perms(model, layout, tau_raw, tau_log, unseen,
                                     k2=cfg.k2, seed=SEED_EVAL)

        # distance breakdown of the UNSEEN set (how far from any training witness)
        ut = torch.tensor(unseen)
        dist = min_hamming(ut, train)
        by_d = {}
        for d in sorted(set(dist.tolist())):
            grp = [unseen[i] for i in range(len(unseen)) if dist[i] == d]
            if len(grp) < 20:
                continue
            r = attack_on_perms(model, layout, tau_raw, tau_log, grp,
                                k2=cfg.k2, seed=SEED_EVAL)
            by_d[d] = {"num": len(grp), "union_topn_pct": r["union"]["topn_pct"]}

        out[n] = {
            "random_topn_pct": res_seen["random_topn_pct"],
            "seen": {"num": len(seen),
                     "union_topn_pct": res_seen["union"]["topn_pct"],
                     "best_topn_pct": max(res_seen["extractors"][m]["topn_pct"]
                                          for m in res_seen["extractors"])},
            "unseen": {"num": len(unseen),
                       "union_topn_pct": res_unseen["union"]["topn_pct"],
                       "best_topn_pct": max(res_unseen["extractors"][m]["topn_pct"]
                                            for m in res_unseen["extractors"])},
            "unseen_by_distance": by_d,
        }
        print(f"[correlation] n={n} random={res_seen['random_topn_pct']:.3g}%  "
              f"seen union={res_seen['union']['topn_pct']:.1f}%  "
              f"unseen union={res_unseen['union']['topn_pct']:.1f}%", flush=True)
        print(f"   unseen-by-distance: "
              f"{ {d: round(v['union_topn_pct'],1) for d, v in by_d.items()} }",
              flush=True)
        save_json(out, os.path.join(RESULTS_DIR, "ablation_correlation.json"))


if __name__ == "__main__":
    main()
