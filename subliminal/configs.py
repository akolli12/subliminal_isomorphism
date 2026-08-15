"""Per-n experiment configuration.

Training hyperparameters reproduce paper Table 4 exactly; tau-estimation
sample budgets (K1 phis per (j,u) cell, K2 psis per phi) reproduce the runs
behind Table 1 (K1 = K2 = 64, 128, 256, 512, 512, 512 for n = 4..9).

Shared across all n (Section D.4): AdamW, lr 3e-4, no weight decay, no LR
schedule. Architecture (Section D.1): d_model 256, 8 layers, 4 heads.
"""

from dataclasses import dataclass, asdict

VAL_SIZE = 500

# seeds (recorded in every result JSON)
SEED_DATA, SEED_TRAIN, SEED_TAU, SEED_EVAL = 0, 0, 42, 0


@dataclass(frozen=True)
class Config:
    n: int
    dataset_size: int   # (phi, psi) training pairs, Table 4 "Dataset size"
    steps: int          # gradient updates, Table 4 "Steps"
    batch: int          # Table 4 "Batch Size"
    k1: int             # phis per (j, u) cell in tau estimation
    k2: int             # psi samples per phi (tau and test marginals)
    lr: float = 3e-4
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 8


CONFIGS = {
    4: Config(n=4, dataset_size=100,   steps=2500,   batch=8,  k1=64,  k2=64),
    5: Config(n=5, dataset_size=100,   steps=2500,   batch=8,  k1=128, k2=128),
    6: Config(n=6, dataset_size=200,   steps=5000,   batch=8,  k1=256, k2=256),
    7: Config(n=7, dataset_size=500,   steps=10000,  batch=8,  k1=512, k2=512),
    8: Config(n=8, dataset_size=10000, steps=50000,  batch=64, k1=512, k2=512),
    9: Config(n=9, dataset_size=20000, steps=100000, batch=64, k1=512, k2=512),
}


# ---------------------------------------------------------------------------
# Defense-experiment configuration (Table 2).
#
# The defenses are TRAINING interventions: the simulator-aligned KL loss only
# makes psi uniform once it has converged. On the Table-4 dataset sizes (as few
# as 100 pairs) the shared network memorizes the training instances and leaks
# phi into psi residually, so the defense under-performs. We therefore train
# the Table-2 provers on enough data / steps to converge. To prove the defense
# — not the extra data — closes the leak, Table 2 also trains a BASELINE at the
# SAME config (the control): it still leaks, while the defenses hit the random
# floor. n=8,9 already use large datasets, so their defense config equals the
# Table-4 config.
# ---------------------------------------------------------------------------
DEFENSE_TRAIN = {
    4: (5000, 15000),
    5: (5000, 15000),
    6: (10000, 20000),
    7: (10000, 20000),
    8: (10000, 50000),    # == Table 4
    9: (20000, 100000),   # == Table 4
}


def defense_train_config(n: int):
    """(dataset_size, steps) for the Table-2 controlled experiment at size n."""
    return DEFENSE_TRAIN[n]


def defense_uses_default(n: int) -> bool:
    """True when the defense config matches the Table-4 baseline config."""
    c = CONFIGS[n]
    return DEFENSE_TRAIN[n] == (c.dataset_size, c.steps)


def config_dict(n: int) -> dict:
    return asdict(CONFIGS[n])
