"""Configuration, paths, and reproducible seeding for the main experiment."""

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    ns: tuple = (4, 5, 6, 7, 8, 9)
    max_n: int = 9
    examples_per_n: int = 8_000
    steps: int = 30_000
    batch_size: int = 64
    learning_rate: float = 3e-4
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 8
    perm_k1: int = 128
    perm_k2: int = 128
    perm_tests: int = 2_000
    graph_k1: int = 48
    graph_k2: int = 48
    graph_tests: int = 100
    graph_sample_batch: int = 8_192
    data_seed: int = 0
    tau_seed: int = 42
    eval_seed: int = 0

    @property
    def counts(self):
        return {n: self.examples_per_n for n in self.ns}


CONFIG = Config()

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
RESULTS = ROOT / "results"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def checkpoint(kind, seed):
    return CHECKPOINTS / f"mt_{kind}_seed{seed}_equal_n9.pt"


def result_path(seed):
    return RESULTS / f"main_table_seed{seed}_equal.json"
