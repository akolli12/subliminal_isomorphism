"""Shared paths, checkpoint naming, test-set construction for experiments."""

import itertools
import json
import os
import random
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from subliminal.contexts import GraphContext, PermContext  # noqa: E402
from subliminal.data import (  # noqa: E402
    build_perm_sequences, build_graph_sequences, rand_graphs, rand_perms,
)
from subliminal.layout import perm_layout, graph_layout  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(ROOT, "checkpoints")
DATA_DIR = os.path.join(ROOT, "data")
RESULTS_DIR = os.path.join(ROOT, "results")

for d in (CKPT_DIR, DATA_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)


def ckpt_path(tag: str, n: int) -> str:
    return os.path.join(CKPT_DIR, f"{tag}_n{n}.pt")


def tau_path(tag: str, n: int, kind: str) -> str:
    return os.path.join(DATA_DIR, f"tau_{kind}_{tag}_n{n}.pt")


def save_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"  wrote {path}")


def full_or_sample_perms(n: int, max_test: int, seed: int):
    """All permutations of S_n, shuffled (seeded); truncated to max_test."""
    perms = list(itertools.permutations(range(n)))
    random.Random(seed).shuffle(perms)
    if max_test is not None and len(perms) > max_test:
        perms = perms[:max_test]
    return perms
