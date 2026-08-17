"""End-to-end equivalence against the legacy pipeline for n=4 and n=5.

The budgets are intentionally tiny so this can run in CI. The test executes
the complete data -> training -> extraction path; changing a seed, draw order,
target, optimizer step, sampler, tau table, or assignment score makes it fail.
"""

import sys
from dataclasses import replace
from pathlib import Path

import torch

REPOSITORY = Path(__file__).resolve().parents[2]
CLEAN_ROOT = REPOSITORY / "subliminal_codex"
LEGACY_EXPERIMENTS = REPOSITORY / "experiments"
sys.path.insert(0, str(CLEAN_ROOT))
sys.path.insert(1, str(LEGACY_EXPERIMENTS))
sys.path.insert(2, str(REPOSITORY))

from graph_perinstance import attack_graph_perinstance_batched as legacy_graph_attack
from main_table import graph_train_soft as legacy_graph_soft_train
from multi_defenses import make_soft_dataset, train_soft as legacy_perm_soft_train
from multi_graph import make_dataset as make_legacy_graph_dataset
from multi_graph import train as legacy_graph_train
from multi_prover import extract as legacy_perm_attack
from multi_prover import make_mixed_dataset, train as legacy_perm_train
from subliminal.configs import SEED_DATA

from src.config import CONFIG
from src.extraction import permutation_attack
from src.graph_attack import graph_attack
from src.layouts import PermutationContext, perm_layout
from src.training import make_dataset, train


NS = (4, 5)
COUNTS = {4: 8, 5: 8}
TRAINING_SEED = 2
K1 = K2 = 2


def tensors_equal(left, right):
    return len(left) == len(right) and all(
        torch.equal(a, b) for a, b in zip(left, right)
    )


def models_equal(left, right):
    return left.state_dict().keys() == right.state_dict().keys() and all(
        torch.equal(left.state_dict()[name], right.state_dict()[name])
        for name in left.state_dict()
    )


def test_full_n4_n5_pipeline_matches_legacy(tmp_path):
    config = replace(
        CONFIG,
        ns=NS,
        max_n=5,
        examples_per_n=8,
        steps=2,
        batch_size=4,
        d_model=256,
        n_heads=4,
        n_layers=8,
    )

    # All four datasets and target formats.
    assert tensors_equal(
        make_mixed_dataset(NS, COUNTS, 5, SEED_DATA),
        make_dataset(config, False, False, reset_global_seed=True),
    )
    assert tensors_equal(
        make_soft_dataset(NS, COUNTS, 5, SEED_DATA, "uniform"),
        make_dataset(config, False, True, reset_global_seed=True),
    )
    assert tensors_equal(
        make_legacy_graph_dataset(NS, COUNTS, 5, SEED_DATA),
        make_dataset(config, True, False, reset_global_seed=True),
    )

    # All four training paths. Exact state equality checks initialization,
    # minibatch draws, losses, optimizer updates, and the graph-soft RNG quirk.
    legacy_perm = legacy_perm_train(
        NS, COUNTS, 5, 2, 4, 3e-4, TRAINING_SEED,
        str(tmp_path / "legacy_perm.pt"),
    )
    clean_perm = train(
        config, False, False, TRAINING_SEED, str(tmp_path / "clean_perm.pt")
    )
    assert models_equal(legacy_perm, clean_perm)

    legacy_perm_soft = legacy_perm_soft_train(
        NS, COUNTS, 5, 2, 4, 3e-4, TRAINING_SEED,
        str(tmp_path / "legacy_perm_soft.pt"), "uniform",
    )
    clean_perm_soft = train(
        config, False, True, TRAINING_SEED,
        str(tmp_path / "clean_perm_soft.pt"),
    )
    assert models_equal(legacy_perm_soft, clean_perm_soft)

    legacy_graph = legacy_graph_train(
        NS, COUNTS, 5, 2, 4, 3e-4, TRAINING_SEED,
        str(tmp_path / "legacy_graph.pt"),
    )
    clean_graph = train(
        config, True, False, TRAINING_SEED, str(tmp_path / "clean_graph.pt")
    )
    assert models_equal(legacy_graph, clean_graph)

    legacy_graph_soft = legacy_graph_soft_train(
        NS, COUNTS, 5, 2, 4, 3e-4, TRAINING_SEED,
        str(tmp_path / "legacy_graph_soft.pt"), "uniform",
    )
    clean_graph_soft = train(
        config, True, True, TRAINING_SEED,
        str(tmp_path / "clean_graph_soft.pt"), preserve_training_seed=True,
    )
    assert models_equal(legacy_graph_soft, clean_graph_soft)

    # Phi-only evaluation for both n values.
    legacy_perm_results = legacy_perm_attack(
        legacy_perm, NS, 5, K1, K2, mask_phi=False, test_cap=6
    )
    clean_perm_results = {
        n: permutation_attack(
            clean_perm,
            perm_layout(n, 5),
            PermutationContext(5),
            K1,
            K2,
            6,
            config.tau_seed,
            config.eval_seed,
        )
        for n in NS
    }
    assert clean_perm_results == legacy_perm_results

    # Full graph evaluation for both n values and both anchoring directions.
    for n in NS:
        for fixed in ("G0", "G1"):
            legacy_result = legacy_graph_attack(
                legacy_graph, n, 5, fix=fixed, num_instances=2,
                k1=K1, k2=K2, seed=TRAINING_SEED,
                mask_phi=False, sample_batch=16,
            )
            clean_result = graph_attack(
                clean_graph, n, 5, fixed=fixed, num_instances=2,
                k1=K1, k2=K2, seed=TRAINING_SEED,
                mask_phi=False, sample_batch=16,
            )
            assert clean_result == legacy_result
