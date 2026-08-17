"""Fast equivalence checks against the current implementation."""

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
CLEAN = ROOT / "subliminal_codex"
sys.path.insert(0, str(CLEAN))
sys.path.insert(1, str(ROOT))

from data_generation.data import (constrained_permutations, graphs, permutations,
                                   permute_graph)
from src.layouts import (build_batch, graph_layout, graph_seq_len, perm_layout,
                         perm_seq_len, specials)

from subliminal.data import (apply_perm_to_graph, rand_graphs, rand_perms,
                             rand_perms_with_constraint)
from subliminal.multi import (build_multi_batch, build_multi_batch_soft,
                              multi_layout, multi_seq_len)
from subliminal.multi_graph import (build_graph_multi_batch,
                                    build_graph_multi_batch_soft,
                                    graph_multi_layout)


def test_seeded_data_matches():
    for seed in (0, 1, 42, 702):
        assert torch.equal(permutations(12, 6, torch.Generator().manual_seed(seed)),
                           rand_perms(12, 6, torch.Generator().manual_seed(seed)))
        assert torch.equal(graphs(12, 6, torch.Generator().manual_seed(seed)),
                           rand_graphs(12, 6, torch.Generator().manual_seed(seed)))
        assert torch.equal(
            constrained_permutations(12, 6, 2, 4, torch.Generator().manual_seed(seed)),
            rand_perms_with_constraint(12, 6, 2, 4,
                                       torch.Generator().manual_seed(seed)),
        )


def test_graph_permutation_matches():
    generator = torch.Generator().manual_seed(9)
    graph = graphs(10, 7, generator)
    phi = permutations(10, 7, generator)
    assert torch.equal(permute_graph(graph, phi, 7), apply_perm_to_graph(graph, phi, 7))


def test_layouts_match():
    for n in range(4, 10):
        clean_perm, old_perm = perm_layout(n, 9), multi_layout(n, 9)
        clean_graph, old_graph = graph_layout(n, 9), graph_multi_layout(n, 9)
        assert clean_perm.blocks == old_perm.blocks
        assert clean_graph.blocks == old_graph.blocks
        assert perm_seq_len(9) == multi_seq_len(9)
        assert graph_seq_len(9) == old_graph.seq_len
        assert specials(9)["vocab"] == old_perm.vocab


def test_tokenized_batches_match():
    generator = torch.Generator().manual_seed(0)
    ns = [4, 5, 6]
    phis = [permutations(1, n, generator)[0] for n in ns]
    psis = [permutations(1, n, generator)[0] for n in ns]
    graph_rows = [graphs(1, n, generator)[0] for n in ns]

    assert all(torch.equal(a, b) for a, b in zip(
        build_batch(ns, phis, psis, 9), build_multi_batch(ns, phis, psis, 9)))
    assert all(torch.equal(a, b) for a, b in zip(
        build_batch(ns, phis, psis, 9, soft=True, uniform_psi=True),
        build_multi_batch_soft(ns, phis, psis, 9, "uniform")))
    assert all(torch.equal(a, b) for a, b in zip(
        build_batch(ns, phis, psis, 9, graph_rows),
        build_graph_multi_batch(ns, graph_rows, phis, psis, 9)))
    assert all(torch.equal(a, b) for a, b in zip(
        build_batch(ns, phis, psis, 9, graph_rows, soft=True, uniform_psi=True),
        build_graph_multi_batch_soft(ns, graph_rows, phis, psis, 9, "uniform")))
