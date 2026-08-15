"""Unit tests for the graph-conditioned shared prover."""

import torch

from subliminal.data import (rand_perms, rand_graphs, apply_perm_to_graph,
                             is_isomorphism, is_perm_rows)
from subliminal.model import TinyTransformer
from subliminal.multi import specials, IGNORE
from subliminal.multi_graph import (m_bits, graph_seq_len, graph_multi_layout,
                                    build_graph_multi_example,
                                    build_graph_multi_batch, GraphMultiContext)
from subliminal.sample import sample_psi

MAX_N = 9


def test_bits_and_seq_len():
    assert m_bits(4) == 6 and m_bits(9) == 36
    assert graph_seq_len(MAX_N) == 6 + 2 * m_bits(MAX_N) + 4 * MAX_N


def test_layout_positions():
    n = 5
    m = m_bits(n)
    lay = graph_multi_layout(n, MAX_N)
    assert lay["g0"] == slice(1, 1 + m)
    assert lay["g1"] == slice(2 + m, 2 + 2 * m)
    assert lay["phi"].start == 3 + 2 * m
    assert lay["psi"].start == 4 + 2 * m + n     # == context prefix width


def test_example_alignment_and_isomorphism():
    n = 6
    g = torch.Generator().manual_seed(0)
    g0 = rand_graphs(1, n, g)[0]
    phi, psi = rand_perms(1, n, g)[0], rand_perms(1, n, g)[0]
    toks, labels = build_graph_multi_example(n, g0, phi, psi, MAX_N)
    lay = graph_multi_layout(n, MAX_N)
    sp = specials(MAX_N)

    assert toks.shape[0] == graph_seq_len(MAX_N)
    assert (toks[lay["g0"]] == g0).all()
    g1 = apply_perm_to_graph(g0.unsqueeze(0), phi.unsqueeze(0), n)[0]
    assert (toks[lay["g1"]] == g1).all()
    assert is_isomorphism(phi.unsqueeze(0), g0.unsqueeze(0), g1.unsqueeze(0), n)[0]
    assert (toks[lay["phi"]] == phi).all()
    assert (toks[lay["psi"]] == psi).all()
    assert (toks[lay["psi_inv"]] == torch.argsort(psi)).all()
    assert (toks[lay["phi_psi_inv"]] == phi[torch.argsort(psi)]).all()
    # loss on the 3 permutation-output blocks only
    assert int((labels != IGNORE).sum()) == 3 * n
    assert (toks[6 + 2 * m_bits(n) + 4 * n:] == sp["PAD"]).all()


def test_batch_mixed_n():
    ns = [4, 7, 5]
    g = torch.Generator().manual_seed(1)
    g0s = [rand_graphs(1, n, g)[0] for n in ns]
    phis = [rand_perms(1, n, g)[0] for n in ns]
    psis = [rand_perms(1, n, g)[0] for n in ns]
    toks, labels = build_graph_multi_batch(ns, g0s, phis, psis, MAX_N)
    assert toks.shape == (3, graph_seq_len(MAX_N))


def test_graph_context_shape_and_g1():
    n = 5
    m = m_bits(n)
    g = torch.Generator().manual_seed(2)
    phis = rand_perms(4, n, g)
    ctx = GraphMultiContext(n, MAX_N, seed=0)(phis)
    sp = specials(MAX_N)
    assert ctx.shape == (4, 4 + 2 * m + n)          # == psi.start
    assert (ctx[:, 0] == sp["BOS"]).all()
    # g1 block equals phi applied to g0 block
    g0 = ctx[:, 1:1 + m]
    g1 = ctx[:, 2 + m:2 + 2 * m]
    assert (apply_perm_to_graph(g0, phis, n) == g1).all()


def test_graph_sampling_valid_perms():
    torch.manual_seed(0)
    dev = "cpu"
    m = TinyTransformer(specials(MAX_N)["vocab"], graph_seq_len(MAX_N),
                        d_model=32, n_heads=4, n_layers=2).to(dev).eval()
    g = torch.Generator().manual_seed(0)
    for n in (4, 5, 7):
        ctx = GraphMultiContext(n, MAX_N, seed=0)(rand_perms(12, n, g))
        psis = sample_psi(m, ctx, graph_multi_layout(n, MAX_N), valid=True)
        assert psis.shape == (12, n)
        assert is_perm_rows(psis, n).all()
