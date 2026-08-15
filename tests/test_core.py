"""Correctness tests for the core library (no GPU / no trained model needed)."""

import numpy as np
import torch

from subliminal.assignment import (
    best_assignment, top_k_assignments, brute_force_assignments)
from subliminal.data import (
    rand_perms, rand_perms_with_constraint, rand_graphs, apply_perm_to_graph,
    is_isomorphism, is_perm_rows, build_perm_sequences, build_graph_sequences)
from subliminal.layout import perm_layout, graph_layout
from subliminal.model import TinyTransformer
from subliminal.sample import sample_psi, generate_blocks
from subliminal.tau import ExtractorBank, pick_witness_coords
from subliminal.train import block_uniform_ce


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
def test_perm_layout_widths():
    lay = perm_layout(6)
    assert lay.seq_len == 24 and lay.vocab == 6
    assert lay["psi"].start == 6 and lay["phi_psi_inv"].stop == 24


def test_graph_layout_widths():
    lay = graph_layout(5)
    m = 5 * 4 // 2
    assert lay.seq_len == 2 * m + 4 * 5
    assert lay["g0"].start == 0 and lay["phi"].start == 2 * m


# ---------------------------------------------------------------------------
# Permutations
# ---------------------------------------------------------------------------
def test_rand_perms_are_permutations():
    g = torch.Generator().manual_seed(0)
    p = rand_perms(200, 7, g)
    assert is_perm_rows(p, 7).all()


def test_constrained_perms_respect_constraint():
    g = torch.Generator().manual_seed(1)
    p = rand_perms_with_constraint(300, 6, j=2, u=4, generator=g)
    assert is_perm_rows(p, 6).all()
    assert (p[:, 2] == 4).all()


# ---------------------------------------------------------------------------
# Graph isomorphism encoding
# ---------------------------------------------------------------------------
def test_g1_is_phi_of_g0():
    g = torch.Generator().manual_seed(2)
    n = 6
    g0, phi = rand_graphs(100, n, g), rand_perms(100, n, g)
    g1 = apply_perm_to_graph(g0, phi, n)
    assert is_isomorphism(phi, g0, g1, n).all()


def test_inverse_perm_recovers_g0():
    g = torch.Generator().manual_seed(3)
    n = 5
    g0, phi = rand_graphs(50, n, g), rand_perms(50, n, g)
    g1 = apply_perm_to_graph(g0, phi, n)
    phi_inv = torch.argsort(phi, dim=1)
    assert (apply_perm_to_graph(g1, phi_inv, n) == g0).all()


def test_wrong_perm_usually_not_isomorphism():
    g = torch.Generator().manual_seed(4)
    n = 7
    g0, phi = rand_graphs(200, n, g), rand_perms(200, n, g)
    g1 = apply_perm_to_graph(g0, phi, n)
    other = rand_perms(200, n, g)
    # for n=7 random graphs, a random perm is almost never an isomorphism
    assert is_isomorphism(other, g0, g1, n).float().mean() < 0.1


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------
def test_perm_sequence_blocks():
    g = torch.Generator().manual_seed(5)
    n = 5
    phi, psi = rand_perms(10, n, g), rand_perms(10, n, g)
    seq = build_perm_sequences(phi, psi)
    lay = perm_layout(n)
    assert (seq[:, lay["phi"]] == phi).all()
    assert (seq[:, lay["psi"]] == psi).all()
    # psi_inv = argsort(psi); phi_psi_inv = phi[psi_inv]
    psi_inv = torch.argsort(psi, dim=1)
    assert (seq[:, lay["psi_inv"]] == psi_inv).all()
    assert (seq[:, lay["phi_psi_inv"]] == phi.gather(1, psi_inv)).all()


def test_graph_sequence_g1_block():
    g = torch.Generator().manual_seed(6)
    n = 4
    g0, phi, psi = rand_graphs(8, n, g), rand_perms(8, n, g), rand_perms(8, n, g)
    seq = build_graph_sequences(g0, phi, psi, n)
    lay = graph_layout(n)
    assert (seq[:, lay["g0"]] == g0).all()
    assert (seq[:, lay["g1"]] == apply_perm_to_graph(g0, phi, n)).all()


# ---------------------------------------------------------------------------
# Simulator-aligned uniform loss floor
# ---------------------------------------------------------------------------
def test_uniform_loss_floor():
    """The uniform-on-remaining loss floor is log(n!)/n, achieved by a model
    that outputs exactly uniform-over-unused at each psi position. A model that
    is uniform over ALL n values instead pays log(n) > log(n!)/n."""
    import math
    import torch.nn.functional as F
    n = 5
    lay = perm_layout(n)
    g = torch.Generator().manual_seed(7)
    phi, psi = rand_perms(64, n, g), rand_perms(64, n, g)
    seq = build_perm_sequences(phi, psi)
    B = seq.shape[0]

    # Build OPTIMAL logits: at psi position t, 0 for values not yet used, -inf
    # for used values, so softmax = uniform-over-remaining = the target.
    logits = torch.zeros(B, lay.seq_len, lay.vocab)
    psi_tok = seq[:, lay["psi"]]
    one_hot = F.one_hot(psi_tok, n).float()
    used = torch.cat([torch.zeros_like(one_hot[:, :1]),
                      one_hot[:, :-1].cumsum(dim=1)], dim=1)     # (B, n, n)
    opt = torch.where(used > 0, torch.full_like(used, -1e9), torch.zeros_like(used))
    # place optimal logits at the positions that PREDICT the psi block
    logits[:, lay["psi"].start - 1:lay["psi"].stop - 1, :] = opt
    loss = block_uniform_ce(logits, seq, lay["psi"], n).item()
    assert abs(loss - math.lgamma(n + 1) / n) < 1e-4

    # a uniform-over-all model pays strictly more (log n)
    flat = torch.zeros(B, lay.seq_len, lay.vocab)
    loss_flat = block_uniform_ce(flat, seq, lay["psi"], n).item()
    assert abs(loss_flat - math.log(n)) < 1e-4


# ---------------------------------------------------------------------------
# Assignment: Murty == brute force
# ---------------------------------------------------------------------------
def test_murty_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(5):
        C = rng.random((5, 5))
        murty = [a for a, _ in top_k_assignments(C, 20)]
        brute = [a for a, _ in brute_force_assignments(C, 20)]
        assert murty == brute


def test_best_assignment_is_top1():
    rng = np.random.default_rng(1)
    C = rng.random((6, 6))
    best, cost = best_assignment(C)
    top = top_k_assignments(C, 1)[0]
    assert best == top[0] and abs(cost - top[1]) < 1e-9


# ---------------------------------------------------------------------------
# Extractor bank shapes
# ---------------------------------------------------------------------------
def test_extractor_cost_shapes():
    n = 4
    tau = torch.rand(n, n, n, n)
    bank = ExtractorBank(tau, tau.clamp_min(1e-9).log())
    marg = torch.rand(n, n)
    marg = marg / marg.sum(dim=1, keepdim=True)
    for name in ["single-max-spread raw", "aggregate-L2 log", "aggregate-L1 raw"]:
        C = bank.cost(name, marg)
        assert C.shape == (n, n)
    coords = pick_witness_coords(tau)
    assert set(coords) == set(range(n))


# ---------------------------------------------------------------------------
# Sampler produces valid permutations and respects width contract
# ---------------------------------------------------------------------------
def test_sample_psi_valid():
    torch.manual_seed(0)
    n = 5
    lay = perm_layout(n)
    device = "cpu"
    model = TinyTransformer(lay.vocab, lay.seq_len, d_model=32, n_layers=2).to(device)
    ctx = rand_perms(16, n, torch.Generator().manual_seed(0))
    psis = sample_psi(model, ctx, lay, valid=True)
    assert is_perm_rows(psis, n).all()


def test_witness_masking_makes_output_constant():
    """Zeroing the phi block makes psi logits identical across different phi."""
    torch.manual_seed(0)
    n = 4
    lay = perm_layout(n)
    device = "cpu"
    model = TinyTransformer(lay.vocab, lay.seq_len, d_model=32, n_layers=2).to(device).eval()
    a = torch.tensor([[0, 1, 2, 3, 0, 0, 0, 0]], device=device)
    b = torch.tensor([[3, 2, 1, 0, 0, 0, 0, 0]], device=device)
    with torch.no_grad():
        la = model(a, zero_blocks=(lay["phi"],))
        lb = model(b, zero_blocks=(lay["phi"],))
    # psi-generating position (last phi position) must be identical
    assert torch.allclose(la[:, lay["psi"].start - 1], lb[:, lay["psi"].start - 1])
