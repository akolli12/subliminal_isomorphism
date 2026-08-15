"""Unit tests for the shared length-generalizing prover (delimiter tokens)."""

import math

import torch
import torch.nn.functional as F

from subliminal.data import rand_perms, is_perm_rows
from subliminal.model import TinyTransformer
from subliminal.multi import (specials, multi_seq_len, multi_layout,
                              build_multi_example, build_multi_example_soft,
                              build_multi_batch, MultiContext, IGNORE)
from subliminal.sample import sample_psi

MAX_N = 9


# ---------------------------------------------------------------------------
# Vocabulary / layout
# ---------------------------------------------------------------------------
def test_specials_distinct_and_after_values():
    sp = specials(MAX_N)
    assert sp["SEP"] == MAX_N and sp["BOS"] == MAX_N + 1 and sp["PAD"] == MAX_N + 2
    assert sp["vocab"] == MAX_N + 3
    assert len({sp["SEP"], sp["BOS"], sp["PAD"]}) == 3


def test_seq_len_and_layout_positions():
    assert multi_seq_len(MAX_N) == 4 * MAX_N + 4
    for n in range(4, MAX_N + 1):
        lay = multi_layout(n, MAX_N)
        assert lay["phi"] == slice(1, 1 + n)
        assert lay["psi"].start == 2 + n
        assert lay["phi_psi_inv"].stop == 4 + 4 * n
        # psi context prefix width matches [BOS, phi, SEP]
        assert lay["psi"].start == n + 2


# ---------------------------------------------------------------------------
# Example construction
# ---------------------------------------------------------------------------
def test_example_block_alignment_and_delimiters():
    n = 6
    g = torch.Generator().manual_seed(0)
    phi, psi = rand_perms(1, n, g)[0], rand_perms(1, n, g)[0]
    toks, labels = build_multi_example(n, phi, psi, MAX_N)
    lay = multi_layout(n, MAX_N)
    sp = specials(n if False else MAX_N)

    assert toks.shape[0] == multi_seq_len(MAX_N)
    assert toks[0] == sp["BOS"]
    assert (toks[lay["phi"]] == phi).all()
    assert (toks[lay["psi"]] == psi).all()
    assert (toks[lay["psi_inv"]] == torch.argsort(psi)).all()
    assert (toks[lay["phi_psi_inv"]] == phi[torch.argsort(psi)]).all()
    # SEP between each block
    for pos in (1 + n, 2 + 2 * n, 3 + 3 * n):
        assert toks[pos] == sp["SEP"]
    # tail padded with PAD
    assert (toks[4 + 4 * n:] == sp["PAD"]).all()
    # loss only on the 3 predicted blocks (3n positions)
    assert int((labels != IGNORE).sum()) == 3 * n


def test_batch_stacks_mixed_n():
    ns = [4, 7, 5]
    g = torch.Generator().manual_seed(1)
    phis = [rand_perms(1, n, g)[0] for n in ns]
    psis = [rand_perms(1, n, g)[0] for n in ns]
    toks, labels = build_multi_batch(ns, phis, psis, MAX_N)
    assert toks.shape == (3, multi_seq_len(MAX_N))
    assert labels.shape == toks.shape


# ---------------------------------------------------------------------------
# Soft targets (baseline CE vs simulator-aligned uniform)
# ---------------------------------------------------------------------------
def test_soft_targets_ce_is_one_hot():
    n = 5
    g = torch.Generator().manual_seed(2)
    phi, psi = rand_perms(1, n, g)[0], rand_perms(1, n, g)[0]
    toks, tgt, mask = build_multi_example_soft(n, phi, psi, MAX_N, "ce")
    assert int(mask.sum()) == 3 * n
    # each masked row is a one-hot on the true next token
    for k in torch.nonzero(mask).flatten().tolist():
        assert torch.allclose(tgt[k].sum(), torch.tensor(1.0))
        assert tgt[k].argmax() == toks[k + 1]


def test_soft_targets_uniform_on_remaining():
    n = 5
    g = torch.Generator().manual_seed(3)
    phi, psi = rand_perms(1, n, g)[0], rand_perms(1, n, g)[0]
    toks, tgt, mask = build_multi_example_soft(n, phi, psi, MAX_N, "uniform")
    lay = multi_layout(n, MAX_N)
    # every masked row is a valid distribution
    for k in torch.nonzero(mask).flatten().tolist():
        assert torch.allclose(tgt[k].sum(), torch.tensor(1.0), atol=1e-5)
    # psi rows: uniform over the values not yet used
    used = set()
    for t, k in enumerate(range(lay["psi"].start, lay["psi"].stop)):
        row = tgt[k - 1]                       # predicts psi_t at position k-1
        remaining = [v for v in range(n) if v not in used]
        for v in range(n):
            expected = 1.0 / len(remaining) if v in remaining else 0.0
            assert abs(row[v].item() - expected) < 1e-5
        used.add(int(toks[k]))


def test_uniform_soft_ce_hits_floor():
    """A model outputting exactly uniform-on-remaining reaches the log(n!)/n floor."""
    n = 5
    g = torch.Generator().manual_seed(4)
    phi, psi = rand_perms(1, n, g)[0], rand_perms(1, n, g)[0]
    toks, tgt, mask = build_multi_example_soft(n, phi, psi, MAX_N, "uniform")
    V = specials(MAX_N)["vocab"]
    # optimal logits = log(target) where target>0 (so softmax == target on support)
    logits = torch.where(tgt > 0, tgt.clamp_min(1e-9).log(),
                         torch.full_like(tgt, -1e9))
    logp = F.log_softmax(logits, dim=-1)
    psi_loss = (-(tgt * logp).sum(-1))[mask]
    # average over the psi block equals log(n!)/n
    lay = multi_layout(n, MAX_N)
    psi_positions = list(range(lay["psi"].start - 1, lay["psi"].stop - 1))
    psi_vals = (-(tgt * logp).sum(-1))[psi_positions]
    assert abs(psi_vals.mean().item() - math.lgamma(n + 1) / n) < 1e-4


# ---------------------------------------------------------------------------
# Context + sampling
# ---------------------------------------------------------------------------
def test_context_shape_and_content():
    n = 6
    g = torch.Generator().manual_seed(5)
    phis = rand_perms(3, n, g)
    ctx = MultiContext(n, MAX_N)(phis)
    sp = specials(MAX_N)
    assert ctx.shape == (3, n + 2)
    assert (ctx[:, 0] == sp["BOS"]).all()
    assert (ctx[:, 1:1 + n] == phis).all()
    assert (ctx[:, -1] == sp["SEP"]).all()


def test_sampling_valid_perms_every_n():
    torch.manual_seed(0)
    dev = "cpu"
    m = TinyTransformer(specials(MAX_N)["vocab"], multi_seq_len(MAX_N),
                        d_model=32, n_heads=4, n_layers=2).to(dev).eval()
    g = torch.Generator().manual_seed(0)
    for n in (4, 6, 9):
        ctx = MultiContext(n, MAX_N)(rand_perms(16, n, g))
        psis = sample_psi(m, ctx, multi_layout(n, MAX_N), valid=True)
        assert psis.shape == (16, n)
        assert is_perm_rows(psis, n).all()


def test_masking_phi_makes_psi_logits_phi_independent():
    torch.manual_seed(0)
    dev = "cpu"
    n = 5
    lay = multi_layout(n, MAX_N)
    m = TinyTransformer(specials(MAX_N)["vocab"], multi_seq_len(MAX_N),
                        d_model=32, n_heads=4, n_layers=2).to(dev).eval()
    a = MultiContext(n, MAX_N)(torch.tensor([[0, 1, 2, 3, 4]])).to(dev)
    b = MultiContext(n, MAX_N)(torch.tensor([[4, 3, 2, 1, 0]])).to(dev)
    with torch.no_grad():
        la = m(a, zero_blocks=(lay["phi"],))
        lb = m(b, zero_blocks=(lay["phi"],))
    # the logit that predicts psi_0 (last context position) must match
    assert torch.allclose(la[:, lay["psi"].start - 1], lb[:, lay["psi"].start - 1])
