"""Test suite for subliminal_isomorphism.

Run from project root:
    conda run --no-capture-output -n aug-spm pytest tests/ -v

Tests are grouped:
  - Layout & data:   build_sequence, val splits
  - Refactor equiv:  pin the recent vectorization claims (old vs new)
  - Sampler invariants: sample_psi_batch* outputs are valid permutations
  - Loss head:       block_loss returns the expected values on known inputs
"""
import math
from collections import Counter

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import config
from generate_dataset import (
    build_sequence,
    generate_dataset,
    generate_val1_dataset,
    generate_val2_dataset,
)
from measure_marginals import (
    sample_psi_batch,
    sample_psi_batch_multi,
    marginal_matrix,
)
from sample_marginals import generate_phis_with_constraint
from train import block_loss


N = config.N
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ---------------------------------------------------------------------------
# Mock model: returns uniform logits regardless of input. Lets us test the
# sampler / mask / aggregator without needing a real trained transformer.
# ---------------------------------------------------------------------------
class MockUniformModel:
    """f(x) = zeros(B, T, VOCAB) — uniform after softmax."""
    def __init__(self, vocab=config.VOCAB):
        self.vocab = vocab

    def __call__(self, x):
        return torch.zeros(x.shape[0], x.shape[1], self.vocab, device=x.device)

    def eval(self):
        pass


# ===========================================================================
# Layout & data
# ===========================================================================

def test_build_sequence_layout():
    """build_sequence produces a (B, 4N) tensor whose blocks match config slices."""
    torch.manual_seed(0)
    B = 5
    phi = torch.stack([torch.randperm(N) for _ in range(B)])
    psi = torch.stack([torch.randperm(N) for _ in range(B)])

    seq = build_sequence(phi, psi)

    assert seq.shape == (B, 4 * N) == (B, config.SEQ_LEN)
    assert torch.equal(seq[:, config.PHI], phi)
    assert torch.equal(seq[:, config.PSI], psi)

    psi_inv = torch.argsort(psi, dim=1)
    assert torch.equal(seq[:, config.PSI_INV], psi_inv)

    phi_psi_inv = phi.gather(1, psi_inv)
    assert torch.equal(seq[:, config.PHI_PSI_INV], phi_psi_inv)


def test_phi_psi_inv_composition_handcrafted():
    """Verify phi ∘ psi^{-1} on a worked example."""
    if N < 4:
        pytest.skip("hand-crafted example requires N >= 4")
    phi = torch.tensor([[2, 0, 3, 1]])
    psi = torch.tensor([[1, 3, 0, 2]])
    # psi as list = [1, 3, 0, 2]  →  positions of 0,1,2,3 in psi = [2, 0, 3, 1]
    # phi[[2,0,3,1]] = [phi[2], phi[0], phi[3], phi[1]] = [3, 2, 1, 0]
    expected = torch.tensor([[3, 2, 1, 0]])

    seq = build_sequence(
        torch.cat([phi, torch.zeros((1, max(0, N - 4)), dtype=torch.long)], dim=1)[:, :N],
        torch.cat([psi, torch.zeros((1, max(0, N - 4)), dtype=torch.long)], dim=1)[:, :N],
    ) if N != 4 else build_sequence(phi, psi)
    if N == 4:
        assert torch.equal(seq[:, config.PHI_PSI_INV], expected)


def test_psi_composed_with_psi_inv_is_identity():
    """For every (phi, psi), psi[psi_inv] == identity."""
    torch.manual_seed(0)
    psi = torch.stack([torch.randperm(N) for _ in range(20)])
    psi_inv = torch.argsort(psi, dim=1)
    composed = psi.gather(1, psi_inv)
    expected = torch.arange(N).unsqueeze(0).expand_as(composed)
    assert torch.equal(composed, expected)


def test_val2_disjoint_from_train():
    """Regenerate train + val2 from seed; val2 must contain no train pairs."""
    train_phi, train_psi = generate_dataset(size=80, n=N, seed=0)
    val2_phi, val2_psi = generate_val2_dataset(train_phi, train_psi, size=30, n=N, seed=0)

    train_pairs = {(tuple(p.tolist()), tuple(q.tolist()))
                   for p, q in zip(train_phi, train_psi)}
    val2_pairs = {(tuple(p.tolist()), tuple(q.tolist()))
                  for p, q in zip(val2_phi, val2_psi)}

    assert val2_pairs.isdisjoint(train_pairs)
    assert len(val2_pairs) == 30   # rejection-sampler met its quota


def test_val_seeds_differ():
    """train, val1, val2 use distinct RNG streams → first-row sanity check."""
    tr_phi, tr_psi = generate_dataset(size=10, n=N, seed=0)
    v1_phi, v1_psi = generate_val1_dataset(size=10, n=N, seed=0)
    # Highly unlikely that all 10 train phi == all 10 val1 phi if seeds differ.
    assert not torch.equal(tr_phi, v1_phi)


# ===========================================================================
# Refactor equivalence — most important; pin old-vs-new claims
# ===========================================================================

def test_marginal_matrix_matches_old_loop():
    """Vectorized one_hot.mean equals the old O(n^2) Python-loop version."""
    torch.manual_seed(123)
    K = 200
    samples = torch.randint(0, N, (K, N))

    # New (vectorized)
    M_new = F.one_hot(samples.long(), N).float().mean(dim=0)

    # Old (O(n^2) loop)
    M_old = torch.zeros(N, N)
    for i in range(N):
        for v in range(N):
            M_old[i, v] = (samples[:, i] == v).float().mean()

    assert torch.allclose(M_new, M_old, atol=1e-6)
    # Also confirm marginal_matrix (the wrapper actually used at runtime) agrees
    assert torch.allclose(marginal_matrix(samples, N), M_old, atol=1e-6)


def test_no_repeats_mask_matches_loop():
    """scatter_-based 'mask used values' is bit-equal to the per-step Python loop."""
    torch.manual_seed(7)
    B, V, j = 8, 10, 4
    step_logits_init = torch.randn(B, V)
    used = torch.stack([torch.randperm(V)[:j] for _ in range(B)])  # (B, j) distinct cols

    # Old style (per-prev loop)
    old_logits = step_logits_init.clone()
    for prev in range(j):
        already = used[:, prev]
        old_logits[torch.arange(B), already] = float('-inf')

    # New style (single scatter_)
    new_logits = step_logits_init.clone()
    new_logits.scatter_(1, used, float('-inf'))

    # Both should have -inf at the same positions and be equal elsewhere.
    assert torch.equal(torch.isinf(old_logits), torch.isinf(new_logits))
    finite = ~torch.isinf(old_logits)
    assert torch.equal(old_logits[finite], new_logits[finite])


def test_no_repeats_mask_handles_duplicates_in_used():
    """Both styles tolerate the (degenerate) case where 'used' has duplicates."""
    B, V = 3, 5
    step_logits_init = torch.randn(B, V)
    used = torch.tensor([[1, 1, 2], [0, 0, 0], [3, 4, 3]])

    old_logits = step_logits_init.clone()
    for prev in range(used.shape[1]):
        old_logits[torch.arange(B), used[:, prev]] = float('-inf')
    new_logits = step_logits_init.clone()
    new_logits.scatter_(1, used, float('-inf'))

    assert torch.equal(torch.isinf(old_logits), torch.isinf(new_logits))


def test_aggregate_mean_identity():
    """Math identity behind the batched aggregate refactor:

      mean_{phi}( mean_{psi}( one_hot(samples_{phi,psi}) ) )
        ==
      mean_{phi,psi}( one_hot(samples_{phi,psi}) )

    when each phi gets *exactly* the same number of psi samples.
    """
    torch.manual_seed(0)
    K1, K2 = 5, 7
    samples = torch.randint(0, N, (K1, K2, N))

    # Old: per-phi marginal then average
    M_per_phi = F.one_hot(samples, N).float().mean(dim=1)   # (K1, N, N)
    avg_old = M_per_phi.mean(dim=0)                          # (N, N)

    # New: flatten K1*K2 and compute one marginal
    samples_flat = samples.view(K1 * K2, N)
    avg_new = F.one_hot(samples_flat, N).float().mean(dim=0)

    assert torch.allclose(avg_old, avg_new, atol=1e-6)


# ===========================================================================
# Sampler invariants
# ===========================================================================

def test_sample_psi_batch_multi_returns_valid_permutations():
    """Every sampled psi is a valid permutation of [0, N) (no repeats, full range)."""
    torch.manual_seed(0)
    model = MockUniformModel()
    B = 50
    phis = torch.stack([torch.randperm(N) for _ in range(B)])

    psi_samples = sample_psi_batch_multi(model, phis)

    assert psi_samples.shape == (B, N)
    expected = torch.arange(N)
    for b in range(B):
        sorted_row, _ = torch.sort(psi_samples[b])
        assert torch.equal(sorted_row, expected), \
            f"row {b} = {psi_samples[b].tolist()} is not a permutation"


def test_sample_psi_batch_returns_valid_permutations():
    """Wrapper sample_psi_batch (single phi, n_samples copies) — also valid perms."""
    torch.manual_seed(1)
    model = MockUniformModel()
    phi = torch.randperm(N)
    samples = sample_psi_batch(model, phi, n_samples=30)
    assert samples.shape == (30, N)
    expected = torch.arange(N)
    for r in range(samples.shape[0]):
        sorted_row, _ = torch.sort(samples[r])
        assert torch.equal(sorted_row, expected)


def test_sample_psi_batch_marginal_is_uniform_under_uniform_model():
    """With a uniform-logit model and large K, the empirical marginal ≈ 1/N."""
    torch.manual_seed(0)
    model = MockUniformModel()
    phi = torch.randperm(N)
    K = 5000
    samples = sample_psi_batch(model, phi, n_samples=K)
    M = marginal_matrix(samples, N)
    target = 1.0 / N
    noise = math.sqrt(target * (1 - target) / K)   # ~1-sigma per cell
    assert (M - target).abs().max().item() < 6 * noise, \
        f"marginal too far from uniform; max-dev={(M - target).abs().max().item():.4f}"


def test_generate_phis_with_constraint_satisfies_constraint():
    """Every output phi has phi[a] == b and is a valid permutation."""
    torch.manual_seed(0)
    a, b, K1 = 1, 2, 100
    if b >= N:
        pytest.skip(f"b={b} requires N >= {b+1}")
    phis = generate_phis_with_constraint(N, a, b, K1)
    assert phis.shape == (K1, N)
    assert (phis[:, a] == b).all()
    expected = torch.arange(N)
    for r in range(K1):
        sorted_row, _ = torch.sort(phis[r])
        assert torch.equal(sorted_row, expected), \
            f"row {r} = {phis[r].tolist()} is not a permutation"


def test_generate_phis_with_constraint_uniform_distribution():
    """Over many samples the (n-1)! permutations of the 'other' positions should
    appear with roughly equal frequency."""
    torch.manual_seed(0)
    a, b = 0, 0
    if b >= N:
        pytest.skip()
    K1 = 4000
    phis = generate_phis_with_constraint(N, a, b, K1)

    # Count distinct permutations over the constrained-positions tuple
    counts = Counter(tuple(p.tolist()) for p in phis)
    n_perms = math.factorial(N - 1)
    assert len(counts) == n_perms, \
        f"saw {len(counts)} distinct phis, expected {n_perms}"
    expected = K1 / n_perms
    for perm, c in counts.items():
        # Loose bound: within ±50% of expected (well above noise for K1=4000).
        assert 0.5 * expected <= c <= 1.5 * expected, \
            f"perm {perm} has count {c} (expected ~{expected:.0f})"


# ===========================================================================
# Loss head
# ===========================================================================

def test_block_loss_uniform_logits_gives_log_vocab():
    """CE on uniform logits = log(VOCAB) for every target."""
    B = 4
    logits = torch.zeros(B, config.SEQ_LEN, config.VOCAB)        # uniform
    sequences = torch.randint(0, config.VOCAB, (B, config.SEQ_LEN))
    loss = block_loss(logits, sequences, config.PSI)
    expected = math.log(config.VOCAB)
    assert abs(loss.item() - expected) < 1e-5, \
        f"got {loss.item()}, expected {expected}"


def test_block_loss_perfect_prediction_gives_zero():
    """Logits sharply peaked at the target token → loss ≈ 0."""
    torch.manual_seed(0)
    B = 4
    sequences = torch.randint(0, config.VOCAB, (B, config.SEQ_LEN))

    BIG = 30.0
    logits = torch.full((B, config.SEQ_LEN, config.VOCAB), -BIG)
    # For each predicted-block target, set the target logit high.
    # block_loss reads logits at [block.start - 1 : block.stop - 1] and targets [block].
    block = config.PSI
    for k in range(block.start, block.stop):
        for b in range(B):
            logits[b, k - 1, sequences[b, k]] = BIG

    loss = block_loss(logits, sequences, block)
    assert loss.item() < 1e-5, f"expected ~0, got {loss.item()}"


def test_block_loss_offset_is_correct():
    """block_loss reads logits at positions [block.start-1, block.stop-1).

    Construct logits where ONLY position block.start-1 is peaked at the right
    token; loss should be < log(VOCAB) (some signal, not perfect)."""
    B = 1
    sequences = torch.randint(0, config.VOCAB, (B, config.SEQ_LEN))

    BIG = 30.0
    logits = torch.zeros(B, config.SEQ_LEN, config.VOCAB)         # uniform
    # Peak the FIRST logit position of the block at the FIRST target token
    block = config.PSI
    logits[0, block.start - 1, sequences[0, block.start]] = BIG

    loss = block_loss(logits, sequences, block)
    # Other (N-1) positions still uniform → contribute log(VOCAB) each;
    # peaked one contributes ~0. Average < log(VOCAB).
    assert 0 < loss.item() < math.log(config.VOCAB), \
        f"expected (0, log(VOCAB)); got {loss.item()}"
