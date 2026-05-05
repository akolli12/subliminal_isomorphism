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
from sample_marginals import (
    generate_phis_with_constraint,
    aggregate_by_constraint,
    aggregate_null,
    echo_signal,
    max_deviation_per_cell,
)
from building_tau import (
    pick_witnessing_coordinates,
    build_tau,
    ell,
    ell_vector,
)
from cost import build_cost_matrix, recover_phi, assignment_cost
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
    appear with roughly equal frequency.

    K1 scales with (N-1)! so we get ~50 samples per bucket regardless of N.
    """
    torch.manual_seed(0)
    a, b = 0, 0
    if b >= N:
        pytest.skip()
    n_perms = math.factorial(N - 1)
    K1 = max(4000, 50 * n_perms)        # ≥50 expected samples per bucket
    phis = generate_phis_with_constraint(N, a, b, K1)

    # Count distinct permutations over the constrained-positions tuple
    counts = Counter(tuple(p.tolist()) for p in phis)
    assert len(counts) == n_perms, \
        f"saw {len(counts)} distinct phis, expected {n_perms}"
    # Chi-squared goodness-of-fit test for uniformity.
    # Under the null (uniform), chi2 ~ χ²(df) with mean df and std sqrt(2*df).
    # We reject only on extreme deviation (±5σ) so the test fails on real
    # non-uniformity but tolerates Poisson noise.
    expected = K1 / n_perms
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    df = n_perms - 1
    chi2_std = math.sqrt(2 * df)
    assert abs(chi2 - df) < 5 * chi2_std, (
        f"chi2 = {chi2:.1f}, expected {df} ± 5σ = ±{5 * chi2_std:.1f} "
        f"(deviation {(chi2 - df) / chi2_std:+.2f}σ)"
    )


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


# ===========================================================================
# sample_marginals — aggregate, null, echo, max-deviation
# ===========================================================================

def test_aggregate_by_constraint_shape_and_uniformity():
    """Under a uniform-logit model, every avg_M[(a,b)] should be ~1/n
    (no leakage), and the dict should have n*n entries each of shape (n, n)."""
    torch.manual_seed(0)
    model = MockUniformModel()
    K1, K2 = 5, 200   # K1*K2=1000 → noise ~sqrt((1/n)(1-1/n)/1000) ≈ 0.014 for n=4
    results = aggregate_by_constraint(model, n=N, k1=K1, k2=K2)

    assert len(results) == N * N
    target = 1.0 / N
    for (a, b), M in results.items():
        assert M.shape == (N, N)
        # row-sums should be exactly 1 (each position must take some value)
        row_sums = M.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones(N), atol=1e-4)
        # under uniform model, M ≈ 1/n everywhere
        max_dev = (M - target).abs().max().item()
        assert max_dev < 0.06, \
            f"({a},{b}) max-dev {max_dev:.4f} too large for uniform model"


def test_aggregate_by_constraint_respects_constraint():
    """If the model were oracular and only emitted ψ that respect the constraint
    phi(a)=b somehow — well, here we just verify it inputs phi's that satisfy
    the constraint (sample_psi_batch_multi takes phis as-is)."""
    # Indirect check: hook into generate_phis_with_constraint and verify all
    # inputs satisfy phi[a] == b (we already test this elsewhere).
    a, b = 1, 2
    if b >= N:
        pytest.skip()
    phis = generate_phis_with_constraint(N, a, b, k1=20)
    assert (phis[:, a] == b).all()


def test_aggregate_null_is_uniform():
    """The null aggregator just samples uniform random perms — marginals must
    converge to 1/n (within sampling noise)."""
    torch.manual_seed(0)
    K1, K2 = 5, 500
    results = aggregate_null(n=N, k1=K1, k2=K2)
    target = 1.0 / N
    for (a, b), M in results.items():
        assert M.shape == (N, N)
        max_dev = (M - target).abs().max().item()
        assert max_dev < 0.05, \
            f"null ({a},{b}) max-dev {max_dev:.4f} unexpectedly high"


def test_echo_signal_indexing():
    """D[a, b] = avg_M[(a, b)][a, b] — verify with hand-crafted results."""
    fake = {}
    for a in range(N):
        for b in range(N):
            M = torch.zeros(N, N)
            M[a, b] = 0.5 + 0.01 * (a * N + b)   # distinguishable per cell
            fake[(a, b)] = M
    D = echo_signal(fake, n=N)
    assert D.shape == (N, N)
    for a in range(N):
        for b in range(N):
            assert D[a, b].item() == pytest.approx(0.5 + 0.01 * (a * N + b))


def test_max_deviation_per_cell():
    """delta[a, b] = max |avg_M[(a,b)] - 1/n|."""
    target = 1.0 / N
    fake = {}
    expected = torch.zeros(N, N)
    for a in range(N):
        for b in range(N):
            M = torch.full((N, N), target)
            spike = 0.01 * (a + 1) * (b + 2)   # unique per (a,b)
            M[0, 0] = target + spike
            fake[(a, b)] = M
            expected[a, b] = spike
    delta = max_deviation_per_cell(fake, n=N)
    assert torch.allclose(delta, expected, atol=1e-6)


# ===========================================================================
# building_tau — pick_witnessing_coordinates, build_tau, ell
# ===========================================================================

def test_pick_witnessing_coordinates_picks_max_spread_cell():
    """Hand-craft model_results so a single (i,v) has obvious max spread for j=0."""
    target = 1.0 / N
    # Default: every cell uniform-ish.
    results = {(j, u): torch.full((N, N), target) for j in range(N) for u in range(N)}

    # For j=0: vary the (i=2, v=1) entry across u: 0.0 at u=0, 0.9 at u=1, ...
    big_spread = [0.0, 0.9, 0.1, 0.4][:N] + [target] * max(0, N - 4)
    big_spread = big_spread[:N]
    for u in range(N):
        results[(0, u)][2, 1] = big_spread[u]

    # For j=1: a smaller but nonzero spread at (i=0, v=0)
    small_spread = [0.20, 0.25, 0.22, 0.21][:N] + [target] * max(0, N - 4)
    small_spread = small_spread[:N]
    for u in range(N):
        results[(1, u)][0, 0] = small_spread[u]

    witnesses, spreads = pick_witnessing_coordinates(results, n=N)

    assert witnesses[0] == (2, 1), f"expected (2,1) for j=0; got {witnesses[0]}"
    expected_spread_j0 = max(big_spread) - min(big_spread)
    assert spreads[0] == pytest.approx(expected_spread_j0, abs=1e-6)

    assert witnesses[1] == (0, 0), f"expected (0,0) for j=1; got {witnesses[1]}"


def test_build_tau_indexing():
    """tau[j, u] = model_results[(j, u)][i_j, v_j]."""
    results = {}
    # Distinguishable values so we can check correct indexing.
    for j in range(N):
        for u in range(N):
            M = torch.zeros(N, N)
            M[j % N, u % N] = 0.5 + 0.01 * (j * N + u)
            results[(j, u)] = M

    # Witnesses point to the cell we set above for each j.
    witnesses = {j: (j % N, j % N) for j in range(N)}
    # ^ note: at j we read results[(j, u)][j%N, j%N] which we set only when
    # u%N == j%N — i.e. u == j for u in range(N). For other u we set
    # results[(j, u)][j, u] not [j, j], so [j, j] is 0. That's fine for
    # asserting build_tau READS from [i_j, v_j]:
    tau = build_tau(results, witnesses, n=N)
    assert tau.shape == (N, N)
    for j in range(N):
        i_j, v_j = witnesses[j]
        for u in range(N):
            assert tau[j, u].item() == results[(j, u)][i_j, v_j].item()


def test_ell_uniform_under_uniform_model():
    """ell(phi, model, j) = empirical P[psi(i_j) = v_j | phi]. Under the
    uniform-logit model this should be ≈ 1/n for any (j, witnesses)."""
    torch.manual_seed(0)
    model = MockUniformModel()
    phi = torch.randperm(N)
    witnesses = {j: (j % N, (j + 1) % N) for j in range(N)}

    n_samples = 4000
    target = 1.0 / N
    noise = math.sqrt(target * (1 - target) / n_samples)
    for j in range(N):
        e = ell(phi, model, j, witnesses, n_samples=n_samples)
        assert abs(e - target) < 6 * noise, \
            f"ell at j={j} = {e:.4f} too far from {target:.4f}"


def test_ell_vector_returns_n_values():
    torch.manual_seed(0)
    model = MockUniformModel()
    phi = torch.randperm(N)
    witnesses = {j: (j % N, j % N) for j in range(N)}
    v = ell_vector(phi, model, witnesses, n=N, n_samples=200)
    assert v.shape == (N,)
    assert (v >= 0).all() and (v <= 1).all()


# ===========================================================================
# cost — build_cost_matrix, recover_phi, assignment_cost
# ===========================================================================

def test_assignment_cost_is_sum():
    """sum_j C[j, sigma(j)] for a known assignment."""
    C = torch.tensor([[1., 2., 3., 4.],
                      [5., 6., 7., 8.],
                      [9., 10., 11., 12.],
                      [13., 14., 15., 16.]])[:N, :N]
    if N != 4:
        pytest.skip("hand-crafted N=4 example")
    sigma = torch.tensor([0, 1, 2, 3])
    assert assignment_cost(C, sigma) == pytest.approx(1 + 6 + 11 + 16)
    sigma2 = torch.tensor([3, 2, 1, 0])
    assert assignment_cost(C, sigma2) == pytest.approx(4 + 7 + 10 + 13)


def test_build_cost_matrix_log_formula():
    """C[j, u] = log(ell[j]) - log(tau[j, u])."""
    torch.manual_seed(0)
    model = MockUniformModel()
    phi = torch.randperm(N)
    witnesses = {j: (j % N, j % N) for j in range(N)}
    tau = torch.full((N, N), 0.25)        # all entries equal → C should be ~0 everywhere
    C, ell_vals = build_cost_matrix(phi, model, witnesses, tau,
                                    n=N, n_samples=2000)
    assert C.shape == (N, N)
    assert ell_vals.shape == (N,)

    # Reconstruct the formula and compare:
    EPS = 1e-12
    expected = torch.log(ell_vals.clamp_min(EPS)).unsqueeze(1) - torch.log(tau.clamp_min(EPS))
    assert torch.allclose(C, expected, atol=1e-6)


def test_recover_phi_with_synthetic_perfect_signal():
    """If we hand-craft a tau s.t. the cost matrix has a unique min-cost
    assignment matching `phi`, recover_phi must return phi."""
    torch.manual_seed(0)
    model = MockUniformModel()        # ell is irrelevant to the *direction* of recovery
    phi = torch.tensor(list(range(N)))[torch.randperm(N)]
    witnesses = {j: (j % N, j % N) for j in range(N)}

    # Build tau such that for each j, tau[j, phi[j]] is highest (so log_tau
    # is highest and -log_tau lowest → cost lowest at the true u=phi[j]).
    tau = torch.full((N, N), 0.05)
    for j in range(N):
        tau[j, phi[j].item()] = 0.95

    phi_hat, success, C, ell_vals = recover_phi(
        phi, model, witnesses, tau, n=N, n_samples=2000,
    )
    assert torch.equal(phi_hat, phi), \
        f"expected {phi.tolist()}; got {phi_hat.tolist()}"
    assert success is True


def test_recover_phi_fails_on_uniform_tau():
    """If tau is constant, every assignment is tied → recovery may or may not
    pick the true phi. Just verify the function returns a valid permutation
    and doesn't crash."""
    torch.manual_seed(1)
    model = MockUniformModel()
    phi = torch.tensor(list(range(N)))
    witnesses = {j: (0, 0) for j in range(N)}
    tau = torch.full((N, N), 0.25)
    phi_hat, success, _, _ = recover_phi(phi, model, witnesses, tau,
                                         n=N, n_samples=200)
    # phi_hat should be a permutation.
    sorted_, _ = torch.sort(phi_hat)
    assert torch.equal(sorted_, torch.arange(N))


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
