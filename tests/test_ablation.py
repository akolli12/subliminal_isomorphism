"""Tests for train_ablation.py.

Covers:
  - apply_phi_mask: correct masking, no-op when disabled, no input mutation
  - uniform_psi_loss: target distribution sums to 1, reduces to log(VOCAB) on
    uniform logits, reduces to log(n!)/n on perfectly-targeted logits, blows
    up when the model puts mass on already-used values
  - block_losses dispatch: 'standard' vs 'uniform' picks the right psi loss;
    psi_inv and phi_psi_inv are unaffected by the mode
  - mode_tag / checkpoint_path_for / final_path_for: filename helpers
"""
import math

import pytest
import torch
import torch.nn.functional as F

import config
from train_ablation import (
    apply_phi_mask,
    uniform_psi_loss,
    block_loss,
    block_losses,
    mode_tag,
    checkpoint_path_for,
    final_path_for,
)
from generate_dataset import build_sequence


N = config.N


# ===========================================================================
# apply_phi_mask
# ===========================================================================

def test_apply_phi_mask_zeros_phi_block():
    """phi positions become 0; other blocks unchanged."""
    torch.manual_seed(0)
    B = 4
    phi = torch.stack([torch.randperm(N) for _ in range(B)])
    psi = torch.stack([torch.randperm(N) for _ in range(B)])
    seq = build_sequence(phi, psi)

    masked = apply_phi_mask(seq, mask_phi=True)

    assert (masked[:, config.PHI] == 0).all(), "phi positions should be 0"
    assert torch.equal(masked[:, config.PSI],         seq[:, config.PSI])
    assert torch.equal(masked[:, config.PSI_INV],     seq[:, config.PSI_INV])
    assert torch.equal(masked[:, config.PHI_PSI_INV], seq[:, config.PHI_PSI_INV])


def test_apply_phi_mask_noop_when_disabled():
    """mask_phi=False returns input unchanged."""
    torch.manual_seed(0)
    phi = torch.stack([torch.randperm(N) for _ in range(3)])
    psi = torch.stack([torch.randperm(N) for _ in range(3)])
    seq = build_sequence(phi, psi)
    out = apply_phi_mask(seq, mask_phi=False)
    assert torch.equal(out, seq)


def test_apply_phi_mask_does_not_mutate_input():
    """Calling apply_phi_mask(..., True) must not modify the original tensor."""
    torch.manual_seed(0)
    phi = torch.stack([torch.randperm(N) for _ in range(3)])
    psi = torch.stack([torch.randperm(N) for _ in range(3)])
    seq = build_sequence(phi, psi)
    seq_copy = seq.clone()
    _ = apply_phi_mask(seq, mask_phi=True)
    assert torch.equal(seq, seq_copy)


# ===========================================================================
# uniform_psi_loss
# ===========================================================================

def _build_used_and_target(psi_tokens, n):
    """Replicate the loss's internal target construction for verification."""
    psi_one_hot = F.one_hot(psi_tokens, n).float()
    used = torch.cat([
        torch.zeros_like(psi_one_hot[:, :1, :]),
        psi_one_hot[:, :-1, :].cumsum(dim=1),
    ], dim=1)
    n_remaining = (n - torch.arange(n, dtype=torch.float)).view(1, n, 1)
    target = (1.0 - used) / n_remaining
    return used, target


def test_uniform_psi_target_sums_to_one():
    """Target distribution at every (b, t) is a valid probability vector."""
    n = N
    B = 4
    torch.manual_seed(0)
    psi_tokens = torch.stack([torch.randperm(n) for _ in range(B)])
    _, target = _build_used_and_target(psi_tokens, n)
    sums = target.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B, n), atol=1e-6)


def test_uniform_psi_target_zero_on_used_values():
    """The target distribution puts 0 mass on every already-used value."""
    n = N
    B = 4
    torch.manual_seed(0)
    psi_tokens = torch.stack([torch.randperm(n) for _ in range(B)])
    used, target = _build_used_and_target(psi_tokens, n)
    # Wherever used == 1, target must be 0.
    assert torch.equal(target[used > 0], torch.zeros_like(target[used > 0]))


def test_uniform_psi_loss_uniform_logits_gives_log_vocab():
    """Uniform logits → log_softmax = -log(VOCAB), so soft CE = log(VOCAB) for any
    valid target distribution. This is the floor for an "all-uniform" model."""
    B = 3
    torch.manual_seed(0)
    logits = torch.zeros(B, config.SEQ_LEN, config.VOCAB)
    sequences = torch.zeros(B, config.SEQ_LEN, dtype=torch.long)
    sequences[:, config.PSI] = torch.stack([torch.randperm(N) for _ in range(B)])

    loss = uniform_psi_loss(logits, sequences)
    expected = math.log(config.VOCAB)
    assert abs(loss.item() - expected) < 1e-5, \
        f"got {loss.item()}, expected {expected}"


def test_uniform_psi_loss_perfect_target_gives_log_n_factorial_over_n():
    """If logits put exactly the target distribution (uniform on remaining,
    -BIG on used), the per-position loss is log(n - t), and the mean over
    t = 0..n-1 is log(n!)/n. For N=4: log(24)/4 ≈ 0.7945."""
    n = N
    B = 4
    BIG = 30.0
    torch.manual_seed(0)

    psi_tokens = torch.stack([torch.randperm(n) for _ in range(B)])
    sequences = torch.zeros(B, config.SEQ_LEN, dtype=torch.long)
    sequences[:, config.PSI] = psi_tokens

    used, _ = _build_used_and_target(psi_tokens, n)
    psi_logits_start = config.PSI.start - 1
    logits = torch.zeros(B, config.SEQ_LEN, config.VOCAB)
    # In the psi-predicting positions, set used values to -BIG, unused to 0.
    for t in range(n):
        for b in range(B):
            for v in range(config.VOCAB):
                if used[b, t, v] > 0:
                    logits[b, psi_logits_start + t, v] = -BIG

    loss = uniform_psi_loss(logits, sequences)
    expected = math.lgamma(n + 1) / n      # log(n!)/n
    assert abs(loss.item() - expected) < 1e-3, \
        f"got {loss.item()}, expected {expected}"


def test_uniform_psi_loss_mass_on_used_value_is_high():
    """If the model puts (almost) all mass on a value already used at this
    position, the loss at this position is ~BIG, dominating the mean.

    Works for any N >= 3 (need t=2 to have a non-trivial used set).
    BIG scales with N so the spike dominates the (N-1) uniform-logit
    positions that contribute log(N) each.
    """
    if N < 3:
        pytest.skip("test needs N >= 3 (t=2 must have used set of size >= 2)")
    B = 1
    BIG = 30.0 * N                                    # ensures spike >> (N-1)·log(N)

    # psi = [0, 1, 2, ..., N-1] → at t=2, used = {0, 1}
    psi_tokens = torch.arange(N).unsqueeze(0)
    sequences = torch.zeros(B, config.SEQ_LEN, dtype=torch.long)
    sequences[:, config.PSI] = psi_tokens

    logits = torch.zeros(B, config.SEQ_LEN, config.VOCAB)
    psi_logits_start = config.PSI.start - 1
    pred_pos = psi_logits_start + 2                   # predicts psi[2]
    logits[0, pred_pos, 0] = BIG                      # peak on value 0 (used)

    loss = uniform_psi_loss(logits, sequences)
    # At t=2: loss ≈ BIG (model assigns ~0 mass on the unused values).
    # Other positions contribute log(N). Mean ≈ ((N-1)·log(N) + BIG) / N,
    # which is always well above log(N) + 1 since BIG = 30N.
    assert loss.item() > math.log(N) + 1, \
        f"expected loss > log(N)+1 = {math.log(N) + 1:.3f}; got {loss.item():.3f}"


# ===========================================================================
# block_losses dispatch
# ===========================================================================

def test_block_losses_standard_mode_matches_block_loss():
    """In 'standard' mode, the psi component equals block_loss(PSI)."""
    torch.manual_seed(0)
    B = 4
    sequences = torch.randint(0, config.VOCAB, (B, config.SEQ_LEN))
    logits    = torch.randn(B, config.SEQ_LEN, config.VOCAB)

    psi_l, _, _ = block_losses(logits, sequences, 'standard')
    expected    = block_loss(logits, sequences, config.PSI)
    assert torch.allclose(psi_l, expected)


def test_block_losses_uniform_mode_matches_uniform_psi_loss():
    """In 'uniform' mode, the psi component equals uniform_psi_loss(...)."""
    torch.manual_seed(0)
    B = 4
    psi_tokens = torch.stack([torch.randperm(N) for _ in range(B)])
    sequences  = torch.zeros(B, config.SEQ_LEN, dtype=torch.long)
    sequences[:, config.PSI] = psi_tokens
    logits     = torch.randn(B, config.SEQ_LEN, config.VOCAB)

    psi_l, _, _ = block_losses(logits, sequences, 'uniform')
    expected    = uniform_psi_loss(logits, sequences)
    assert torch.allclose(psi_l, expected)


def test_block_losses_psi_inv_and_phi_psi_inv_independent_of_mode():
    """psi_inv and phi_psi_inv losses don't depend on psi_loss_mode."""
    torch.manual_seed(0)
    B = 4
    sequences = torch.randint(0, config.VOCAB, (B, config.SEQ_LEN))
    logits    = torch.randn(B, config.SEQ_LEN, config.VOCAB)

    _, std_inv, std_pp = block_losses(logits, sequences, 'standard')
    _, uni_inv, uni_pp = block_losses(logits, sequences, 'uniform')
    assert torch.allclose(std_inv, uni_inv)
    assert torch.allclose(std_pp,  uni_pp)


# ===========================================================================
# Path/tag helpers
# ===========================================================================

class _Args:
    def __init__(self, psi_loss_mode='standard', mask_phi=False):
        self.psi_loss_mode = psi_loss_mode
        self.mask_phi = mask_phi


def test_mode_tag_standard_default():
    assert mode_tag(_Args()) == 'standard'

def test_mode_tag_psi_uniform_only():
    assert mode_tag(_Args(psi_loss_mode='uniform')) == 'psi-uniform'

def test_mode_tag_mask_phi_only():
    assert mode_tag(_Args(mask_phi=True)) == 'mask-phi'

def test_mode_tag_both_flags():
    assert mode_tag(_Args(psi_loss_mode='uniform', mask_phi=True)) == 'psi-uniform_mask-phi'


def test_checkpoint_path_for_includes_step_tag_and_N():
    """checkpoint_path_for now returns `..._<tag>_<step>_<N>.pt` (N appended
    so that checkpoints from different N's don't collide)."""
    path = checkpoint_path_for(500, 'psi-uniform_mask-phi')
    assert path.endswith(f'_psi-uniform_mask-phi_500_{config.N}.pt'), path


def test_final_path_for_includes_only_tag():
    path = final_path_for('mask-phi')
    assert path.endswith('_mask-phi.pt'), path
    assert '_500' not in path
