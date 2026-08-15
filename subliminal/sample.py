"""Autoregressive sampling of the commitment permutation psi from a prover.

The extractor is an oracle: it only ever observes psi samples conditioned on a
context (phi for the perm-only prover; (G0,G1,phi) for the graph prover). Two
sampling modes:

  unconstrained : sample each psi token from the raw softmax. Used to measure
                  the psi-validity diagnostic (Table 5) — how often the emitted
                  psi is a genuine permutation.
  valid         : mask already-used values at each step so every sample is a
                  valid permutation. Used for all marginal / tau estimation, so
                  the extractor works with the model's distribution *over S_n*.

`zero_blocks` implements the witness-masking defense (Section 4.2): passing the
context blocks that carry witness information forces P[psi | context] = P[psi].
"""

import torch
import torch.nn.functional as F

from .data import is_perm_rows
from .layout import Layout

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def generate_blocks(model, prefix: torch.Tensor, layout: Layout,
                    block_names, *, valid: bool = True, zero_blocks=()):
    """Autoregressively fill the named permutation blocks after `prefix`.

    prefix: (B, C) tokens up to the first block's start. Blocks are filled in
    the given order (each length n, treated as a permutation of [n]). Returns
    the full sequence (B, C + n*len(block_names)); slice with layout blocks to
    read individual results.
    """
    model.eval()
    n = layout.n
    device = next(model.parameters()).device
    prefix = prefix.to(device)
    B = prefix.shape[0]
    starts = [layout[name].start for name in block_names]
    assert prefix.shape[1] == starts[0], \
        f"expected prefix width {starts[0]}, got {prefix.shape[1]}"

    seq = torch.cat(
        [prefix, torch.zeros((B, n * len(block_names)), dtype=torch.long,
                             device=device)], dim=1)
    for start in starts:
        for t in range(n):
            logits = model(seq, zero_blocks=zero_blocks)
            step = logits[:, start - 1 + t, :n].clone()
            if valid and t > 0:
                step.scatter_(1, seq[:, start:start + t], float("-inf"))
            probs = F.softmax(step, dim=-1)
            seq[:, start + t] = torch.multinomial(probs, 1).squeeze(-1)
    return seq


@torch.no_grad()
def sample_psi(model, contexts: torch.Tensor, layout: Layout, *,
               valid: bool = True, zero_blocks=()) -> torch.Tensor:
    """Sample one psi per context row.

    contexts: (B, C) tokens covering everything up to (not including) the psi
    block, i.e. columns [0, psi.start). Returns (B, n) psi samples on `device`.
    """
    seq = generate_blocks(model, contexts, layout, ["psi"],
                          valid=valid, zero_blocks=zero_blocks)
    return seq[:, layout["psi"]]


def marginal_matrix(samples: torch.Tensor, n: int) -> torch.Tensor:
    """M[i, v] = fraction of samples with psi(i) = v. Shape (n, n)."""
    return F.one_hot(samples.long(), n).float().mean(dim=0)


@torch.no_grad()
def psi_valid_rate(model, contexts: torch.Tensor, layout: Layout) -> float:
    """Fraction of unconstrained samples that are valid permutations (Table 5)."""
    samples = sample_psi(model, contexts, layout, valid=False)
    return is_perm_rows(samples, layout.n).float().mean().item()
