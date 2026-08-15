"""Output-validity diagnostics (paper Table 5).

  psi_valid            : fraction of unconstrained psi samples that are valid
                         permutations of [n].
  psi_inv_correct_psi  : with (phi, psi) teacher-forced, fraction of test
                         instances whose argmax prediction of the psi_inv block
                         equals argsort(psi) at every position.
"""

import torch

from .layout import Layout
from .sample import sample_psi
from .data import is_perm_rows
from .seeding import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def psi_valid_diag(model, contexts: torch.Tensor, layout: Layout,
                   chunk: int = 1 << 14) -> float:
    n = layout.n
    valid = 0
    total = contexts.shape[0]
    for lo in range(0, total, chunk):
        block = contexts[lo:lo + chunk]
        samples = sample_psi(model, block, layout, valid=False)
        valid += is_perm_rows(samples, n).sum().item()
    return valid / total


@torch.no_grad()
def psi_inv_correct_diag(model, seqs: torch.Tensor, layout: Layout) -> float:
    """seqs: full teacher-forced sequences. Returns fraction correct."""
    blk = layout["psi_inv"]
    logits = model(seqs.to(DEVICE))
    pred = logits[:, blk.start - 1:blk.stop - 1, :layout.n].argmax(dim=-1)
    targ = seqs[:, blk].to(DEVICE)
    return (pred == targ).all(dim=1).float().mean().item()
