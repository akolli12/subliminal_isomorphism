"""Conditional log-marginal table tau and the coordinate-wise extractor.

tau (paper Eq. 5):

    tau_{i,j,v}(u) = E_{phi' ~ Unif(S_n | phi'(j)=u)} [ log p*_{i,v}(x, phi') ]

estimated by Monte Carlo (Lemma B.5): for each (j, u) sample K1 phis with
phi(j)=u, draw K2 valid psi samples per phi, form the marginal M[i,v], and
average log M (log table) and M (raw table) over the K1 phis. tau is stored as
an (n, n, n, n) tensor indexed [i, j, v, u].

The extractor builds an (n x n) cost matrix C[j, u] from a test instance's
marginal, then returns phi_hat = argmin over assignments (Hungarian for top-1,
Murty for top-n). Six extractors (paper Section 5.1):

  single-max-spread {raw, log} : one witnessing coordinate (i_j, v_j) per j,
      chosen to maximise max_u tau - min_u tau; cost |L(i_j,v_j) - tau(u)|.
  aggregate-L^p {raw, log}, p in {1, 2} : all (i, v) jointly,
      cost ( sum_{i,v} |L(i,v) - tau_{i,j,v}(u)|^p )^{1/p}.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .data import rand_perms_with_constraint
from .layout import Layout
from .sample import sample_psi, marginal_matrix
from .seeding import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12


@torch.no_grad()
def per_phi_marginals(model, phis: torch.Tensor, layout: Layout, *, k2: int,
                      context_fn, chunk: int, zero_blocks=()) -> torch.Tensor:
    """(K1, n, n) marginals: row p is M[i,v] over k2 valid psi samples of phi p."""
    n = layout.n
    k1 = phis.shape[0]
    marg = torch.zeros((k1, n, n), device=DEVICE)
    phi_per_batch = max(1, chunk // k2)
    for lo in range(0, k1, phi_per_batch):
        block = phis[lo:lo + phi_per_batch]                    # (P, n)
        rep = block.repeat_interleave(k2, dim=0)               # (P*k2, n)
        psis = sample_psi(model, context_fn(rep), layout, valid=True,
                          zero_blocks=zero_blocks)
        oh = F.one_hot(psis.long(), n).float().view(block.shape[0], k2, n, n)
        marg[lo:lo + block.shape[0]] = oh.mean(dim=1)
    return marg


@torch.no_grad()
def estimate_tau(model, layout: Layout, *, k1: int, k2: int, seed: int,
                 context_fn, chunk: int = 1 << 15, zero_blocks=()):
    """Estimate raw and log tau tables. Returns (tau_raw, tau_log): (n,n,n,n)."""
    set_seed(seed)
    n = layout.n
    g = torch.Generator().manual_seed(seed)
    tau_raw = torch.zeros((n, n, n, n))
    tau_log = torch.zeros((n, n, n, n))
    for j in range(n):
        for u in range(n):
            phis = rand_perms_with_constraint(k1, n, j, u, g)
            marg = per_phi_marginals(model, phis, layout, k2=k2,
                                     context_fn=context_fn, chunk=chunk,
                                     zero_blocks=zero_blocks)
            tau_raw[:, j, :, u] = marg.mean(dim=0).cpu()
            tau_log[:, j, :, u] = marg.clamp_min(EPS).log().mean(dim=0).cpu()
        print(f"  tau: row j={j} done", flush=True)
    return tau_raw, tau_log


# ---------------------------------------------------------------------------
# Witness selection (single-coordinate extractors)
# ---------------------------------------------------------------------------
def pick_witness_coords(tau: torch.Tensor):
    """For each j pick (i_j, v_j) maximising spread max_u tau - min_u tau.

    tau indexed [i, j, v, u]. Returns dict j -> (i_j, v_j).
    """
    n = tau.shape[0]
    coords = {}
    for j in range(n):
        block = tau[:, j, :, :]                     # (i, v, u)
        spread = block.amax(dim=-1) - block.amin(dim=-1)   # (i, v)
        flat = int(spread.argmax())
        coords[j] = (flat // n, flat % n)
    return coords


# ---------------------------------------------------------------------------
# Cost matrices
# ---------------------------------------------------------------------------
def cost_single(tau: torch.Tensor, coords: dict, test_marg: torch.Tensor) -> np.ndarray:
    """C[j,u] = |test_marg[i_j,v_j] - tau[i_j,j,v_j,u]| (raw or log per caller)."""
    n = tau.shape[0]
    C = torch.zeros((n, n))
    for j in range(n):
        i_j, v_j = coords[j]
        C[j] = (test_marg[i_j, v_j] - tau[i_j, j, v_j, :]).abs()
    return C.numpy()


def cost_aggregate(tau: torch.Tensor, test_marg: torch.Tensor, p: int) -> np.ndarray:
    """C[j,u] = (sum_{i,v} |test_marg[i,v] - tau[i,j,v,u]|^p)^{1/p}."""
    diff = (test_marg.unsqueeze(1).unsqueeze(3) - tau).abs()   # (i, j, v, u)
    agg = (diff ** p).sum(dim=(0, 2)) ** (1.0 / p)             # (j, u)
    return agg.numpy()


# ---------------------------------------------------------------------------
# Extractor registry
# ---------------------------------------------------------------------------
EXTRACTORS = [
    "single-max-spread raw", "single-max-spread log",
    "aggregate-L1 raw", "aggregate-L2 raw",
    "aggregate-L1 log", "aggregate-L2 log",
]


class ExtractorBank:
    """Holds tau tables + witness coords; builds cost matrices per test marginal."""

    def __init__(self, tau_raw: torch.Tensor, tau_log: torch.Tensor):
        self.tau_raw = tau_raw
        self.tau_log = tau_log
        self.coords_raw = pick_witness_coords(tau_raw)
        self.coords_log = pick_witness_coords(tau_log)

    def cost(self, name: str, test_marg: torch.Tensor) -> np.ndarray:
        log_marg = test_marg.clamp_min(EPS).log()
        if name == "single-max-spread raw":
            return cost_single(self.tau_raw, self.coords_raw, test_marg)
        if name == "single-max-spread log":
            return cost_single(self.tau_log, self.coords_log, log_marg)
        if name == "aggregate-L1 raw":
            return cost_aggregate(self.tau_raw, test_marg, 1)
        if name == "aggregate-L2 raw":
            return cost_aggregate(self.tau_raw, test_marg, 2)
        if name == "aggregate-L1 log":
            return cost_aggregate(self.tau_log, log_marg, 1)
        if name == "aggregate-L2 log":
            return cost_aggregate(self.tau_log, log_marg, 2)
        raise ValueError(name)
