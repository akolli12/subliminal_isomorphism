"""Dataset generation and sequence building (seeded).

Permutation-only prover: (phi, psi) pairs sampled iid uniform from S_n x S_n.
Graph-conditioned prover: (G0, phi, psi) with G0 ~ Erdos-Renyi(1/2) on [n],
phi, psi ~ Unif(S_n), and G1 := phi(G0) derived deterministically.

G1 = phi(G0) means: (a, b) is an edge of G0  iff  (phi(a), phi(b)) is an edge
of G1 — i.e. phi is the isomorphism from G0 to G1.
"""

import torch

from .layout import Layout


def rand_perms(count: int, n: int, generator: torch.Generator) -> torch.Tensor:
    """(count, n) iid uniform permutations of [n]."""
    keys = torch.rand((count, n), generator=generator)
    return keys.argsort(dim=1)


def rand_perms_with_constraint(count: int, n: int, j: int, u: int,
                               generator: torch.Generator) -> torch.Tensor:
    """(count, n) iid uniform permutations conditioned on phi(j) = u."""
    other_pos = torch.tensor([i for i in range(n) if i != j])
    other_val = torch.tensor([v for v in range(n) if v != u])
    keys = torch.rand((count, n - 1), generator=generator)
    out = torch.empty((count, n), dtype=torch.long)
    out[:, j] = u
    out[:, other_pos] = other_val[keys.argsort(dim=1)]
    return out


# ---------------------------------------------------------------------------
# Graphs (upper-triangular bit encoding)
# ---------------------------------------------------------------------------
def pair_index(n: int) -> torch.Tensor:
    """(n, n) matrix mapping unordered pair (a, b) -> its bit position."""
    idx = torch.full((n, n), -1, dtype=torch.long)
    k = 0
    for a in range(n):
        for b in range(a + 1, n):
            idx[a, b] = idx[b, a] = k
            k += 1
    return idx


def rand_graphs(count: int, n: int, generator: torch.Generator) -> torch.Tensor:
    """(count, m) iid Bernoulli(1/2) upper-triangular adjacency bits."""
    m = n * (n - 1) // 2
    return (torch.rand((count, m), generator=generator) < 0.5).long()


def apply_perm_to_graph(g_bits: torch.Tensor, phi: torch.Tensor, n: int) -> torch.Tensor:
    """Return bits of phi(G): edge (a,b) of G becomes edge (phi(a),phi(b))."""
    dev = phi.device
    pidx = pair_index(n).to(dev)
    pairs = torch.tensor([(a, b) for a in range(n) for b in range(a + 1, n)],
                         device=dev)
    src_a, src_b = pairs[:, 0], pairs[:, 1]           # (m,)
    dst = pidx[phi[:, src_a], phi[:, src_b]]          # (B, m) target bit slot
    out = torch.zeros_like(g_bits)
    out.scatter_(1, dst, g_bits)
    return out


def is_isomorphism(phi: torch.Tensor, g0_bits: torch.Tensor,
                   g1_bits: torch.Tensor, n: int) -> torch.Tensor:
    """(B,) bool: does each phi map G0 to G1? (phi rows must be permutations)"""
    return (apply_perm_to_graph(g0_bits, phi, n) == g1_bits).all(dim=1)


def is_perm_rows(samples: torch.Tensor, n: int) -> torch.Tensor:
    """(B,) bool: is each row a valid permutation of [n]?"""
    sorted_, _ = torch.sort(samples, dim=1)
    return (sorted_ == torch.arange(n, device=samples.device)).all(dim=1)


# ---------------------------------------------------------------------------
# Sequence building
# ---------------------------------------------------------------------------
def build_perm_sequences(phi: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
    """[ phi | psi | psi_inv | phi o psi^{-1} ], shape (B, 4n)."""
    psi_inv = torch.argsort(psi, dim=1)
    return torch.cat([phi, psi, psi_inv, phi.gather(1, psi_inv)], dim=1)


def build_graph_sequences(g0_bits: torch.Tensor, phi: torch.Tensor,
                          psi: torch.Tensor, n: int) -> torch.Tensor:
    """[ g0 | g1 | phi | psi | psi_inv | phi o psi^{-1} ], G1 = phi(G0)."""
    g1_bits = apply_perm_to_graph(g0_bits, phi, n)
    psi_inv = torch.argsort(psi, dim=1)
    return torch.cat(
        [g0_bits, g1_bits, phi, psi, psi_inv, phi.gather(1, psi_inv)], dim=1)


def make_perm_dataset(size: int, n: int, seed: int):
    """(phi, psi) train tensors; validation uses seed+1 (fresh iid pairs)."""
    g = torch.Generator().manual_seed(seed)
    return rand_perms(size, n, g), rand_perms(size, n, g)


def make_graph_dataset(size: int, n: int, seed: int):
    """(g0_bits, phi, psi) tensors, G1 derived at sequence-build time."""
    g = torch.Generator().manual_seed(seed)
    return rand_graphs(size, n, g), rand_perms(size, n, g), rand_perms(size, n, g)
