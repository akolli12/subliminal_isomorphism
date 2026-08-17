"""Uniform permutations and G(n, 1/2) graphs used by every experiment."""

import torch


def permutations(num_samples, n, generator):
    """Draw `num_samples` iid uniform permutations of size `n`."""
    return torch.rand((num_samples, n), generator=generator).argsort(dim=1)


def constrained_permutations(num_samples, n, j, u, generator):
    """Draw `num_samples` uniform permutations conditioned on phi[j] == u."""
    positions = torch.tensor([i for i in range(n) if i != j])
    values = torch.tensor([v for v in range(n) if v != u])
    order = torch.rand((num_samples, n - 1), generator=generator).argsort(dim=1)
    out = torch.empty((num_samples, n), dtype=torch.long)
    out[:, j] = u
    out[:, positions] = values[order]
    return out


def graphs(num_samples, n, generator):
    """Draw `num_samples` undirected G(n, 1/2) graphs as edge-bit vectors.

    The output has shape `(num_samples, n*(n-1)//2)`. Row `i` is graph `i`.
    Column `k` records the unordered edge at position `k` in lexicographic
    upper-triangle order:

        (0,1), (0,2), ..., (0,n-1), (1,2), ..., (n-2,n-1).

    A value of 1 means the edge is present. The lower adjacency triangle is
    implicit because every edge is undirected, and self-edges are excluded.
    """
    edges = n * (n - 1) // 2
    return (torch.rand((num_samples, edges), generator=generator) < 0.5).long()


def permute_graph(graph, phi, n):
    """Relabel each compressed undirected graph by its corresponding `phi`.

    This deliberately uses full adjacency matrices for clarity:

    1. Decode each upper-triangle edge vector into a symmetric matrix `A`.
    2. Relabel vertices so `A_new[phi(a), phi(b)] = A[a, b]`.
    3. Encode the upper triangle of `A_new` back into an edge vector.
    """
    device = phi.device
    row, column = torch.triu_indices(n, n, offset=1, device=device)

    adjacency = torch.zeros((len(graph), n, n), dtype=graph.dtype, device=device)
    adjacency[:, row, column] = graph
    adjacency[:, column, row] = graph

    inverse_phi = torch.argsort(phi, dim=1)
    batch = torch.arange(len(graph), device=device)[:, None, None]
    relabeled = adjacency[
        batch,
        inverse_phi[:, :, None],
        inverse_phi[:, None, :],
    ]
    return relabeled[:, row, column]
