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
    """Draw `num_samples` upper-triangular G(n, 1/2) adjacency vectors."""
    edges = n * (n - 1) // 2
    return (torch.rand((num_samples, edges), generator=generator) < 0.5).long()

