"""
Generate a dataset of (phi, psi) permutation pairs.

phi ~ Unif(S_n), psi ~ Unif(S_n), sampled INDEPENDENTLY of each other.
psi_inv and phi ∘ psi^{-1} are deterministic functions of (phi, psi)
and are computed at sequence-building time.
"""

import os
import torch

import config


def generate_dataset(size=config.DATASET_SIZE, n=config.N, seed=config.SEED):
    """Sample `size` independent (phi, psi) pairs."""
    g = torch.Generator().manual_seed(seed)
    phi = torch.stack([torch.randperm(n, generator=g) for _ in range(size)])
    psi = torch.stack([torch.randperm(n, generator=g) for _ in range(size)])
    return phi, psi


def generate_val1_dataset(size=config.VAL_SIZE, n=config.N, seed=config.SEED):
    """Sample `size` independent (phi, psi) pairs from a different RNG stream."""
    g = torch.Generator().manual_seed(seed + 1)
    phi = torch.stack([torch.randperm(n, generator=g) for _ in range(size)])
    psi = torch.stack([torch.randperm(n, generator=g) for _ in range(size)])
    return phi, psi


def generate_val2_dataset(train_phi, train_psi, size=config.VAL_SIZE,
                          n=config.N, seed=config.SEED):
    """Rejection-sample (phi, psi) pairs disjoint from training."""
    train_pairs = {
        (tuple(p.tolist()), tuple(q.tolist()))
        for p, q in zip(train_phi, train_psi)
    }
    g = torch.Generator().manual_seed(seed + 2)
    phi_list, psi_list = [], []
    while len(phi_list) < size:
        p = torch.randperm(n, generator=g)
        q = torch.randperm(n, generator=g)
        if (tuple(p.tolist()), tuple(q.tolist())) not in train_pairs:
            phi_list.append(p)
            psi_list.append(q)
    return torch.stack(phi_list), torch.stack(psi_list)


def save_dataset(phi, psi, path=config.DATASET_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"phi": phi, "psi": psi}, path)
    print(f"Saved dataset of size {len(phi)} to {path}")


def load_dataset(path=config.DATASET_PATH):
    d = torch.load(path)
    return d["phi"], d["psi"]


def build_sequence(phi, psi, n=config.N):
    """
    Construct the token sequence:
      [ phi_1..phi_n | psi_1..psi_n | psi_inv_1..psi_inv_n | (phi ∘ psi^{-1})_1..n ]

    psi_inv      = argsort(psi)
    phi_psi_inv  = phi gathered at psi_inv (i.e., (phi ∘ psi^{-1})(i) = phi[psi^{-1}(i)])

    Returns LongTensor of shape (B, 4n).
    """
    psi_inv = torch.argsort(psi, dim=1)
    phi_psi_inv = phi.gather(1, psi_inv)
    return torch.cat([phi, psi, psi_inv, phi_psi_inv], dim=1)


def sample_online(batch_size, n=config.N):
    """Fresh-sample alternative to the fixed dataset (unused by default)."""
    phi = torch.stack([torch.randperm(n) for _ in range(batch_size)])
    psi = torch.stack([torch.randperm(n) for _ in range(batch_size)])
    return phi, psi


if __name__ == "__main__":
    phi, psi = generate_dataset()
    val_phi, val_psi = generate_val1_dataset()
    val2_phi, val2_psi = generate_val2_dataset(phi, psi)
    print(f"\nFirst 3 train examples:")
    for i in range(3):
        print(f"  phi[{i}] = {phi[i].tolist()}, psi[{i}] = {psi[i].tolist()}")
    print(f"\nUnique phi values in train: "
          f"{len({tuple(p.tolist()) for p in phi})}/{config.DATASET_SIZE}")
    save_dataset(phi, psi)
    save_dataset(val_phi, val_psi, "data/val1.pt")
    save_dataset(val2_phi, val2_psi, "data/val2.pt")
