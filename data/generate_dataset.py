"""
Generate a dataset of (phi, psi) permutation pairs.

phi ~ Unif(S_n), psi ~ Unif(S_n), sampled INDEPENDENTLY of each other.
phi_inv is computed deterministically from phi at sequence-building time
(it's a function of phi, not part of the stored dataset).
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

def generate_val1_dataset(size=config.VAL_SIZE, n=config.N, seed=config.SEED): # won't this produce the same dataset since same seed?
    """Sample `size` independent (phi, psi) pairs."""
    seed+=1 #val seed is 1 higher
    g = torch.Generator().manual_seed(seed)
    phi = torch.stack([torch.randperm(n, generator=g) for _ in range(size)])
    psi = torch.stack([torch.randperm(n, generator=g) for _ in range(size)])
    return phi, psi

def generate_val2_dataset(size=config.VAL_SIZE, n=config.N, seed=config.SEED):
    """Sample validation pairs rejects anything in training."""

    # Reconstruct training set
    train_phi, train_psi = generate_dataset(size=size, n=n, seed=seed)
    train_pairs = {
        (tuple(p.tolist()), tuple(q.tolist()))
        for p, q in zip(train_phi, train_psi)
    }

    # Use different RNG stream for val2
    g = torch.Generator().manual_seed(seed + 2)

    phi_list, psi_list = [], []

    while len(phi_list) < size:
        p = torch.randperm(n, generator=g)
        q = torch.randperm(n, generator=g)

        pair = (tuple(p.tolist()), tuple(q.tolist()))

        if pair not in train_pairs:
            phi_list.append(p)
            psi_list.append(q)

    phi = torch.stack(phi_list)
    psi = torch.stack(psi_list)

    return phi, psi


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
      [BOS] phi_1..phi_n [SEP] psi_1..psi_n [SEP] phi_inv_1..phi_inv_n

    Returns a LongTensor of shape (B, 3n + 3).
    """
    B = phi.shape[0]
    phi_inv = torch.argsort(phi, dim=1)
    bos = torch.full((B, 1), config.BOS, dtype=torch.long)
    sep = torch.full((B, 1), config.SEP, dtype=torch.long)
    return torch.cat([bos, phi, sep, psi, sep, phi_inv], dim=1)


def sample_online(batch_size, n=config.N):
    """
    Alternative to a fixed dataset: sample a fresh batch every call.
    Use this if you want to see psi loss plateau at log(n!)/n
    instead of being memorized to ~0.
    """
    phi = torch.stack([torch.randperm(n) for _ in range(batch_size)])
    psi = torch.stack([torch.randperm(n) for _ in range(batch_size)])
    return phi, psi


if __name__ == "__main__":
    phi, psi = generate_dataset()
    save_dataset(phi, psi)
    val_phi, val_psi = generate_val1_dataset()
    val2_phi, val2_psi = generate_val2_dataset()
    print(f"\nFirst 3 examples:")
    for i in range(3):
        print(f"  phi[{i}] = {phi[i].tolist()}, psi[{i}] = {psi[i].tolist()}")
    print(f"\nUnique phi values in dataset: "
          f"{len({tuple(p.tolist()) for p in phi})}/{config.DATASET_SIZE}")
    print(f"\nFirst 3 examples:")
    for i in range(3):
        print(f"  val_phi[{i}] = {val_phi[i].tolist()}, val_psi[{i}] = {val_psi[i].tolist()}")
    print(f"\nFirst 3 examples:")
    for i in range(3):
        print(f"  val2_phi[{i}] = {val2_phi[i].tolist()}, val2_psi[{i}] = {val2_psi[i].tolist()}")
    save_dataset(val_phi, val_psi, "data/val1.pt")
    save_dataset(val2_phi, val2_psi, "data/val2.pt")    