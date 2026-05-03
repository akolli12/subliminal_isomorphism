"""
Aggregate marginal evaluator.
=============================

For each constraint (a, b) in [n] x [n]:
  1. Generate k1 phis with phi[a] = b
  2. For each phi, sample k2 psis from the trained model
  3. Build the n x n marginal matrix M[i, v] = P_hat[psi(i) = v | phi]
  4. Average the k1 matrices

Output: a dictionary {(a, b): avg_M}, where avg_M[i, v] estimates
        E_{phi : phi(a) = b} [ P[psi(i) = v | phi] ].
"""

import os

import torch
import matplotlib.pyplot as plt

import config
import torch.nn.functional as F
from model import TinyTransformer
from measure_marginals import (
    sample_psi_batch,
    sample_psi_batch_multi,
    marginal_matrix,
    uniform_null_samples,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

K1 = 20
K2 = 200

PLOT_PATH = "aggregate_marginals.png"


# phi generation with fixed constraint phi[a] = b
def generate_phis_with_constraint(n, a, b, k1, generator=None):
    """Sample k1 phis uniformly at random from {phi in S_n : phi[a] = b}.

    Vectorized: produce k1 random permutations of the n-1 "other" values
    using the argsort-of-random-keys trick (each row of `keys` argsorted
    is a uniform random permutation of [0, n-1)), then scatter into a
    (k1, n) tensor with `b` placed at column `a`.
    """
    other_positions = torch.tensor([i for i in range(n) if i != a], dtype=torch.long)
    other_values    = torch.tensor([v for v in range(n) if v != b], dtype=torch.long)
    n_other = n - 1

    keys = torch.rand((k1, n_other), generator=generator)
    perms = keys.argsort(dim=1)                       # (k1, n_other), each row is a perm
    perm_values = other_values[perms]                 # (k1, n_other)

    phis = torch.zeros((k1, n), dtype=torch.long)
    phis[:, a] = b
    phis[:, other_positions] = perm_values
    return phis
    # --- old, equivalent (Python loop): ----------------------------------
    # other_positions = [i for i in range(n) if i != a]
    # other_values    = [v for v in range(n) if v != b]
    # n_other = n - 1
    # phis = []
    # for _ in range(k1):
    #     perm = torch.randperm(n_other, generator=generator)
    #     phi = torch.zeros(n, dtype=torch.long)
    #     phi[a] = b
    #     for idx, pos in enumerate(other_positions):
    #         phi[pos] = other_values[perm[idx]]
    #     phis.append(phi)
    # return torch.stack(phis)
    # ---------------------------------------------------------------------


def aggregate_by_constraint(model, n=config.N, k1=K1, k2=K2):
    """Returns dict {(a, b): avg_M} where avg_M[i, v] estimates
    E_{phi : phi(a)=b}[ P[psi(i) = v | phi] ].

    Vectorized: for each (a, b) cell, sample k1 phis, replicate each k2
    times, and run ONE batched sample call of size k1*k2. Then collapse
    to a single marginal (the mean-of-marginals reduces to the marginal
    over all k1*k2 samples since each phi gets the same count k2).
    """
    results = {}
    total_cells = n * n
    cell_idx = 0

    for a in range(n):
        for b in range(n):
            cell_idx += 1
            phis = generate_phis_with_constraint(n, a, b, k1)         # (k1, n)
            phis_rep = phis.repeat_interleave(k2, dim=0)              # (k1*k2, n)
            samples = sample_psi_batch_multi(model, phis_rep)         # (k1*k2, n)

            # Equal counts per phi → mean-of-per-phi-marginals == overall mean.
            avg_M = F.one_hot(samples.long(), n).float().mean(dim=0)  # (n, n)

            results[(a, b)] = avg_M
            print(f"  cell {cell_idx}/{total_cells}  (a={a}, b={b})  done")
    return results
    # --- old, equivalent (per-phi inference loop): -----------------------
    # results = {}
    # for a in range(n):
    #     for b in range(n):
    #         phis = generate_phis_with_constraint(n, a, b, k1)
    #         avg_M = torch.zeros(n, n)
    #         for phi in phis:
    #             samples = sample_psi_batch(model, phi, k2)
    #             avg_M += marginal_matrix(samples)
    #         avg_M /= k1
    #         results[(a, b)] = avg_M
    # return results
    # ---------------------------------------------------------------------


# null baseline
def aggregate_null(n=config.N, k1=K1, k2=K2):
    """
    noise floor: the null max-deviation per cell under genuine uniformity
    """
    results = {}
    for a in range(n):
        for b in range(n):
            avg_M = torch.zeros(n, n)
            for _ in range(k1):
                samples = uniform_null_samples(n, k2)
                avg_M += marginal_matrix(samples)
            avg_M /= k1
            results[(a, b)] = avg_M
    return results


def echo_signal(results, n=config.N):
    """D[a, b] = avg_M[(a,b)][a, b]. Should be 1/n under no leakage."""
    D = torch.zeros(n, n)
    for a in range(n):
        for b in range(n):
            D[a, b] = results[(a, b)][a, b]
    return D


def max_deviation_per_cell(results, n=config.N):
    """delta[a, b] = max_{i,v} |avg_M[(a,b)][i, v] - 1/n|."""
    delta = torch.zeros(n, n)
    for a in range(n):
        for b in range(n):
            delta[a, b] = (results[(a, b)] - 1.0 / n).abs().max()
    return delta


# plot
def plot_summary(model_D, null_D, model_delta, null_delta, n=config.N,
                 path=PLOT_PATH):
    target = 1.0 / n
    fig, ax = plt.subplots(2, 2, figsize=(11, 9))

    # echo signal — model
    im00 = ax[0, 0].imshow(model_D, vmin=0, vmax=2 * target,
                           cmap='RdBu_r', extent=[0, n, n, 0])
    ax[0, 0].set(title=r'MODEL  $D[a,b] = \hat{P}[\psi(a)=b \mid \varphi(a)=b]$',
                 xlabel='b', ylabel='a')
    for a in range(n):
        for b in range(n):
            # print(a, b)
            ax[0, 0].text(b + 0.5,
            a + 0.5, f"{model_D[a,b]:.3f}",
                          ha='center', va='center', fontsize=9)
    plt.colorbar(im00, ax=ax[0, 0])

    # echo signal — null
    im01 = ax[0, 1].imshow(null_D, vmin=0, vmax=2 * target, cmap='RdBu_r', extent=[0, n, n, 0])
    ax[0, 1].set(title=fr'NULL   (target $1/n = {target:.3f}$)',
                 xlabel='b', ylabel='a')
    for a in range(n):
        for b in range(n):
            ax[0, 1].text(b+0.5, a+0.5, f"{null_D[a,b]:.3f}",
                          ha='center', va='center', fontsize=9)
    plt.colorbar(im01, ax=ax[0, 1])

    # max deviation — model
    vmax = max(model_delta.max().item(), null_delta.max().item())
    im10 = ax[1, 0].imshow(model_delta, vmin=0, vmax=vmax, cmap='Reds', extent=[0, n, n, 0])
    ax[1, 0].set(title=r'MODEL  $\delta[a,b] = \max_{i,v} |\hat{M}[i,v] - 1/n|$',
                 xlabel='b', ylabel='a')
    for a in range(n):
        for b in range(n):
            ax[1, 0].text(b+0.5, a+0.5, f"{model_delta[a,b]:.3f}",
                          ha='center', va='center', fontsize=9)
    plt.colorbar(im10, ax=ax[1, 0])

    # max deviation — null (noise floor)
    im11 = ax[1, 1].imshow(null_delta, vmin=0, vmax=vmax, cmap='Reds', extent=[0, n, n, 0])
    ax[1, 1].set(title=r'NULL   $\delta[a,b]$  (sampling noise floor)',
                 xlabel='b', ylabel='a')
    for a in range(n):
        for b in range(n):
            ax[1, 1].text(b+0.5, a+0.5, f"{null_delta[a,b]:.3f}",
                          ha='center', va='center', fontsize=9)
    plt.colorbar(im11, ax=ax[1, 1])

    plt.tight_layout()
    plt.savefig(path, dpi=120)
    print(f"\nSaved summary plot to {path}")
    plt.show()


# dictionary printed
def print_full_dict(results, label, n=config.N, max_cells=4):
    print(f"\n{label}: showing first {max_cells} cells")
    print("-" * 60)
    cells = list(results.keys())[:max_cells]
    for (a, b) in cells:
        M = results[(a, b)]
        print(f"\n  (a={a}, b={b})  E[ M | phi({a})={b} ]:")
        for i in range(n):
            row = "    " + " ".join(f"{M[i, v]:.3f}" for v in range(n))
            print(row)

def load_model(path=config.MODEL_PATH):
    model = TinyTransformer().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def main():
    torch.manual_seed(config.SEED)
    n = config.N
    print(f"n = {n},  k1 = {K1} phis per (a,b),  k2 = {K2} psis per phi")
    print(f"Per-entry sample count: k1 * k2 = {K1 * K2}")
    print(f"Approx 1-sigma noise per entry: "
          f"{((1.0/n)*(1-1.0/n)/(K1*K2))**0.5:.4f}")

    model = load_model()
    model_results = aggregate_by_constraint(model)

    null_results = aggregate_null()

    model_D     = echo_signal(model_results)
    null_D      = echo_signal(null_results)
    model_delta = max_deviation_per_cell(model_results)
    null_delta  = max_deviation_per_cell(null_results)

    print(f"\nModel: echo D[a,b] (target {1.0/n:.3f}):")
    print(model_D.numpy())
    print(f"\nTruly uniform: echo D[a,b]:")
    print(null_D.numpy())
    print(f"\nModel: max-deviation delta[a,b]:")
    print(model_delta.numpy())
    print(f"\nTruly uniform: max-deviation delta[a,b]:")
    print(null_delta.numpy())
    print(f"\nRatio:  model_delta.mean() / truly_uniform_delta.mean() = "
          f"{(model_delta.mean() / null_delta.mean()).item():.2f}x")

    print_full_dict(model_results, "Model: full matrices", max_cells=16)
    plot_summary(model_D, null_D, model_delta, null_delta)


if __name__ == "__main__":
    main()