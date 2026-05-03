import torch

import config
from model import TinyTransformer
from measure_marginals import sample_psi_batch
from sample_marginals import aggregate_by_constraint

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# witness coordinates (best i, v for each j)
def pick_witnessing_coordinates(model_results, n=config.N):
    """
    For each j in [n], find (i_j, v_j) maximizing
        spread(j, i, v) :=  max_u P_hat[psi(i)=v | phi(j)=u]
                         -  min_u P_hat[psi(i)=v | phi(j)=u].
    """
    witnesses = {}
    spreads   = {}

    for j in range(n):
        best_spread = -1.0
        best_i, best_v = 0, 0

        for i in range(n):
            for v in range(n):
                # Vary u over [n] for fixed (j, i, v)
                vals = torch.tensor([model_results[(j, u)][i, v]
                                     for u in range(n)])
                spread = (vals.max() - vals.min()).item()

                if spread > best_spread:
                    best_spread = spread
                    best_i, best_v = i, v

        witnesses[j] = (best_i, best_v)
        spreads[j]   = best_spread

    return witnesses, spreads


# tau
def build_tau(model_results, witnesses, n=config.N):
    """
    Tau[j, u] = model_results[(j, u)][i_j, v_j]
              = E_{phi : phi(j) = u}[ P[psi(i_j) = v_j | phi] ].
    """
    tau = torch.zeros(n, n)
    for j in range(n):
        i_j, v_j = witnesses[j]
        for u in range(n):
            tau[j, u] = model_results[(j, u)][i_j, v_j]
    return tau


# l function
@torch.no_grad()
def ell(phi, model, j, witnesses, n_samples=2000):
    """
    Estimate P_{psi ~ D_{x,phi}}[ psi(i_j) = v_j ] for the given phi - empirical probability that psi(i_j) = v_j under the model
    """
    i_j, v_j = witnesses[j]
    samples = sample_psi_batch(model, phi, n_samples)
    return (samples[:, i_j] == v_j).float().mean().item()


def ell_vector(phi, model, witnesses, n=config.N, n_samples=2000):
    """Compute ell(phi, model, j) for all j in [n]."""
    return torch.tensor([ell(phi, model, j, witnesses, n_samples)
                         for j in range(n)])


# ============================================================
# Reporting
# ============================================================
def print_witnesses(witnesses, spreads, n=config.N):
    print("\n=== Witnessing coordinates ===")
    print(f"  j  | (i_j, v_j) | max-spread (c - d)")
    print("  ---|------------|-------------------")
    for j in range(n):
        i_j, v_j = witnesses[j]
        print(f"  {j}  |  ({i_j}, {v_j})    |  {spreads[j]:.4f}")


def print_tau(tau, n=config.N):
    print("\n=== Tau table  Tau[j, u] = E[P[psi(i_j)=v_j] | phi(j)=u] ===")
    print(f"  rows = j (coordinate in phi)")
    print(f"  cols = u (value at phi(j))")
    header = "       " + "  ".join(f"u={u}   " for u in range(n))
    print(header)
    for j in range(n):
        row = f"  j={j}  " + "  ".join(f"{tau[j, u]:.4f}" for u in range(n))
        print(row)

def print_ell(phi, ell_vals, witnesses, n=config.N):
    print(f"\n=== ell vector for phi = {phi.tolist()} ===")
    print(f"  j  | (i_j, v_j) | ell(phi, j) = P[psi(i_j) = v_j | phi]")
    print(f"  ---|------------|---------------------------------------")
    for j in range(n):
        i_j, v_j = witnesses[j]
        print(f"  {j}  |  ({i_j}, {v_j})    |  {ell_vals[j]:.4f}")

def load_model(path=config.MODEL_PATH):
    model = TinyTransformer().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


def main():
    torch.manual_seed(config.SEED)
    model = load_model()

    model_results = aggregate_by_constraint(model)

    witnesses, spreads = pick_witnessing_coordinates(model_results)
    print_witnesses(witnesses, spreads)

    tau = build_tau(model_results, witnesses)
    print_tau(tau)

    # evaluate ell on a few example phis
    print("\n=== ell oracle evaluations ===")
    test_phis = [
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([3, 2, 1, 0]),
        torch.tensor([1, 3, 0, 2]),
    ]
    for phi in test_phis:
        ell_vals = ell_vector(phi, model, witnesses)
        print_ell(phi, ell_vals, witnesses)

    return witnesses, tau, model_results


if __name__ == "__main__":
    main()