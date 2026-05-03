import math

import torch
from scipy.optimize import linear_sum_assignment

import config
from building_tau import (
    load_model,
    pick_witnessing_coordinates,
    build_tau,
    ell_vector,
)
from sample_marginals import aggregate_by_constraint

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12


def build_cost_matrix(phi, model, witnesses, tau, n=config.N, n_samples=2000):
    """
    C[j, u] = log(ell(phi, j)) - log(tau[j, u]) - do we want log?
    """
    ell_vals = ell_vector(phi, model, witnesses, n=n, n_samples=n_samples)

    log_ell = torch.log(ell_vals.clamp_min(EPS))            # (n,)
    log_tau = torch.log(tau.clamp_min(EPS))                 # (n, n)

    C = log_ell.unsqueeze(1) - log_tau                      # (n, n)
    return C, ell_vals


# recover phi via min-cost assignment
def recover_phi(phi, model, witnesses, tau, n=config.N, n_samples=2000):
    C, ell_vals = build_cost_matrix(phi, model, witnesses, tau,
                                    n=n, n_samples=n_samples)

    # scipy expects numpy
    row_ind, col_ind = linear_sum_assignment(C.numpy())
    phi_hat = torch.tensor(col_ind, dtype=torch.long)

    success = bool((phi_hat == phi).all().item())
    return phi_hat, success, C, ell_vals


def assignment_cost(C, sigma):
    """sum_j C[j, sigma(j)]."""
    n = C.shape[0]
    return sum(C[j, sigma[j].item()].item() for j in range(n))

def main():
    torch.manual_seed(config.SEED)
    n = config.N

    model = load_model()
    model_results = aggregate_by_constraint(model)

    witnesses, spreads = pick_witnessing_coordinates(model_results)
    tau = build_tau(model_results, witnesses)

    print(f"\nWitnesses (j: (i_j, v_j), spread):")
    for j in range(n):
        print(f"  j={j}: {witnesses[j]},  spread={spreads[j]:.4f}")

    # Run the attack on every permutation in S_n (for n=4, all 24)
    from itertools import permutations
    all_phis = [torch.tensor(list(p)) for p in permutations(range(n))]

    print(f"\n Running attack on all {len(all_phis)} permutations of S_{n} ")
    print(f"  phi (true)         | phi_hat (recovered)  | match | cost(true)  | cost(recov)")

    n_correct = 0
    cost_diffs = []

    for phi in all_phis:
        phi_hat, success, C, ell_vals = recover_phi(phi, model, witnesses, tau)
        cost_true  = assignment_cost(C, phi)
        cost_recov = assignment_cost(C, phi_hat)
        cost_diffs.append(cost_true - cost_recov)

        n_correct += int(success)
        flag = "OK   " if success else "WRONG"
        print(f"  {phi.tolist()}        |  {phi_hat.tolist()}         "
              f"|  {flag} |  {cost_true:+.4f}   |  {cost_recov:+.4f}")

    total = len(all_phis)
    print(f"\nRecovery accuracy: {n_correct}/{total} ({100*n_correct/total:.1f}%)")

    cost_diffs_t = torch.tensor(cost_diffs)
    print(f"\nCost-gap diagnostics (cost(true) - cost(recovered)):")
    print(f"  mean: {cost_diffs_t.mean().item():+.4f}")
    print(f"  max : {cost_diffs_t.max().item():+.4f}   (positive = true was suboptimal)")
    print(f"  min : {cost_diffs_t.min().item():+.4f}   (zero or negative = true was optimal)")


if __name__ == "__main__":
    main()