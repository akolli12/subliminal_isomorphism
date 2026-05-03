# (1) pick a fixed phi
# (2) sample k different psi’s from the model conditioned on this phi
#     - model outputs probability distribution at each psi position
#     - to draw samples: sample from the distribution and feed in that sampled token autoregressively
# (3) empirical marginal: for each position i and value v, count what fraction of samples had psi(i) = v
# - this should give an nxn matrix of empirical probabilities for each phi
# - if the model is actually uniform, every entry should be 1/n - we want to find deviations from 1/n
#     - To call something "leakage" with confidence, the deviation has to substantially exceed this null fluctuation
#         - compute max_{i,v} |P̂[ψ(i) = v | φ] − 1/n| and compare to the same quantity computed on a known-uniform sampler (just sample K random permutations from torch.randperm). If the model's max-deviation is meaningfully larger —> LEAKAGE!

import math

import torch
import torch.nn.functional as F

import config
from model import TinyTransformer
from generate_dataset import build_sequence

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(path=config.MODEL_PATH):
    model = TinyTransformer().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# =====================================================================
# Autoregressive sampling
# =====================================================================
@torch.no_grad()
def sample_psi_batch_multi(model, phis):
    """Sample one psi per row in `phis`.

    Args:
      phis: LongTensor (B, N). Each row is one phi conditioning context.
    Returns:
      LongTensor (B, N) on CPU. Row b is one sampled psi conditioned on phis[b].

    We only feed [phis | psi_buf] (length 2N): the downstream
    (psi_inv, phi∘psi^{-1}) blocks aren't needed because psi positions only
    attend backward under the causal mask.
    """
    model.eval()                                                     # safety
    N = config.N
    B = phis.shape[0]
    psi_start = config.PSI.start                                     # = N

    psi_buf = torch.zeros((B, N), dtype=torch.long, device=DEVICE)
    seq = torch.cat([phis.to(DEVICE), psi_buf], dim=1)               # (B, 2N)

    for j in range(N):
        logits = model(seq)
        # logits[k] predicts input[k+1]; psi_j sits at input pos psi_start+j.
        step_logits = logits[:, psi_start - 1 + j, :].clone()

        # ---- vectorized "no repeats" mask (replaces a Python `for prev`) ----
        if j > 0:
            already = seq[:, psi_start : psi_start + j]              # (B, j)
            step_logits.scatter_(1, already, float('-inf'))
        # --- old per-step masking, equivalent: -------------------------------
        # for prev in range(j):
        #     already = seq[:, psi_start + prev]
        #     step_logits[torch.arange(B, device=DEVICE), already] = float('-inf')
        # ---------------------------------------------------------------------

        probs = F.softmax(step_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        seq[:, psi_start + j] = sampled

    return seq[:, psi_start:psi_start + N].cpu()


@torch.no_grad()
def sample_psi_batch(model, phi, n_samples):
    """Autoregressively sample n_samples psi-permutations conditioned on a single phi.

    Thin wrapper around `sample_psi_batch_multi` for backward compat with
    callers (building_tau.ell, evaluate_phi, ...) that pass one phi and
    a desired sample count.
    """
    phis = phi.unsqueeze(0).expand(n_samples, -1).contiguous()
    return sample_psi_batch_multi(model, phis)


# =====================================================================
# Empirical marginals
# =====================================================================
def marginal_matrix(samples, n=config.N):
    """Build M[i, v] = fraction of `samples` with sample[i] == v.

    `samples` is a LongTensor of shape (K, N) of permutation values.
    """
    return F.one_hot(samples.long(), n).float().mean(dim=0)
    # --- old, equivalent (n^2 nested loops): -----------------------------
    # K = samples.shape[0]
    # M = torch.zeros(n, n)
    # for i in range(n):
    #     for v in range(n):
    #         M[i, v] = (samples[:, i] == v).float().mean()
    # return M
    # ---------------------------------------------------------------------


def uniform_null_samples(n, K, seed=None):
    # genuine uniform random permutations K times - baseline to compare against
    g = torch.Generator()
    if seed is not None:
        g.manual_seed(seed)
    return torch.stack([torch.randperm(n, generator=g) for _ in range(K)])


# deviation of empirical marginals from 1/n - what we care about wanting to be high/large
def max_deviation(M, n=config.N):
    """max_{i,v} |M[i,v] - 1/n|."""
    # PrDx,φ [ψ(i) = v]] - 1/n
    return (M - 1.0 / n).abs().max().item()


def expected_null_deviation(n, K):
    """
    approximate scale of max |M[i,v] - 1/n| under genuine uniform sampling - each entry has stddev ~ sqrt((1/n)(1-1/n)/K)
    """
    return math.sqrt((1.0 / n) * (1.0 - 1.0 / n) / K)


# testing against actual uniform
def evaluate_phi(model, phi, K=2000):
    # model + baseline marginal comparison for singular phi
    n = config.N

    # Model marginals
    model_samples = sample_psi_batch(model, phi, K)
    M_model = marginal_matrix(model_samples)
    dev_model = max_deviation(M_model)

    # uniform marginals
    null_samples = uniform_null_samples(n, K)
    M_null = marginal_matrix(null_samples)
    dev_null = max_deviation(M_null)
    print(M_null)

    sigma_null = expected_null_deviation(n, K)

    return {
        "phi":          phi.tolist(),
        "M_model":      M_model,
        "M_null":       M_null,
        "dev_model":    dev_model,
        "dev_null":     dev_null,
        "sigma_null":   sigma_null,
    }


def print_report(result):
    n = config.N
    print(f"\nphi = {result['phi']}")
    print(f"uniform target : {1.0 / n:.4f}  (= 1/n)")
    print(f"expected 1-sigma noise : {result['sigma_null']:.4f}")
    print(f"Truly uniform - max deviation : {result['dev_null']:.4f}")
    print(f"Model - max deviation : {result['dev_model']:.4f}")
    ratio = result['dev_model'] / max(result['dev_null'], 1e-9)
    print(f"  ratio MODEL / null      : {ratio:.2f}x")

    print(f"\n  Model marginal matrix  M[i, v] = P_hat[psi(i) = v | phi]:")
    print(f"  rows = position i, cols = value v")
    for i in range(n):
        row = "    " + " ".join(f"{result['M_model'][i, v]:.3f}"
                                for v in range(n))
        print(row)


def main():
    torch.manual_seed(config.SEED)
    model = load_model()
    K = 2000

    test_phis = [
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([1, 0, 2, 3]),
        torch.tensor([1, 2, 3, 0]),
        torch.tensor([3, 2, 1, 0]),
    ]

    print(f"Sampling K={K} psi's per phi from the trained model.\n")
    print("=" * 60)

    for phi in test_phis:
        result = evaluate_phi(model, phi, K=K)
        print_report(result)
        print("-" * 60)


if __name__ == "__main__":
    main()