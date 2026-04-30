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
from data.generate_dataset import build_sequence

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model(path=config.MODEL_PATH):
    model = TinyTransformer().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


# masked autoregressive sampling
@torch.no_grad()
def sample_psi_batch(model, phi, n_samples):
    N = config.N
    B = n_samples

    # Build the prefix [BOS] phi [SEP], replicated B times.
    # We don't need phi_inv for sampling psi (it appears later in the sequence,
    # after psi). We will just leave space for psi tokens to be filled in.
    bos = torch.full((B, 1), config.BOS, dtype=torch.long, device=DEVICE)
    sep = torch.full((B, 1), config.SEP, dtype=torch.long, device=DEVICE)
    phi_b = phi.unsqueeze(0).expand(B, -1).to(DEVICE)
    psi_buf = torch.zeros((B, N), dtype=torch.long, device=DEVICE)  # placeholder

    # We only need the prefix up through the psi block for sampling psi.
    # Sequence layout up to end-of-psi: [BOS] phi [SEP] psi  -> length 2N + 2.
    seq = torch.cat([bos, phi_b, sep, psi_buf], dim=1)

    psi_start = config.PSI.start

    for j in range(N):
        logits = model(seq)
        step_logits = logits[:, psi_start - 1 + j, :]      # logits predicting psi_{j+1}

        # masking tokens not related to permutation
        step_logits = step_logits.clone()
        step_logits[:, config.BOS] = float('-inf')
        step_logits[:, config.SEP] = float('-inf')

        # masking so not repeat numbers in permutation
        for prev in range(j):
            already = seq[:, psi_start + prev] 
            step_logits[torch.arange(B, device=DEVICE), already] = float('-inf')

        # sampling
        probs = F.softmax(step_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)

        # write back into sequence for future steps
        seq[:, psi_start + j] = sampled

    return seq[:, psi_start:psi_start + N].cpu()


# matrix of probabilities that sample[i] = v for each i, v pair (4x4 matrix to my understanding)
def marginal_matrix(samples, n=config.N):
    """
    Build M[i, v] = fraction of samples with sample[i] == v.
    PrDx,φ [ψ(i) = v]] = 1/n
    """
    K = samples.shape[0]
    M = torch.zeros(n, n)
    for i in range(n):
        for v in range(n):
            M[i, v] = (samples[:, i] == v).float().mean()
    return M


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
    Approximate scale of max |M[i,v] - 1/n| under genuine uniform sampling.
    Each entry has stddev ~ sqrt((1/n)(1-1/n)/K). The max over n^2 entries
    is roughly 2-3 sigma. Returns a 1-sigma reference, not the max itself.
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
    print(f"  uniform target          : {1.0 / n:.4f}  (= 1/n)")
    print(f"  expected 1-sigma noise  : {result['sigma_null']:.4f}")
    print(f"  null   max deviation    : {result['dev_null']:.4f}")
    print(f"  MODEL  max deviation    : {result['dev_model']:.4f}")
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
        torch.tensor([0, 1, 2, 3]),  # identity
        torch.tensor([1, 0, 2, 3]),  # transposition (0,1)
        torch.tensor([1, 2, 3, 0]),  # 4-cycle
        torch.tensor([3, 2, 1, 0]),  # reversal
    ]

    print(f"Sampling K={K} psi's per phi from the trained model.\n")
    print("=" * 60)

    for phi in test_phis:
        result = evaluate_phi(model, phi, K=K)
        print_report(result)
        print("-" * 60)


if __name__ == "__main__":
    main()