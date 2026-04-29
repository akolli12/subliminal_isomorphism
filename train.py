"""
Train TinyTransformer on the (phi, psi) dataset.

Loss is split into two heads:
  - psi loss     (random target; theoretical floor log(n!)/n in population,
                  but goes to ~0 with a fixed dataset because of memorization)
  - phi_inv loss (deterministic target; goes to ~0 either way)

Both are tracked and plotted with their theoretical baselines.
"""

import math
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config
from model import TinyTransformer
from data.generate_dataset import load_dataset, build_sequence

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def ema(xs, alpha=0.05):
    """Exponential moving average for cleaner loss curves."""
    out, m = [], xs[0]
    for x in xs:
        m = alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def train():
    torch.manual_seed(config.SEED)

    # Load dataset and build sequences once (small enough to keep on device)
    phi, psi = load_dataset()
    sequences = build_sequence(phi, psi).to(DEVICE)
    print(f"Loaded {sequences.shape[0]} sequences of length {sequences.shape[1]} "
          f"on {DEVICE}")

    # Model
    model = TinyTransformer().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=config.LR)
    n_dataset = sequences.shape[0]

    psi_hist     = []
    phi_inv_hist = []

    for step in range(config.STEPS):
        # Sample a batch from the fixed dataset
        idx = torch.randint(0, n_dataset, (config.BATCH,), device=DEVICE)
        x   = sequences[idx]

        logits = model(x)

        psi_loss = F.cross_entropy(
            logits[:, config.PSI_LOGITS, :].reshape(-1, config.VOCAB),
            x[:, config.PSI].reshape(-1),
        )
        phi_inv_loss = F.cross_entropy(
            logits[:, config.PHI_INV_LOGITS, :].reshape(-1, config.VOCAB),
            x[:, config.PHI_INV].reshape(-1),
        )
        loss = psi_loss + phi_inv_loss

        opt.zero_grad()
        loss.backward()
        opt.step()

        psi_hist.append(psi_loss.item())
        phi_inv_hist.append(phi_inv_loss.item())

        if step % 200 == 0 or step == config.STEPS - 1:
            print(f"step {step:5d}  psi={psi_loss.item():.4f}  "
                  f"phi_inv={phi_inv_loss.item():.4f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"\nSaved model to {config.MODEL_PATH}")

    plot_curves(psi_hist, phi_inv_hist)
    return model


def plot_curves(psi_hist, phi_inv_hist):
    random_uniform_token = math.log(config.N)
    random_uniform_perm  = math.lgamma(config.N + 1) / config.N
    learnable_optimum    = 0.0

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))

    # --- psi head ---
    ax[0].plot(psi_hist,        alpha=0.25, color='C0', label='train (raw)')
    ax[0].plot(ema(psi_hist),               color='C0', lw=2, label='train (EMA)')
    ax[0].axhline(random_uniform_token, ls='--', color='red',
                  label=fr'uniform-token $\log(n)={random_uniform_token:.3f}$')
    ax[0].axhline(random_uniform_perm,  ls=':',  color='green',
                  label=fr'perm-aware $\log(n!)/n={random_uniform_perm:.3f}$')
    ax[0].set(title=r'$\psi$ loss (random target)',
              xlabel='step', ylabel='CE loss (nats)')
    ax[0].legend(loc='upper right', fontsize=9)
    ax[0].grid(alpha=0.3)

    # --- phi_inv head ---
    ax[1].plot(phi_inv_hist,        alpha=0.25, color='C1', label='train (raw)')
    ax[1].plot(ema(phi_inv_hist),               color='C1', lw=2, label='train (EMA)')
    ax[1].axhline(learnable_optimum, ls='--', color='green',
                  label='learnable optimum $= 0$')
    ax[1].axhline(random_uniform_token, ls=':',  color='red',
                  label=fr'uniform-token $\log(n)={random_uniform_token:.3f}$')
    ax[1].set(title=r'$\varphi^{-1}$ loss (deterministic target)',
              xlabel='step', ylabel='CE loss (nats)')
    ax[1].legend(loc='upper right', fontsize=9)
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.PLOT_PATH, dpi=120)
    print(f"Saved plot to {config.PLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    train()