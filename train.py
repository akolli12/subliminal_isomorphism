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

def evaluate(model, sequences):
    model.eval()

    with torch.no_grad():
        logits = model(sequences)

        psi_loss = F.cross_entropy(
            logits[:, config.PSI_LOGITS, :].reshape(-1, config.VOCAB),
            sequences[:, config.PSI].reshape(-1),
        )

        phi_inv_loss = F.cross_entropy(
            logits[:, config.PHI_INV_LOGITS, :].reshape(-1, config.VOCAB),
            sequences[:, config.PHI_INV].reshape(-1),
        )

    model.train()
    return psi_loss.item(), phi_inv_loss.item()


def train():
    torch.manual_seed(config.SEED)

    # Load dataset and build sequences once (small enough to keep on device)
    phi, psi = load_dataset()
    sequences = build_sequence(phi, psi).to(DEVICE)

    # val1
    v1_phi, v1_psi = load_dataset("data/val1.pt")
    val1_sequences = build_sequence(v1_phi, v1_psi).to(DEVICE)

    # val2
    v2_phi, v2_psi = load_dataset("data/val2.pt")
    val2_sequences = build_sequence(v2_phi, v2_psi).to(DEVICE)

    print(f"Loaded {sequences.shape[0]} sequences of length {sequences.shape[1]} "
          f"on {DEVICE}")

    # Model
    model = TinyTransformer().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=config.LR)
    n_dataset = sequences.shape[0]

    psi_hist     = []
    phi_inv_hist = []

    val1_psi_hist = []
    val1_phi_hist = []

    val2_psi_hist = []
    val2_phi_hist = []

    eval_steps = []

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

        if step % 10 == 0 or step == config.STEPS - 1:
            v1_psi, v1_phi = evaluate(model, val1_sequences)
            v2_psi, v2_phi = evaluate(model, val2_sequences)

            val1_psi_hist.append(v1_psi)
            val1_phi_hist.append(v1_phi)

            val2_psi_hist.append(v2_psi)
            val2_phi_hist.append(v2_phi)

            eval_steps.append(step)

            # print(
            #     f"step {step:5d} "
            #     f"train psi={psi_loss.item():.4f} "
            #     f"val1={v1_psi:.4f} "
            #     f"val2={v2_psi:.4f}"
            # )

        if step % 200 == 0 or step == config.STEPS - 1:
            # v1_psi, v1_phi = evaluate(model, val1_sequences)
            # v2_psi, v2_phi = evaluate(model, val2_sequences)

            # val1_psi_hist.append(v1_psi)
            # val1_phi_hist.append(v1_phi)

            # val2_psi_hist.append(v2_psi)
            # val2_phi_hist.append(v2_phi)

            # eval_steps.append(step)
            print(f"step {step:5d}  psi={psi_loss.item():.4f}  "
                  f"phi_inv={phi_inv_loss.item():.4f}")
            
            # print(
            #     f"step {step:5d} "
            #     f"train psi={psi_loss.item():.4f} "
            #     f"val1={v1_psi:.4f} "
            #     f"val2={v2_psi:.4f}"
            # )

    # Save checkpoint
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"\nSaved model to {config.MODEL_PATH}")

    # plot_curves(psi_hist, phi_inv_hist)

    plot_all_curves(psi_hist, phi_inv_hist, val1_psi_hist,
    val1_phi_hist,
    val2_psi_hist,
    val2_phi_hist,
    eval_steps)
    return model

def plot_all_curves(train_psi, train_phi,
                val1_psi, val1_phi,
                val2_psi, val2_phi,
                eval_steps):

    random_uniform_token = math.log(config.N)
    random_uniform_perm = math.lgamma(config.N + 1) / config.N

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # psi
    ax[0].plot(ema(train_psi), label='train')
    ax[0].plot(eval_steps, val1_psi, label='val1')
    ax[0].plot(eval_steps, val2_psi, label='val2')

    ax[0].axhline(random_uniform_perm, ls=':', label='log(n!)/n')
    ax[0].axhline(random_uniform_token, ls='--', label='log(n)')

    ax[0].set_title(r'$\psi$ loss')
    ax[0].set_xlabel('step')
    ax[0].set_ylabel('loss')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    # phi^-1
    ax[1].plot(ema(train_phi), label='train')
    ax[1].plot(eval_steps, val1_phi, label='val1')
    ax[1].plot(eval_steps, val2_phi, label='val2')

    ax[1].axhline(0, ls='--', label='optimum')

    ax[1].set_title(r'$\varphi^{-1}$ loss')
    ax[1].set_xlabel('step')
    ax[1].set_ylabel('loss')
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.PLOT_PATH, dpi=120)
    plt.show()


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