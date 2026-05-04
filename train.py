"""
Train TinyTransformer on the (phi, psi) dataset.

Three loss heads, each over a permutation block:
  - psi loss          (random target; population floor log(n!)/n; goes to ~0
                       under memorization on the fixed dataset)
  - psi_inv loss      (deterministic from psi; population floor 0)
  - phi ∘ psi^{-1}    (deterministic from (phi, psi); population floor 0)

Total loss = psi + psi_inv + phi_psi_inv. Each head is tracked separately
on train / val1 / val2 splits.
"""

import math
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config
from model import TinyTransformer
from generate_dataset import load_dataset, build_sequence

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def checkpoint_path_for(step):
    """`checkpoints/model.pt` → `checkpoints/model_<step>.pt`."""
    base, ext = os.path.splitext(config.MODEL_PATH)
    return f"{base}_{step}_{config.N}{ext}"


def save_checkpoint(model, step):
    path = checkpoint_path_for(step)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  [checkpoint] saved {path} (after {step} updates)")


def ema(xs, alpha=0.05):
    out, m = [], xs[0]
    for x in xs:
        m = alpha * x + (1 - alpha) * m
        out.append(m)
    return out


def block_loss(logits, sequences, block):
    """CE loss for predicting tokens in `block` from prior context.

    Causal mask: logits[k] predicts input[k+1]. So tokens at input
    positions [block.start, block.stop) are predicted by logits at
    positions [block.start - 1, block.stop - 1).
    """
    pred = logits[:, block.start - 1 : block.stop - 1, :]
    targ = sequences[:, block]
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), targ.reshape(-1))


def block_losses(logits, sequences):
    """Return (psi_loss, psi_inv_loss, phi_psi_inv_loss) as tensors."""
    return (
        block_loss(logits, sequences, config.PSI),
        block_loss(logits, sequences, config.PSI_INV),
        block_loss(logits, sequences, config.PHI_PSI_INV),
    )


@torch.no_grad()
def evaluate(model, sequences):
    model.eval()
    logits = model(sequences)
    psi, psi_inv, phi_psi_inv = block_losses(logits, sequences)
    model.train()
    return psi.item(), psi_inv.item(), phi_psi_inv.item()


def train():
    torch.manual_seed(config.SEED)

    # Load fixed dataset and build sequences once (small enough to keep on device)
    phi, psi = load_dataset(config.DATASET_PATH)
    sequences = build_sequence(phi, psi).to(DEVICE)

    v1_phi, v1_psi = load_dataset(config.VAL1_PATH)
    val1_sequences = build_sequence(v1_phi, v1_psi).to(DEVICE)

    v2_phi, v2_psi = load_dataset(config.VAL2_PATH)
    val2_sequences = build_sequence(v2_phi, v2_psi).to(DEVICE)

    print(f"Loaded {sequences.shape[0]} sequences of length {sequences.shape[1]} "
          f"on {DEVICE}")

    model = TinyTransformer().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=config.LR)
    n_dataset = sequences.shape[0]

    # Train histories (per step)
    train_psi_hist, train_psi_inv_hist, train_phi_psi_inv_hist = [], [], []
    # Val histories (per eval-step)
    val1_psi_hist, val1_psi_inv_hist, val1_phi_psi_inv_hist = [], [], []
    val2_psi_hist, val2_psi_inv_hist, val2_phi_psi_inv_hist = [], [], []
    eval_steps = []

    checkpoint_steps = set(config.CHECKPOINT_STEPS)

    for step in range(config.STEPS):
        idx = torch.randint(0, n_dataset, (config.BATCH,), device=DEVICE)
        x   = sequences[idx]

        logits = model(x)
        psi_l, psi_inv_l, phi_psi_inv_l = block_losses(logits, x)
        loss = psi_l + psi_inv_l + phi_psi_inv_l

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_psi_hist.append(psi_l.item())
        train_psi_inv_hist.append(psi_inv_l.item())
        train_phi_psi_inv_hist.append(phi_psi_inv_l.item())

        # Save intermediate checkpoint AFTER this step's gradient update has
        # finished — so e.g. step==499 gives a model that has seen 500 updates.
        if (step + 1) in checkpoint_steps:
            save_checkpoint(model, step + 1)

        if step % config.EVAL_EVERY == 0 or step == config.STEPS - 1:
            v1 = evaluate(model, val1_sequences)
            v2 = evaluate(model, val2_sequences)

            val1_psi_hist.append(v1[0])
            val1_psi_inv_hist.append(v1[1])
            val1_phi_psi_inv_hist.append(v1[2])

            val2_psi_hist.append(v2[0])
            val2_psi_inv_hist.append(v2[1])
            val2_phi_psi_inv_hist.append(v2[2])

            eval_steps.append(step)

        if step % 200 == 0 or step == config.STEPS - 1:
            print(f"step {step:5d}  psi={psi_l.item():.4f}  "
                  f"psi_inv={psi_inv_l.item():.4f}  "
                  f"phi∘psi^-1={phi_psi_inv_l.item():.4f}")

    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"\nSaved model to {config.MODEL_PATH}")

    plot_all_curves(
        train_psi_hist, train_psi_inv_hist, train_phi_psi_inv_hist,
        val1_psi_hist,  val1_psi_inv_hist,  val1_phi_psi_inv_hist,
        val2_psi_hist,  val2_psi_inv_hist,  val2_phi_psi_inv_hist,
        eval_steps,
    )
    return model


def plot_all_curves(
    train_psi, train_psi_inv, train_phi_psi_inv,
    val1_psi,  val1_psi_inv,  val1_phi_psi_inv,
    val2_psi,  val2_psi_inv,  val2_phi_psi_inv,
    eval_steps,
):
    random_uniform_token = math.log(config.N)
    random_uniform_perm  = math.lgamma(config.N + 1) / config.N

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    panels = [
        (ax[0], r'$\psi$ loss',
         train_psi, val1_psi, val2_psi,
         [(random_uniform_perm, ':',  r'$\log(n!)/n$ (perm-uniform)'),
          (random_uniform_token, '--', r'$\log(n)$ (token-uniform)')]),
        (ax[1], r'$\psi^{-1}$ loss',
         train_psi_inv, val1_psi_inv, val2_psi_inv,
         [(0.0, '--', 'optimum (deterministic)')]),
        (ax[2], r'$\varphi \circ \psi^{-1}$ loss',
         train_phi_psi_inv, val1_phi_psi_inv, val2_phi_psi_inv,
         [(0.0, '--', 'optimum (deterministic)')]),
    ]

    for axi, title, tr, v1, v2, baselines in panels:
        axi.plot(ema(tr), label='train (EMA)')
        axi.plot(eval_steps, v1, label='val1')
        axi.plot(eval_steps, v2, label='val2')
        for y, ls, lab in baselines:
            axi.axhline(y, ls=ls, color='gray', label=lab)
        axi.set_title(title)
        axi.set_xlabel('step')
        axi.set_ylabel('CE loss (nats)')
        axi.legend(fontsize=9)
        axi.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(config.PLOT_PATH, dpi=120)
    print(f"Saved plot to {config.PLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    train()
