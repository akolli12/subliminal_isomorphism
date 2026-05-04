"""Ablation training modes.

Two independent flags, combinable:

  --psi_loss_mode {standard,uniform}
      standard (default): cross-entropy against the actual psi tokens.
      uniform           : soft CE against the uniform-on-remaining target
                          distribution. At psi position t, target is uniform
                          over the (N - t) values not already used. Floor:
                          log(n!)/n. Removes memorization-driven leakage.

  --mask_phi
      Zero out the phi positions in the input (both training and eval) so the
      model never sees real phi values. psi generation is unconditional. The
      phi*psi^-1 task becomes ~impossible; its loss will plateau near log(n).

Usage:
    conda run --no-capture-output -n aug-spm python train_ablation.py
    conda run --no-capture-output -n aug-spm python train_ablation.py --psi_loss_mode uniform
    conda run --no-capture-output -n aug-spm python train_ablation.py --mask_phi
    conda run --no-capture-output -n aug-spm python train_ablation.py --psi_loss_mode uniform --mask_phi

Outputs are saved with a mode-tag suffix:
    checkpoints/model_psi-uniform.pt
    checkpoints/model_mask-phi.pt
    checkpoints/model_psi-uniform_mask-phi.pt
    checkpoints/model_<tag>_<step>.pt   (for each step in CHECKPOINT_STEPS)
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import config
from model import TinyTransformer
from generate_dataset import load_dataset, build_sequence

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--psi_loss_mode', type=str, default='standard',
                   choices=['standard', 'uniform'])
    p.add_argument('--mask_phi', action='store_true')
    return p.parse_args()


def ema(xs, alpha=0.05):
    out, m = [], xs[0]
    for x in xs:
        m = alpha * x + (1 - alpha) * m
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Input / loss helpers
# ---------------------------------------------------------------------------
def apply_phi_mask(seq, mask_phi):
    """If mask_phi: replace phi positions with 0. Returns a clone (input only)."""
    if not mask_phi:
        return seq
    seq = seq.clone()
    seq[:, config.PHI] = 0
    return seq


def block_loss(logits, sequences, block):
    """Standard cross-entropy for predicting tokens in `block`."""
    pred = logits[:, block.start - 1 : block.stop - 1, :]
    targ = sequences[:, block]
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), targ.reshape(-1))


def uniform_psi_loss(logits, sequences):
    """Soft CE against uniform-on-remaining target at each psi position.

    At position t (0-indexed within psi), the target distribution is
        target[v] = 1/(N - t)   if v is NOT in {psi[0], ..., psi[t-1]}
                  = 0           otherwise
    so target[t, :] sums to 1 (it's a valid distribution over the unused
    values). The loss is the standard soft-CE:
        -sum_v target[v] * log_softmax(logits)[v], averaged over (b, t).
    """
    n = config.N
    psi_block = config.PSI
    psi_logits = logits[:, psi_block.start - 1 : psi_block.stop - 1, :]   # (B, N, V)
    psi_tokens = sequences[:, psi_block]                                  # (B, N)

    psi_one_hot = F.one_hot(psi_tokens, n).float()                        # (B, N, V)
    # used_mask[b, t, v] = 1 if v appeared at any psi[b, 0..t-1]
    used_mask = torch.cat([
        torch.zeros_like(psi_one_hot[:, :1, :]),
        psi_one_hot[:, :-1, :].cumsum(dim=1),
    ], dim=1)

    n_remaining = (n - torch.arange(n, device=logits.device).float()).view(1, n, 1)
    target_dist = (1.0 - used_mask) / n_remaining                         # (B, N, V), rows sum to 1

    log_probs = F.log_softmax(psi_logits, dim=-1)
    return -(target_dist * log_probs).sum(dim=-1).mean()


def block_losses(logits, sequences, psi_loss_mode):
    """Returns (psi_loss, psi_inv_loss, phi_psi_inv_loss)."""
    if psi_loss_mode == 'uniform':
        psi = uniform_psi_loss(logits, sequences)
    else:
        psi = block_loss(logits, sequences, config.PSI)
    psi_inv     = block_loss(logits, sequences, config.PSI_INV)
    phi_psi_inv = block_loss(logits, sequences, config.PHI_PSI_INV)
    return psi, psi_inv, phi_psi_inv


@torch.no_grad()
def evaluate(model, sequences, mask_phi, psi_loss_mode):
    model.eval()
    inp = apply_phi_mask(sequences, mask_phi)
    logits = model(inp)
    psi, psi_inv, phi_psi_inv = block_losses(logits, sequences, psi_loss_mode)
    model.train()
    return psi.item(), psi_inv.item(), phi_psi_inv.item()


# ---------------------------------------------------------------------------
# Checkpoint paths
# ---------------------------------------------------------------------------
def mode_tag(args):
    parts = []
    if args.psi_loss_mode != 'standard':
        parts.append(f'psi-{args.psi_loss_mode}')
    if args.mask_phi:
        parts.append('mask-phi')
    return '_'.join(parts) or 'standard'


def checkpoint_path_for(step, tag):
    """checkpoints/model_<tag>_<step>.pt"""
    base, ext = os.path.splitext(config.MODEL_PATH)
    return f"{base}_{tag}_{step}_{config.N}{ext}"


def final_path_for(tag):
    """checkpoints/model_<tag>.pt"""
    base, ext = os.path.splitext(config.MODEL_PATH)
    return f"{base}_{tag}{ext}"


def save_checkpoint(model, step, tag):
    path = checkpoint_path_for(step, tag)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"  [checkpoint] saved {path}")


# ---------------------------------------------------------------------------
# Train loop
# ---------------------------------------------------------------------------
def train(args):
    torch.manual_seed(config.SEED)
    tag = mode_tag(args)
    print(f"Training mode: {tag}")

    phi, psi = load_dataset(config.DATASET_PATH)
    sequences = build_sequence(phi, psi).to(DEVICE)

    v1_phi, v1_psi = load_dataset(config.VAL1_PATH)
    val1_sequences = build_sequence(v1_phi, v1_psi).to(DEVICE)

    v2_phi, v2_psi = load_dataset(config.VAL2_PATH)
    val2_sequences = build_sequence(v2_phi, v2_psi).to(DEVICE)
    print(f"Loaded {sequences.shape[0]} sequences of length {sequences.shape[1]} on {DEVICE}")

    model = TinyTransformer().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=config.LR)
    n_dataset = sequences.shape[0]

    train_hist = {'psi': [], 'psi_inv': [], 'phi_psi_inv': []}
    val1_hist  = {'psi': [], 'psi_inv': [], 'phi_psi_inv': []}
    val2_hist  = {'psi': [], 'psi_inv': [], 'phi_psi_inv': []}
    eval_steps = []
    checkpoint_steps = set(config.CHECKPOINT_STEPS)

    for step in range(config.STEPS):
        idx = torch.randint(0, n_dataset, (config.BATCH,), device=DEVICE)
        x   = sequences[idx]
        x_in = apply_phi_mask(x, args.mask_phi)         # input to model

        logits = model(x_in)
        # Note: targets always come from the ORIGINAL x — masking only affects input.
        psi_l, psi_inv_l, phi_psi_inv_l = block_losses(logits, x, args.psi_loss_mode)
        loss = psi_l + psi_inv_l + phi_psi_inv_l

        opt.zero_grad()
        loss.backward()
        opt.step()

        train_hist['psi'].append(psi_l.item())
        train_hist['psi_inv'].append(psi_inv_l.item())
        train_hist['phi_psi_inv'].append(phi_psi_inv_l.item())

        if (step + 1) in checkpoint_steps:
            save_checkpoint(model, step + 1, tag)

        if step % config.EVAL_EVERY == 0 or step == config.STEPS - 1:
            v1 = evaluate(model, val1_sequences, args.mask_phi, args.psi_loss_mode)
            v2 = evaluate(model, val2_sequences, args.mask_phi, args.psi_loss_mode)
            for k, v in zip(('psi', 'psi_inv', 'phi_psi_inv'), v1):
                val1_hist[k].append(v)
            for k, v in zip(('psi', 'psi_inv', 'phi_psi_inv'), v2):
                val2_hist[k].append(v)
            eval_steps.append(step)

        if step % 200 == 0 or step == config.STEPS - 1:
            print(f"step {step:5d}  psi={psi_l.item():.4f}  "
                  f"psi_inv={psi_inv_l.item():.4f}  "
                  f"phi*psi^-1={phi_psi_inv_l.item():.4f}")

    final = final_path_for(tag)
    os.makedirs(os.path.dirname(final), exist_ok=True)
    torch.save(model.state_dict(), final)
    print(f"\nSaved final model to {final}")

    plot_all_curves(train_hist, val1_hist, val2_hist, eval_steps, tag, args.psi_loss_mode)
    return model


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_all_curves(train_hist, val1_hist, val2_hist, eval_steps, tag, psi_loss_mode):
    n = config.N
    log_n             = math.log(n)
    log_n_fact_over_n = math.lgamma(n + 1) / n

    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    panels = [
        ('psi',         r'$\psi$ loss',
         [(log_n_fact_over_n, ':',  r'$\log(n!)/n$'),
          (log_n,             '--', r'$\log(n)$')]),
        ('psi_inv',     r'$\psi^{-1}$ loss',
         [(0.0, '--', 'optimum')]),
        ('phi_psi_inv', r'$\varphi \circ \psi^{-1}$ loss',
         [(0.0, '--', 'optimum')]),
    ]
    for axi, (key, title, baselines) in zip(ax, panels):
        axi.plot(ema(train_hist[key]), label='train (EMA)')
        axi.plot(eval_steps, val1_hist[key], label='val1')
        axi.plot(eval_steps, val2_hist[key], label='val2')
        for y, ls, lab in baselines:
            axi.axhline(y, ls=ls, color='gray', label=lab)
        axi.set(title=title, xlabel='step', ylabel='CE loss (nats)')
        axi.legend(fontsize=9); axi.grid(alpha=0.3)
    plt.tight_layout()
    plot_path = config.PLOT_PATH.replace('.png', f'_{tag}.png')
    plt.savefig(plot_path, dpi=120)
    print(f"Saved plot to {plot_path}")
    plt.show()


if __name__ == "__main__":
    train(parse_args())
