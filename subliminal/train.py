"""Training loop for all prover variants.

A prover is specified by a layout plus a per-block loss spec:

  baseline (perm-only):   {psi: ce, psi_inv: ce, phi_psi_inv: ce}  (phi input-only)
  simulator-aligned:      {psi: uniform, psi_inv: ce, phi_psi_inv: ce}
  graph, phi given:       {psi: ce, psi_inv: ce, phi_psi_inv: ce}  (g0,g1,phi input-only)
  graph, phi learned:     {phi: ce, psi: ce, psi_inv: ce, phi_psi_inv: ce}

'ce' is standard next-token cross-entropy on the block. 'uniform' is the
simulator-aligned defense (paper Section 4.1): soft cross-entropy against the
autoregressive factorization of Unif(S_n) — at psi position t the target is
uniform over the n-t values not yet used. The witness-masked defense needs no
training changes (it is applied at inference; see model.py / sample.py).
"""

import json
import math
import os

import torch
import torch.nn.functional as F

from .layout import Layout
from .model import TinyTransformer
from .seeding import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def block_ce(logits, seqs, blk: slice) -> torch.Tensor:
    pred = logits[:, blk.start - 1:blk.stop - 1, :]
    targ = seqs[:, blk]
    return F.cross_entropy(pred.reshape(-1, pred.size(-1)), targ.reshape(-1))


def block_uniform_ce(logits, seqs, blk: slice, n: int) -> torch.Tensor:
    """Soft CE against uniform-on-remaining targets over a permutation block."""
    pred = logits[:, blk.start - 1:blk.stop - 1, :]        # (B, n, V)
    toks = seqs[:, blk]                                    # (B, n)
    one_hot = F.one_hot(toks, pred.size(-1)).float()
    used = torch.cat(
        [torch.zeros_like(one_hot[:, :1]), one_hot[:, :-1].cumsum(dim=1)], dim=1)
    remaining = (n - torch.arange(n, device=logits.device).float()).view(1, n, 1)
    target = (1.0 - used) / remaining                      # rows sum to 1
    return -(target * F.log_softmax(pred, dim=-1)).sum(-1).mean()


def losses(logits, seqs, layout: Layout, loss_spec: dict) -> dict:
    out = {}
    for name, kind in loss_spec.items():
        blk = layout[name]
        if kind == "ce":
            out[name] = block_ce(logits, seqs, blk)
        elif kind == "uniform":
            out[name] = block_uniform_ce(logits, seqs, blk, layout.n)
        else:
            raise ValueError(kind)
    return out


def train_prover(layout: Layout, loss_spec: dict, train_seqs: torch.Tensor,
                 val_seqs: torch.Tensor, *, steps: int, batch: int, lr: float,
                 seed: int, ckpt_path: str, d_model=256, n_heads=4, n_layers=8,
                 eval_every: int = 1000, log_every: int = 500) -> TinyTransformer:
    set_seed(seed)
    train_seqs = train_seqs.to(DEVICE)
    val_seqs = val_seqs.to(DEVICE)

    model = TinyTransformer(layout.vocab, layout.seq_len,
                            d_model=d_model, n_heads=n_heads,
                            n_layers=n_layers).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"step": [], "train": [], "val": []}
    for step in range(steps):
        idx = torch.randint(0, train_seqs.shape[0], (batch,), device=DEVICE)
        x = train_seqs[idx]
        loss_terms = losses(model(x), x, layout, loss_spec)
        loss = sum(loss_terms.values())
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % eval_every == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                val_terms = losses(model(val_seqs), val_seqs, layout, loss_spec)
            model.train()
            history["step"].append(step)
            history["train"].append({k: v.item() for k, v in loss_terms.items()})
            history["val"].append({k: v.item() for k, v in val_terms.items()})
        if step % log_every == 0 or step == steps - 1:
            terms = "  ".join(f"{k}={v.item():.4f}" for k, v in loss_terms.items())
            print(f"  step {step:6d}  {terms}", flush=True)

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(model.state_dict(), ckpt_path)
    with open(ckpt_path.replace(".pt", "_history.json"), "w") as f:
        json.dump(history, f)
    print(f"  saved {ckpt_path}")
    return model


def load_prover(ckpt_path: str, layout: Layout,
                d_model=256, n_heads=4, n_layers=8) -> TinyTransformer:
    model = TinyTransformer(layout.vocab, layout.seq_len, d_model=d_model,
                            n_heads=n_heads, n_layers=n_layers).to(DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    model.eval()
    return model
