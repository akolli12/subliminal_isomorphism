"""
TinyTransformer: small decoder-only transformer.

Trained as a next-token language model over the sequence
  [ phi | psi | psi_inv | phi ∘ psi^{-1} ]
with a causal mask so position k only attends to positions < k.
Loss is applied only to the psi, psi_inv, and phi ∘ psi^{-1} blocks
(see train.py).

The architecture is parameterized by N (the permutation size). VOCAB and
SEQ_LEN are derived as N and 4*N respectively. By default N is read from
config.N, but you can override it by passing `TinyTransformer(N=4)` — useful
when loading checkpoints trained at a different N than the current config.
"""

import torch
import torch.nn as nn

import config


class TinyTransformer(nn.Module):
    def __init__(self, N=None):
        super().__init__()
        if N is None:
            N = config.N
        self.N       = N
        self.VOCAB   = N
        self.SEQ_LEN = 4 * N

        self.tok = nn.Embedding(self.VOCAB,   config.D_MODEL)
        self.pos = nn.Embedding(self.SEQ_LEN, config.D_MODEL)

        layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=4 * config.D_MODEL,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.tr   = nn.TransformerEncoder(layer, config.N_LAYERS)
        self.head = nn.Linear(config.D_MODEL, self.VOCAB)

    def forward(self, x):
        """
        x: LongTensor of shape (B, T) with token ids
        returns: FloatTensor of shape (B, T, VOCAB) with logits
        """
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand_as(x)
        h   = self.tok(x) + self.pos(pos)

        # Causal mask: -inf above the diagonal so position k can only see <= k
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1,
        )
        h = self.tr(h, mask=mask)
        return self.head(h)
