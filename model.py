# model.py
"""
TinyTransformer: small decoder-only transformer

Trained as a next-token language model over the sequence
  [BOS] phi [SEP] psi [SEP] phi_inv
with a causal mask so position k only attends to positions < k.
Loss is applied only to the psi and phi_inv blocks (see train.py).
"""

import torch
import torch.nn as nn

import config


class TinyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(config.VOCAB,   config.D_MODEL)
        self.pos = nn.Embedding(config.SEQ_LEN, config.D_MODEL)

        layer = nn.TransformerEncoderLayer(
            d_model=config.D_MODEL,
            nhead=config.N_HEADS,
            dim_feedforward=4 * config.D_MODEL,
            batch_first=True,
            activation='gelu',
            norm_first=True,
        )
        self.tr   = nn.TransformerEncoder(layer, config.N_LAYERS)
        self.head = nn.Linear(config.D_MODEL, config.VOCAB)

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