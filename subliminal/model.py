"""TinyTransformer: small decoder-only transformer (paper Section D.1).

d_model=256, 8 layers, 4 heads, feedforward 4*d_model, pre-LayerNorm, GELU,
causal self-attention, learned token + positional embeddings. The vocabulary
and sequence length come from the layout; there are no separator tokens.

`forward(x, zero_blocks=...)` supports the witness-masking defense (paper
Section 4.2): token embeddings at the given blocks are zeroed (positional
embeddings retained), so the model's output is constant in the content of
those blocks. This implements P[psi | x, phi] = P[psi] exactly, avoiding the
empty-attention-row pathology a pure attention mask would create at the first
psi-generating position.
"""

import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab: int, seq_len: int,
                 d_model: int = 256, n_heads: int = 4, n_layers: int = 8):
        super().__init__()
        self.vocab, self.seq_len = vocab, seq_len
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, x: torch.Tensor, zero_blocks=()) -> torch.Tensor:
        """x: (B, T) token ids, T <= seq_len. Returns (B, T, vocab) logits."""
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        tok = self.tok(x)
        for blk in zero_blocks:
            stop = min(blk.stop, T)
            if blk.start < T:
                tok = tok.clone()
                tok[:, blk.start:stop, :] = 0.0
        h = tok + self.pos(pos)
        mask = torch.triu(
            torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        return self.head(self.tr(h, mask=mask))
