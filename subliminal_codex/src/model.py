"""The exact TinyTransformer architecture used by the current experiment."""

import torch
from torch import nn


class TinyTransformer(nn.Module):
    def __init__(self, vocab, seq_len, d_model=256, n_heads=4, n_layers=8):
        super().__init__()
        self.vocab, self.seq_len = vocab, seq_len
        # These short attribute names are retained for checkpoint compatibility.
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.tr = nn.TransformerEncoder(layer, n_layers)
        self.head = nn.Linear(d_model, vocab)

    def forward(self, tokens, zero_blocks=()):
        batch_size, sequence_length = tokens.shape
        positions = torch.arange(
            sequence_length, device=tokens.device
        ).expand(batch_size, sequence_length)
        embeddings = self.tok(tokens)
        for block in zero_blocks:
            stop = min(block.stop, sequence_length)
            if block.start < sequence_length:
                embeddings = embeddings.clone()
                embeddings[:, block.start:stop] = 0.0
        causal_mask = torch.triu(
            torch.full(
                (sequence_length, sequence_length),
                float("-inf"),
                device=tokens.device,
            ),
            diagonal=1,
        )
        return self.head(self.tr(embeddings + self.pos(positions), mask=causal_mask))
