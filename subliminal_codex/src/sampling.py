"""Autoregressive sampling of valid commitment permutations."""

import torch
import torch.nn.functional as F


@torch.no_grad()
def sample_psi(model, contexts, layout, valid=True, zero_blocks=()):
    model.eval()
    device = next(model.parameters()).device
    contexts = contexts.to(device)
    num_contexts = len(contexts)
    start, n = layout["psi"].start, layout.n
    assert contexts.shape[1] == start
    generated = torch.zeros((num_contexts, n), dtype=torch.long, device=device)
    sequence = torch.cat([contexts, generated], 1)
    for position in range(n):
        logits = model(sequence, zero_blocks=zero_blocks)[
            :, start - 1 + position, :n
        ].clone()
        if valid and position:
            logits.scatter_(1, sequence[:, start:start + position], float("-inf"))
        sequence[:, start + position] = torch.multinomial(
            F.softmax(logits, -1), 1
        ).squeeze(1)
    return sequence[:, layout["psi"]]
