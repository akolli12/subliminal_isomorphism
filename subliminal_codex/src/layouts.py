"""Delimited variable-n layouts and exact hard/soft training targets."""

from dataclasses import dataclass

import torch

from data_generation.data import permute_graph

IGNORE = -100


def specials(max_n):
    return {"SEP": max_n, "BOS": max_n + 1, "PAD": max_n + 2, "vocab": max_n + 3}


def perm_seq_len(max_n):
    return 4 * max_n + 4


def graph_seq_len(max_n):
    return 6 + max_n * (max_n - 1) + 4 * max_n


@dataclass(frozen=True)
class Layout:
    n: int
    vocab: int
    seq_len: int
    blocks: dict

    def __getitem__(self, name):
        return self.blocks[name]


def perm_layout(n, max_n):
    return Layout(n, specials(max_n)["vocab"], perm_seq_len(max_n), {
        "phi": slice(1, 1 + n),
        "psi": slice(2 + n, 2 + 2 * n),
        "psi_inv": slice(3 + 2 * n, 3 + 3 * n),
        "phi_psi_inv": slice(4 + 3 * n, 4 + 4 * n),
    })


def graph_layout(n, max_n):
    edges = n * (n - 1) // 2
    return Layout(n, specials(max_n)["vocab"], graph_seq_len(max_n), {
        "g0": slice(1, 1 + edges),
        "g1": slice(2 + edges, 2 + 2 * edges),
        "phi": slice(3 + 2 * edges, 3 + 2 * edges + n),
        "psi": slice(4 + 2 * edges + n, 4 + 2 * edges + 2 * n),
        "psi_inv": slice(5 + 2 * edges + 2 * n, 5 + 2 * edges + 3 * n),
        "phi_psi_inv": slice(6 + 2 * edges + 3 * n, 6 + 2 * edges + 4 * n),
    })


def _build_example(n, phi, psi, max_n, g0=None, soft=False, uniform_psi=False):
    """Build one padded sequence and either hard or soft next-token targets."""
    token_ids = specials(max_n)
    one = lambda value: torch.tensor([value])
    psi_inv = torch.argsort(psi)
    response = phi[psi_inv]
    if g0 is None:
        tokens = torch.cat([one(token_ids["BOS"]), phi, one(token_ids["SEP"]), psi,
                            one(token_ids["SEP"]), psi_inv,
                            one(token_ids["SEP"]), response])
        layout = perm_layout(n, max_n)
    else:
        g1 = permute_graph(g0[None], phi[None], n)[0]
        tokens = torch.cat([one(token_ids["BOS"]), g0, one(token_ids["SEP"]), g1,
                            one(token_ids["SEP"]), phi, one(token_ids["SEP"]), psi,
                            one(token_ids["SEP"]), psi_inv,
                            one(token_ids["SEP"]), response])
        layout = graph_layout(n, max_n)

    if soft:
        targets = torch.zeros(layout.seq_len, layout.vocab)
        mask = torch.zeros(layout.seq_len, dtype=torch.bool)

        def one_hot(block):
            for k in range(block.start, block.stop):
                targets[k - 1, tokens[k]] = 1.0
                mask[k - 1] = True

        one_hot(layout["psi_inv"])
        one_hot(layout["phi_psi_inv"])
        if uniform_psi:
            used = torch.zeros(layout.vocab, dtype=torch.bool)
            for k in range(layout["psi"].start, layout["psi"].stop):
                remaining = [v for v in range(n) if not used[v]]
                targets[k - 1, remaining] = 1.0 / len(remaining)
                mask[k - 1] = True
                used[tokens[k]] = True
        else:
            one_hot(layout["psi"])
        if len(tokens) < layout.seq_len:
            tokens = torch.cat([
                tokens, torch.full((layout.seq_len - len(tokens),), token_ids["PAD"])
            ])
        return tokens, targets, mask

    loss_tokens = torch.zeros(len(tokens), dtype=torch.bool)
    for name in ("psi", "psi_inv", "phi_psi_inv"):
        loss_tokens[layout[name]] = True
    labels = torch.full((len(tokens),), IGNORE, dtype=torch.long)
    labels[:-1] = torch.where(loss_tokens[1:], tokens[1:], torch.full_like(tokens[1:], IGNORE))
    if len(tokens) < layout.seq_len:
        padding = layout.seq_len - len(tokens)
        tokens = torch.cat([tokens, torch.full((padding,), token_ids["PAD"])])
        labels = torch.cat([labels, torch.full((padding,), IGNORE)])
    return tokens, labels


def build_batch(ns, phis, psis, max_n, g0s=None, soft=False, uniform_psi=False):
    """Build a mixed-n permutation or graph batch with identical padding."""
    graph_rows = g0s if g0s is not None else [None] * len(ns)
    rows = [
        _build_example(n, phi, psi, max_n, g0,
                       soft=soft, uniform_psi=uniform_psi)
        for n, phi, psi, g0 in zip(ns, phis, psis, graph_rows)
    ]
    return tuple(torch.stack(parts) for parts in zip(*rows))


class PermutationContext:
    """Convert phi rows to the prefix [BOS | phi | SEP]."""

    def __init__(self, max_n):
        self.max_n = max_n

    def __call__(self, phis):
        token_ids = specials(self.max_n)
        num_contexts = len(phis)
        bos = torch.full((num_contexts, 1), token_ids["BOS"], dtype=torch.long)
        sep = torch.full((num_contexts, 1), token_ids["SEP"], dtype=torch.long)
        return torch.cat([bos, phis, sep], 1)


def graph_prefix(g0, g1, phis, max_n):
    """Build [BOS | G0 | SEP | G1 | SEP | phi | SEP]."""
    token_ids = specials(max_n)
    num_contexts = len(phis)
    bos = torch.full((num_contexts, 1), token_ids["BOS"], dtype=torch.long)
    sep = torch.full((num_contexts, 1), token_ids["SEP"], dtype=torch.long)
    g0 = g0 if g0.dim() == 2 else g0[None].expand(num_contexts, -1)
    g1 = g1 if g1.dim() == 2 else g1[None].expand(num_contexts, -1)
    return torch.cat([bos, g0, sep, g1, sep, phis, sep], 1)
