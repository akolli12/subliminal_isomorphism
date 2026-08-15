"""Graph-conditioned length-generalizing prover: the full GMW-GI input with
delimiter tokens, one model for n=4..MAX_N.

Sequence for size n (m = n(n-1)/2 upper-triangular adjacency bits):

  [ BOS | g0(m) | SEP | g1(m) | SEP | phi(n) | SEP | psi(n) | SEP
        | psi^{-1}(n) | SEP | phi o psi^{-1}(n) ]

length 6 + 2m + 4n, right-padded to the n=MAX_N maximum. G1 = phi(G0). Graph
bits {0,1} share tokens 0/1 with the permutation values; delimiters + position
disambiguate. Loss (CE) on psi, psi^{-1}, phi o psi^{-1}; phi and the graphs are
input-only (phi given — the graph-given prover of Table 3).
"""

from dataclasses import dataclass

import torch

from .data import rand_graphs, apply_perm_to_graph
from .multi import specials, IGNORE


def m_bits(n):
    return n * (n - 1) // 2


def graph_seq_len(max_n):
    return 6 + 2 * m_bits(max_n) + 4 * max_n


@dataclass(frozen=True)
class GraphMultiLayout:
    n: int
    vocab: int
    seq_len: int
    blocks: dict

    def __getitem__(self, name):
        return self.blocks[name]


def graph_multi_layout(n, max_n):
    m = m_bits(n)
    b = {
        "g0":          slice(1,              1 + m),
        "g1":          slice(2 + m,          2 + 2 * m),
        "phi":         slice(3 + 2 * m,      3 + 2 * m + n),
        "psi":         slice(4 + 2 * m + n,  4 + 2 * m + 2 * n),
        "psi_inv":     slice(5 + 2 * m + 2 * n, 5 + 2 * m + 3 * n),
        "phi_psi_inv": slice(6 + 2 * m + 3 * n, 6 + 2 * m + 4 * n),
    }
    return GraphMultiLayout(n=n, vocab=specials(max_n)["vocab"],
                            seq_len=graph_seq_len(max_n), blocks=b)


def build_graph_multi_example(n, g0_bits, phi_row, psi_row, max_n):
    sp = specials(max_n)
    SEP, BOS, PAD = sp["SEP"], sp["BOS"], sp["PAD"]
    g1_bits = apply_perm_to_graph(g0_bits.unsqueeze(0), phi_row.unsqueeze(0), n)[0]
    psi_inv = torch.argsort(psi_row)
    phi_psi_inv = phi_row[psi_inv]
    one = lambda t: torch.tensor([t])
    toks = torch.cat([one(BOS), g0_bits, one(SEP), g1_bits, one(SEP),
                      phi_row, one(SEP), psi_row, one(SEP),
                      psi_inv, one(SEP), phi_psi_inv])
    T = toks.shape[0]

    lay = graph_multi_layout(n, max_n)
    is_loss = torch.zeros(T, dtype=torch.bool)
    for blk in ("psi", "psi_inv", "phi_psi_inv"):
        is_loss[lay[blk]] = True
    labels = torch.full((T,), IGNORE, dtype=torch.long)
    labels[:-1] = torch.where(is_loss[1:], toks[1:],
                              torch.full_like(toks[1:], IGNORE))

    full = graph_seq_len(max_n)
    if T < full:
        toks = torch.cat([toks, torch.full((full - T,), PAD, dtype=torch.long)])
        labels = torch.cat([labels, torch.full((full - T,), IGNORE, dtype=torch.long)])
    return toks, labels


def build_graph_multi_batch(ns, g0s, phis, psis, max_n):
    toks, labs = [], []
    for n, g0, p, q in zip(ns, g0s, phis, psis):
        t, l = build_graph_multi_example(n, g0, p, q, max_n)
        toks.append(t); labs.append(l)
    return torch.stack(toks), torch.stack(labs)


def build_graph_multi_example_soft(n, g0_bits, phi_row, psi_row, max_n, psi_mode="ce"):
    """Soft-target version (for the simulator-aligned defense on the graph prover):
    psi_mode='uniform' trains psi to uniform-over-remaining; psi_inv and
    phi o psi_inv stay one-hot CE. Returns (toks, target (T,V), mask (T,))."""
    sp = specials(max_n)
    SEP, BOS, PAD, V = sp["SEP"], sp["BOS"], sp["PAD"], sp["vocab"]
    g1_bits = apply_perm_to_graph(g0_bits.unsqueeze(0), phi_row.unsqueeze(0), n)[0]
    psi_inv = torch.argsort(psi_row)
    phi_psi_inv = phi_row[psi_inv]
    one = lambda t: torch.tensor([t])
    toks = torch.cat([one(BOS), g0_bits, one(SEP), g1_bits, one(SEP),
                      phi_row, one(SEP), psi_row, one(SEP),
                      psi_inv, one(SEP), phi_psi_inv])
    T = graph_seq_len(max_n)
    target = torch.zeros(T, V)
    mask = torch.zeros(T, dtype=torch.bool)
    lay = graph_multi_layout(n, max_n)

    def one_hot_block(blk):
        for k in range(blk.start, blk.stop):
            target[k - 1, toks[k]] = 1.0
            mask[k - 1] = True

    one_hot_block(lay["psi_inv"])
    one_hot_block(lay["phi_psi_inv"])
    psi_blk = lay["psi"]
    if psi_mode == "uniform":
        used = torch.zeros(V, dtype=torch.bool)
        for k in range(psi_blk.start, psi_blk.stop):
            remaining = [v for v in range(n) if not used[v]]
            for v in remaining:
                target[k - 1, v] = 1.0 / len(remaining)
            mask[k - 1] = True
            used[toks[k]] = True
    else:
        one_hot_block(psi_blk)

    if toks.shape[0] < T:
        toks = torch.cat([toks, torch.full((T - toks.shape[0],), PAD, dtype=torch.long)])
    return toks, target, mask


def build_graph_multi_batch_soft(ns, g0s, phis, psis, max_n, psi_mode="ce"):
    toks, tgts, masks = [], [], []
    for n, g0, p, q in zip(ns, g0s, phis, psis):
        t, tg, m = build_graph_multi_example_soft(n, g0, p, q, max_n, psi_mode)
        toks.append(t); tgts.append(tg); masks.append(m)
    return torch.stack(toks), torch.stack(tgts), torch.stack(masks)


class GraphMultiContext:
    """phi (B,n) -> [BOS | g0 | SEP | g1 | SEP | phi | SEP], with G1=phi(G0).

    G0 is sampled fresh per call (seeded) unless a fixed g0 is provided (for a
    pinned test instance).
    """

    def __init__(self, n, max_n, seed=0, g0=None):
        self.n, self.max_n = n, max_n
        self.g = torch.Generator().manual_seed(seed)
        self.g0 = g0

    def __call__(self, phis):
        sp = specials(self.max_n)
        B = phis.shape[0]
        if self.g0 is None:
            g0 = rand_graphs(B, self.n, self.g)
        else:
            g0 = self.g0.unsqueeze(0).expand(B, -1) if self.g0.dim() == 1 else \
                 self.g0.expand(B, -1)
        g1 = apply_perm_to_graph(g0, phis, self.n)
        bos = torch.full((B, 1), sp["BOS"], dtype=torch.long)
        sep = torch.full((B, 1), sp["SEP"], dtype=torch.long)
        return torch.cat([bos, g0, sep, g1, sep, phis, sep], dim=1)
