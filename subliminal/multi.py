"""Single length-generalizing prover for variable n, using delimiter tokens.

One model handles n = 4..MAX_N. Each example is a delimited, variable-length
sequence:

    [ BOS | phi(n) | SEP | psi(n) | SEP | psi^{-1}(n) | SEP | phi o psi^{-1}(n) ]

length 4n+4, right-padded with PAD to the max length 4*MAX_N+4. Permutation
values 0..n-1 share one vocabulary across all n; the model infers n from where
the delimiters fall. Loss (standard next-token CE with ignore_index) is applied
only on the psi, psi^{-1}, and phi o psi^{-1} blocks.

Vocabulary (MAX_N value tokens + 3 specials):
    0 .. MAX_N-1  permutation values
    SEP  = MAX_N
    BOS  = MAX_N + 1
    PAD  = MAX_N + 2
"""

from dataclasses import dataclass

import torch

IGNORE = -100


def specials(max_n):
    return {"SEP": max_n, "BOS": max_n + 1, "PAD": max_n + 2, "vocab": max_n + 3}


def multi_seq_len(max_n):
    return 4 * max_n + 4


@dataclass(frozen=True)
class MultiLayout:
    n: int
    vocab: int
    seq_len: int          # full padded length (4*MAX_N+4)
    blocks: dict

    def __getitem__(self, name):
        return self.blocks[name]


def multi_layout(n, max_n):
    return MultiLayout(
        n=n, vocab=specials(max_n)["vocab"], seq_len=multi_seq_len(max_n),
        blocks={
            "phi":         slice(1,          1 + n),
            "psi":         slice(2 + n,      2 + 2 * n),
            "psi_inv":     slice(3 + 2 * n,  3 + 3 * n),
            "phi_psi_inv": slice(4 + 3 * n,  4 + 4 * n),
        },
    )


def build_multi_example(n, phi_row, psi_row, max_n):
    """Return (tokens, labels), each length 4*MAX_N+4 (right-padded).

    labels[k] = tokens[k+1] when position k predicts a loss-block token, else
    IGNORE — ready for CE(logits, labels, ignore_index=IGNORE).
    """
    sp = specials(max_n)
    SEP, BOS, PAD = sp["SEP"], sp["BOS"], sp["PAD"]
    psi_inv = torch.argsort(psi_row)
    phi_psi_inv = phi_row[psi_inv]
    one = lambda t: torch.tensor([t])
    toks = torch.cat([one(BOS), phi_row, one(SEP), psi_row, one(SEP),
                      psi_inv, one(SEP), phi_psi_inv])
    T = toks.shape[0]

    lay = multi_layout(n, max_n)
    is_loss = torch.zeros(T, dtype=torch.bool)
    for b in ("psi", "psi_inv", "phi_psi_inv"):
        is_loss[lay[b]] = True

    labels = torch.full((T,), IGNORE, dtype=torch.long)
    labels[:-1] = torch.where(is_loss[1:], toks[1:], torch.full_like(toks[1:], IGNORE))

    full = multi_seq_len(max_n)
    if T < full:
        toks = torch.cat([toks, torch.full((full - T,), PAD, dtype=torch.long)])
        labels = torch.cat([labels, torch.full((full - T,), IGNORE, dtype=torch.long)])
    return toks, labels


def build_multi_batch(ns, phis, psis, max_n):
    """Stack a batch of (n, phi, psi). ns: list; phis/psis: list of (n,) tensors."""
    toks, labs = [], []
    for n, p, q in zip(ns, phis, psis):
        t, l = build_multi_example(n, p, q, max_n)
        toks.append(t)
        labs.append(l)
    return torch.stack(toks), torch.stack(labs)


def build_multi_example_soft(n, phi_row, psi_row, max_n, psi_mode="ce"):
    """Like build_multi_example but returns SOFT targets (T, vocab) + loss mask.

    Unifies the standard and simulator-aligned objectives via soft cross-entropy:
      psi_mode='ce'      -> one-hot target on the true psi token (standard CE),
      psi_mode='uniform' -> uniform-over-remaining target (defense, §4.1).
    psi^{-1} and phi o psi^{-1} are always one-hot CE.
    """
    sp = specials(max_n)
    SEP, BOS, PAD, V = sp["SEP"], sp["BOS"], sp["PAD"], sp["vocab"]
    psi_inv = torch.argsort(psi_row)
    phi_psi_inv = phi_row[psi_inv]
    one = lambda t: torch.tensor([t])
    toks = torch.cat([one(BOS), phi_row, one(SEP), psi_row, one(SEP),
                      psi_inv, one(SEP), phi_psi_inv])
    T = multi_seq_len(max_n)
    target = torch.zeros(T, V)
    mask = torch.zeros(T, dtype=torch.bool)
    lay = multi_layout(n, max_n)

    def one_hot_block(blk):
        for k in range(blk.start, blk.stop):        # predicted at position k-1
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


def build_multi_batch_soft(ns, phis, psis, max_n, psi_mode="ce"):
    toks, tgts, masks = [], [], []
    for n, p, q in zip(ns, phis, psis):
        t, tg, m = build_multi_example_soft(n, p, q, max_n, psi_mode)
        toks.append(t); tgts.append(tg); masks.append(m)
    return torch.stack(toks), torch.stack(tgts), torch.stack(masks)


class MultiContext:
    """Wrap phi (B, n) -> [BOS, phi, SEP] (B, n+2), the prefix before the psi block."""

    def __init__(self, n, max_n):
        self.n, self.max_n = n, max_n

    def __call__(self, phis):
        sp = specials(self.max_n)
        B = phis.shape[0]
        bos = torch.full((B, 1), sp["BOS"], dtype=torch.long)
        sep = torch.full((B, 1), sp["SEP"], dtype=torch.long)
        return torch.cat([bos, phis, sep], dim=1)
