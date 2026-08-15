"""Token-sequence layouts for the two prover families.

Permutation-only prover (paper Section D.2/D.3 — graphs omitted):

    [ phi | psi | psi_inv | phi_psi_inv ]           4n tokens, vocab n

Graph-conditioned prover (full GMW-GI input, Section D.3 / Table 3 and the
non-abstracted experiment):

    [ g0 | g1 | phi | psi | psi_inv | phi_psi_inv ]  2m + 4n tokens, m = n(n-1)/2

Graphs are encoded as upper-triangular adjacency bits in row-major pair order
(0,1), (0,2), ..., (n-2,n-1). Bit tokens {0,1} share the vocabulary with
permutation values {0..n-1}; positional embeddings disambiguate blocks.

Next-token convention everywhere: logits at position k predict the token at
input position k+1, so the block [a, b) is predicted by logits [a-1, b-1).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Layout:
    n: int
    vocab: int
    seq_len: int
    blocks: dict = field(default_factory=dict)  # name -> slice

    def __getitem__(self, name: str) -> slice:
        return self.blocks[name]


def perm_layout(n: int) -> Layout:
    return Layout(
        n=n, vocab=n, seq_len=4 * n,
        blocks={
            "phi":         slice(0 * n, 1 * n),
            "psi":         slice(1 * n, 2 * n),
            "psi_inv":     slice(2 * n, 3 * n),
            "phi_psi_inv": slice(3 * n, 4 * n),
        },
    )


def graph_layout(n: int) -> Layout:
    m = n * (n - 1) // 2
    return Layout(
        n=n, vocab=max(n, 2), seq_len=2 * m + 4 * n,
        blocks={
            "g0":          slice(0, m),
            "g1":          slice(m, 2 * m),
            "phi":         slice(2 * m + 0 * n, 2 * m + 1 * n),
            "psi":         slice(2 * m + 1 * n, 2 * m + 2 * n),
            "psi_inv":     slice(2 * m + 2 * n, 2 * m + 3 * n),
            "phi_psi_inv": slice(2 * m + 3 * n, 2 * m + 4 * n),
        },
    )
