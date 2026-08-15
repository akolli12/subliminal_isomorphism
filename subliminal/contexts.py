"""Context builders: turn a witness phi into the token prefix that precedes
the psi block, for each prover family.

Perm-only prover:  context(phi) = [ phi ]                       width n
Graph prover:      context(phi) = [ G0 | G1=phi(G0) | phi ]     width 2m + n

For tau estimation and test-set construction on the graph prover we must
supply a G0 for every phi. `GraphContext` samples a fresh random G0 per row
(seeded) so tau averages over the instance distribution, matching Eq. 5's
expectation over x induced by phi'. Test instances instead pin an explicit G0
so the true witness is well defined.
"""

import torch

from .data import apply_perm_to_graph, rand_graphs
from .layout import Layout


class PermContext:
    """Identity: the context for sampling psi is phi itself."""

    def __init__(self, layout: Layout):
        self.layout = layout

    def __call__(self, phis: torch.Tensor) -> torch.Tensor:
        return phis


class GraphContext:
    """[G0 | G1 | phi] with G1 = phi(G0). G0 sampled fresh per call unless given."""

    def __init__(self, layout: Layout, seed: int = 0, g0: torch.Tensor = None):
        self.layout = layout
        self.n = layout.n
        self.g = torch.Generator().manual_seed(seed)
        self.g0 = g0                      # if set, reused (fixed test instance)

    def __call__(self, phis: torch.Tensor) -> torch.Tensor:
        b = phis.shape[0]
        if self.g0 is None:
            g0 = rand_graphs(b, self.n, self.g)
        else:
            g0 = self.g0.expand(b, -1) if self.g0.dim() == 2 else \
                 self.g0.unsqueeze(0).expand(b, -1)
        g1 = apply_perm_to_graph(g0, phis, self.n)
        return torch.cat([g0, g1, phis], dim=1)
