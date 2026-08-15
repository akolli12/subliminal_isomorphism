"""Run the witness-extraction attack and score top-1 / top-n recovery.

Given a prover and its estimated tau tables, for each test instance we:
  1. draw K2 valid psi samples conditioned on the instance context,
  2. form the test marginal M[i,v],
  3. for each extractor build the cost matrix and rank assignments,
  4. record whether the true witness is the top-1 / within the top-n.

top-1 uses Hungarian; top-n uses Murty's algorithm (subliminal.assignment),
which scales to n = 9 where enumerating n! assignments is infeasible.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

from .assignment import best_assignment, top_k_assignments
from .layout import Layout
from .sample import sample_psi
from .seeding import set_seed
from .tau import ExtractorBank, EXTRACTORS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def test_marginal(model, context: torch.Tensor, layout: Layout, *, k2: int,
                  chunk: int, zero_blocks=()) -> torch.Tensor:
    """(n,n) marginal M[i,v] over k2 valid psi samples for ONE test context."""
    n = layout.n
    counts = torch.zeros((n, n), device=DEVICE)
    for lo in range(0, k2, chunk):
        s = min(chunk, k2 - lo)
        rep = context.expand(s, -1).contiguous()
        psis = sample_psi(model, rep, layout, valid=True, zero_blocks=zero_blocks)
        counts += F.one_hot(psis.long(), n).float().sum(dim=0)
    return counts / k2


def recover_scores(bank: ExtractorBank, test_marg: torch.Tensor,
                   true_witness: tuple, n: int) -> dict:
    """Per-extractor {'top1': bool, 'topn': bool} for one test instance."""
    out = {}
    for name in EXTRACTORS:
        C = bank.cost(name, test_marg)
        best, _ = best_assignment(C)
        topn = top_k_assignments(C, n)
        topn_set = {a for a, _ in topn}
        out[name] = {
            "top1": best == true_witness,
            "topn": true_witness in topn_set,
        }
    return out


def run_extraction(model, layout: Layout, bank: ExtractorBank, *,
                   test_contexts, true_witnesses, k2: int, chunk: int,
                   seed: int, zero_blocks=(), progress_every: int = 25) -> dict:
    """Aggregate recovery over a list of (context, true_witness) test instances.

    Returns a results dict with per-extractor top-1/top-n counts, totals, and
    the random baselines 100/(n-1)! (top-1) and 100n/n! (top-n).
    """
    set_seed(seed)
    n = layout.n
    T = len(test_contexts)
    top1 = {m: 0 for m in EXTRACTORS}
    topn = {m: 0 for m in EXTRACTORS}
    union_top1 = union_topn = 0

    for t, (ctx, truth) in enumerate(zip(test_contexts, true_witnesses)):
        ctx = ctx.to(DEVICE).unsqueeze(0) if ctx.dim() == 1 else ctx.to(DEVICE)
        M = test_marginal(model, ctx, layout, k2=k2, chunk=chunk,
                          zero_blocks=zero_blocks)
        scores = recover_scores(bank, M.cpu(), tuple(truth), n)
        any1 = anyn = False
        for m in EXTRACTORS:
            top1[m] += scores[m]["top1"]
            topn[m] += scores[m]["topn"]
            any1 |= scores[m]["top1"]
            anyn |= scores[m]["topn"]
        union_top1 += any1
        union_topn += anyn
        if (t + 1) % progress_every == 0 or t == T - 1:
            print(f"    extract {t+1}/{T}", flush=True)

    fact = math.factorial
    return {
        "n": n,
        "num_test": T,
        "k2": k2,
        "random_top1_pct": 100.0 / fact(n),
        "random_topn_pct": 100.0 * n / fact(n),
        "extractors": {
            m: {
                "top1": top1[m], "topn": topn[m], "total": T,
                "top1_pct": 100.0 * top1[m] / T,
                "topn_pct": 100.0 * topn[m] / T,
            } for m in EXTRACTORS
        },
        "union": {
            "top1": union_top1, "topn": union_topn, "total": T,
            "top1_pct": 100.0 * union_top1 / T,
            "topn_pct": 100.0 * union_topn / T,
        },
    }
