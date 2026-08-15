"""Per-instance graph-conditioned attack (fix-G0 / fix-G1).

For the non-abstracted GMW-GI prover [G0 | G1 | phi | psi | ...], the marginal
P[psi | .] depends on the graphs, so tau must be built PER TEST INSTANCE,
anchored to that instance's graph:

  fix-G0 : hold this instance's G0 fixed; sample phi' (with phi'(j)=u), let
           G1' = phi'(G0). Context per phi' = [BOS|G0|SEP|G1'|SEP|phi'|SEP].
  fix-G1 : hold this instance's G1 fixed; sample phi', let G0' = phi'^{-1}(G1).
           Context per phi' = [BOS|G0'|SEP|G1|SEP|phi'|SEP].

We reuse subliminal.tau.estimate_tau with a per-instance context_fn, then build
the instance's own test marginal (context [G0|G1|phi_true]) and score recovery
with every extractor. This is the expensive, faithful attack the paper's §5.1
abstracts away — so we use fewer test instances than the perm-only case.
"""

import math

import torch
import torch.nn.functional as F

from subliminal.data import (rand_perms, rand_graphs, apply_perm_to_graph,
                             rand_perms_with_constraint)
from subliminal.extract import recover_scores
from subliminal.multi import specials
from subliminal.multi_graph import graph_multi_layout
from subliminal.sample import sample_psi
from subliminal.tau import estimate_tau, ExtractorBank, EXTRACTORS
from subliminal.seeding import set_seed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _prefix_batch(g0, g1, phis, max_n):
    """[BOS|g0|SEP|g1|SEP|phis|SEP]; g0/g1 each (m,) fixed or (B,m) per-row."""
    sp = specials(max_n)
    B = phis.shape[0]
    bos = torch.full((B, 1), sp["BOS"], dtype=torch.long)
    sep = torch.full((B, 1), sp["SEP"], dtype=torch.long)
    g0b = g0 if g0.dim() == 2 else g0.unsqueeze(0).expand(B, -1)
    g1b = g1 if g1.dim() == 2 else g1.unsqueeze(0).expand(B, -1)
    return torch.cat([bos, g0b, sep, g1b, sep, phis, sep], dim=1)


def _context_fn(g0_star, g1_star, n, max_n, fix):
    """Return context_fn(phis) anchored to a fixed graph of THIS instance."""
    def fn(phis):
        B = phis.shape[0]
        if fix == "G0":                                   # G0 fixed, G1 = phi(G0)
            g0 = g0_star.unsqueeze(0).expand(B, -1)
            g1 = apply_perm_to_graph(g0, phis, n)
        else:                                             # G1 fixed, G0 = phi^{-1}(G1)
            g1 = g1_star.unsqueeze(0).expand(B, -1)
            g0 = apply_perm_to_graph(g1, torch.argsort(phis, dim=1), n)
        return _prefix_batch(g0, g1, phis, max_n)
    return fn


@torch.no_grad()
def _test_marginal(model, g0, g1, phi, layout, k2, max_n, zero_blocks=(),
                   chunk=1 << 14):
    n = layout.n
    counts = torch.zeros((n, n), device=DEVICE)
    for lo in range(0, k2, chunk):
        s = min(chunk, k2 - lo)
        phis = phi.unsqueeze(0).expand(s, -1)
        ctx = _prefix_batch(g0, g1, phis, max_n)
        psis = sample_psi(model, ctx, layout, valid=True, zero_blocks=zero_blocks)
        counts += F.one_hot(psis.long(), n).float().sum(dim=0)
    return counts / k2


def attack_graph_perinstance(model, n, max_n, *, fix, num_instances, k1, k2, seed,
                             mask_phi=False):
    """Per-instance fix-G0 or fix-G1 attack. Returns per-extractor recovery %."""
    set_seed(seed)
    layout = graph_multi_layout(n, max_n)
    g = torch.Generator().manual_seed(seed + 700)
    g0s = rand_graphs(num_instances, n, g)
    phis = rand_perms(num_instances, n, g)

    zb = (layout["phi"],) if mask_phi else ()
    top1 = {m: 0 for m in EXTRACTORS}
    topn = {m: 0 for m in EXTRACTORS}
    union = 0
    for t in range(num_instances):
        g0 = g0s[t]
        phi = phis[t]
        g1 = apply_perm_to_graph(g0.unsqueeze(0), phi.unsqueeze(0), n)[0]
        ctx_fn = _context_fn(g0, g1, n, max_n, fix)
        tr, tl = estimate_tau(model, layout, k1=k1, k2=k2, seed=seed + t,
                              context_fn=ctx_fn, chunk=1 << 14, zero_blocks=zb)
        bank = ExtractorBank(tr, tl)
        M = _test_marginal(model, g0, g1, phi, layout, k2, max_n, zero_blocks=zb)
        sc = recover_scores(bank, M.cpu(), tuple(phi.tolist()), n)
        any_n = False
        for m in EXTRACTORS:
            top1[m] += sc[m]["top1"]
            topn[m] += sc[m]["topn"]
            any_n |= sc[m]["topn"]
        union += any_n
        if (t + 1) % 25 == 0:
            print(f"      graph-{fix} n={n} instance {t+1}/{num_instances}", flush=True)

    T = num_instances
    return {
        "n": n, "fix": fix, "num_test": T,
        "random_topn_pct": 100.0 * n / math.factorial(n),
        "union_topn_pct": 100.0 * union / T,
        "extractors": {m: {"topn_pct": 100.0 * topn[m] / T,
                           "top1_pct": 100.0 * top1[m] / T} for m in EXTRACTORS},
    }


# ---------------------------------------------------------------------------
# Batched version: identical math (same total samples), instances batched per
# (j,u) cell so tau for all instances is built together -> big GPU batches.
# Validated to match attack_graph_perinstance within sampling noise.
# ---------------------------------------------------------------------------
EPS = 1e-12


@torch.no_grad()
def attack_graph_perinstance_batched(model, n, max_n, *, fix, num_instances,
                                     k1, k2, seed, mask_phi=False,
                                     sample_batch=8192):
    # process contexts so that (#contexts * k2) sequences per forward <= sample_batch
    row_chunk = max(1, sample_batch // k2)
    set_seed(seed)
    layout = graph_multi_layout(n, max_n)
    zb = (layout["phi"],) if mask_phi else ()
    g = torch.Generator().manual_seed(seed + 700)
    g0s = rand_graphs(num_instances, n, g)                     # (M, m) CPU
    phis_true = rand_perms(num_instances, n, g)                # (M, n) CPU
    g1s = apply_perm_to_graph(g0s, phis_true, n)               # (M, m) CPU
    M = num_instances

    tau_raw = torch.zeros(M, n, n, n, n)                       # all on CPU;
    tau_log = torch.zeros(M, n, n, n, n)                       # sample_psi -> GPU
    gg = torch.Generator().manual_seed(seed)

    for j in range(n):
        for u in range(n):
            phip = rand_perms_with_constraint(k1, n, j, u, gg)  # (k1,n) CPU
            phip_rep = phip.unsqueeze(0).expand(M, k1, -1).reshape(M * k1, n)
            if fix == "G0":
                g0c = g0s.unsqueeze(1).expand(M, k1, -1).reshape(M * k1, -1)
                g1c = apply_perm_to_graph(g0c, phip_rep, n)
            else:
                g1c = g1s.unsqueeze(1).expand(M, k1, -1).reshape(M * k1, -1)
                g0c = apply_perm_to_graph(g1c, torch.argsort(phip_rep, dim=1), n)
            ctx = _prefix_batch(g0c, g1c, phip_rep, max_n)     # (M*k1, prefix) CPU

            marg = torch.zeros(M * k1, n, n)
            for lo in range(0, M * k1, row_chunk):
                blk = ctx[lo:lo + row_chunk]
                rep = blk.repeat_interleave(k2, dim=0)
                psis = sample_psi(model, rep, layout, valid=True, zero_blocks=zb)
                oh = F.one_hot(psis.cpu().long(), n).float().view(blk.shape[0], k2, n, n)
                marg[lo:lo + blk.shape[0]] = oh.mean(dim=1)
            marg = marg.view(M, k1, n, n)
            tau_raw[:, :, j, :, u] = marg.mean(dim=1)
            tau_log[:, :, j, :, u] = marg.clamp_min(EPS).log().mean(dim=1)

    # test marginals for the true instances
    test = torch.zeros(M, n, n)
    ctx_test = _prefix_batch(g0s, g1s, phis_true, max_n)
    for lo in range(0, M, row_chunk):
        blk = ctx_test[lo:lo + row_chunk]
        rep = blk.repeat_interleave(k2, dim=0)
        psis = sample_psi(model, rep, layout, valid=True, zero_blocks=zb)
        oh = F.one_hot(psis.cpu().long(), n).float().view(blk.shape[0], k2, n, n)
        test[lo:lo + blk.shape[0]] = oh.mean(dim=1)

    top1 = {m: 0 for m in EXTRACTORS}
    topn = {m: 0 for m in EXTRACTORS}
    union = 0
    for i in range(M):
        bank = ExtractorBank(tau_raw[i], tau_log[i])
        sc = recover_scores(bank, test[i], tuple(phis_true[i].tolist()), n)
        anyn = False
        for m in EXTRACTORS:
            top1[m] += sc[m]["top1"]; topn[m] += sc[m]["topn"]; anyn |= sc[m]["topn"]
        union += anyn
    T = M
    return {"n": n, "fix": fix, "num_test": T,
            "random_topn_pct": 100.0 * n / math.factorial(n),
            "union_topn_pct": 100.0 * union / T,
            "extractors": {m: {"topn_pct": 100.0 * topn[m] / T,
                               "top1_pct": 100.0 * top1[m] / T} for m in EXTRACTORS}}
