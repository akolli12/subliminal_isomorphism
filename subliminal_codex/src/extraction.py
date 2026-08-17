"""Marginal tables and the six coordinate-wise witness extractors."""

import itertools
import heapq
import math
import random

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from data_generation.data import constrained_permutations
from src.config import set_seed
from src.sampling import sample_psi

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPS = 1e-12
EXTRACTORS = (
    "single-max-spread raw", "single-max-spread log",
    "aggregate-L1 raw", "aggregate-L2 raw",
    "aggregate-L1 log", "aggregate-L2 log",
)
INF = 1e9


def best_assignment(cost):
    rows, cols = linear_sum_assignment(cost)
    return tuple(cols.tolist()), float(cost[rows, cols].sum())


def _constrained_assignment(cost, includes, excludes):
    work = cost.copy()
    for row, col in excludes:
        work[row, col] = INF
    free_rows = [r for r in range(len(cost)) if r not in includes]
    forced_cols = set(includes.values())
    free_cols = [c for c in range(len(cost)) if c not in forced_cols]
    base = sum(cost[r, c] for r, c in includes.items())
    if not free_rows:
        return tuple(includes[r] for r in range(len(cost))), float(base)
    sub = work[np.ix_(free_rows, free_cols)]
    rows, cols = linear_sum_assignment(sub)
    if sub[rows, cols].sum() >= INF:
        return None, float("inf")
    result = dict(includes)
    for row, col in zip(rows, cols):
        result[free_rows[row]] = free_cols[col]
    return tuple(result[r] for r in range(len(cost))), base + float(sub[rows, cols].sum())


def top_k_assignments(cost, k):
    assignment, score = best_assignment(cost)
    heap = [(score, assignment, (), ())]
    results, seen = [], set()
    while heap and len(results) < k:
        score, assignment, inc_items, exc_items = heapq.heappop(heap)
        if assignment in seen:
            continue
        seen.add(assignment)
        results.append((assignment, score))
        includes, excludes = dict(inc_items), set(exc_items)
        current = dict(includes)
        for row in [r for r in range(len(cost)) if r not in includes]:
            child_excludes = set(excludes)
            child_excludes.add((row, assignment[row]))
            child, child_score = _constrained_assignment(cost, current, child_excludes)
            if child is not None:
                heapq.heappush(heap, (child_score, child, tuple(sorted(current.items())),
                                      tuple(sorted(child_excludes))))
            current[row] = assignment[row]
    return results


def estimate_tau(model, layout, k1, k2, seed, make_context,
                 zero_blocks=(), chunk=1 << 15):
    """Return raw/log tau[i,j,v,u], preserving the original RNG order."""
    set_seed(seed)
    n = layout.n
    generator = torch.Generator().manual_seed(seed)
    raw, log = torch.zeros((n, n, n, n)), torch.zeros((n, n, n, n))
    phis_per_batch = max(1, chunk // k2)
    for j in range(n):
        for u in range(n):
            candidate_phis = constrained_permutations(k1, n, j, u, generator)
            marginals = torch.zeros((k1, n, n), device=DEVICE)
            for start_index in range(0, k1, phis_per_batch):
                phi_batch = candidate_phis[start_index:start_index + phis_per_batch]
                repeated_phis = phi_batch.repeat_interleave(k2, 0)
                sampled_psis = sample_psi(
                    model, make_context(repeated_phis), layout,
                    valid=True, zero_blocks=zero_blocks,
                )
                one_hot = F.one_hot(sampled_psis.long(), n).float().view(
                    len(phi_batch), k2, n, n
                )
                marginals[start_index:start_index + len(phi_batch)] = one_hot.mean(1)
            raw[:, j, :, u] = marginals.mean(0).cpu()
            log[:, j, :, u] = marginals.clamp_min(EPS).log().mean(0).cpu()
        print(f"  tau: row j={j} done", flush=True)
    return raw, log


def _witness_coordinates(tau):
    n = len(tau)
    coordinates = {}
    for j in range(n):
        spread = tau[:, j].amax(-1) - tau[:, j].amin(-1)
        flat = int(spread.argmax())
        coordinates[j] = (flat // n, flat % n)
    return coordinates


def _single_cost(tau, coordinates, marginal):
    cost = torch.zeros((len(tau), len(tau)))
    for j, (i, v) in coordinates.items():
        cost[j] = (marginal[i, v] - tau[i, j, v]).abs()
    return cost.numpy()


def _aggregate_cost(tau, marginal, norm_order):
    difference = (marginal[:, None, :, None] - tau).abs()
    return (
        (difference ** norm_order).sum((0, 2)) ** (1 / norm_order)
    ).numpy()


class ExtractorBank:
    def __init__(self, raw, log):
        self.raw, self.log = raw, log
        self.raw_coordinates = _witness_coordinates(raw)
        self.log_coordinates = _witness_coordinates(log)

    def cost(self, name, marginal):
        log_marginal = marginal.clamp_min(EPS).log()
        if name == "single-max-spread raw":
            return _single_cost(self.raw, self.raw_coordinates, marginal)
        if name == "single-max-spread log":
            return _single_cost(self.log, self.log_coordinates, log_marginal)
        table, values = (self.log, log_marginal) if name.endswith("log") else (self.raw, marginal)
        return _aggregate_cost(table, values, 1 if "L1" in name else 2)


def recover(bank, marginal, truth, n):
    scores = {}
    for name in EXTRACTORS:
        cost = bank.cost(name, marginal)
        best, _ = best_assignment(cost)
        top_n = {assignment for assignment, _ in top_k_assignments(cost, n)}
        scores[name] = {"top1": best == truth, "topn": truth in top_n}
    return scores


def _test_marginal(model, context, layout, k2, zero_blocks=(), chunk=1 << 15):
    counts = torch.zeros((layout.n, layout.n), device=DEVICE)
    for start_index in range(0, k2, chunk):
        num_samples = min(chunk, k2 - start_index)
        sampled_psis = sample_psi(
            model, context.expand(num_samples, -1).contiguous(), layout,
            valid=True, zero_blocks=zero_blocks,
        )
        counts += F.one_hot(sampled_psis.long(), layout.n).float().sum(0)
    return counts.cpu() / k2


def permutation_attack(model, layout, make_context, k1, k2, test_cap,
                       tau_seed, eval_seed, mask_phi=False):
    zero_blocks = (layout["phi"],) if mask_phi else ()
    raw, log = estimate_tau(model, layout, k1, k2, tau_seed, make_context, zero_blocks)
    bank = ExtractorBank(raw, log)
    witnesses = list(itertools.permutations(range(layout.n)))
    if len(witnesses) > test_cap:
        random.Random(eval_seed).shuffle(witnesses)
        witnesses = witnesses[:test_cap]
    set_seed(eval_seed)
    top1 = {name: 0 for name in EXTRACTORS}
    topn = {name: 0 for name in EXTRACTORS}
    union = 0
    for index, truth in enumerate(witnesses):
        phi = torch.tensor(truth).unsqueeze(0)
        marginal = _test_marginal(model, make_context(phi), layout, k2, zero_blocks)
        scores = recover(bank, marginal, truth, layout.n)
        union += any(score["topn"] for score in scores.values())
        for name, score in scores.items():
            top1[name] += score["top1"]
            topn[name] += score["topn"]
        if (index + 1) % 25 == 0 or index + 1 == len(witnesses):
            print(f"    extract {index + 1}/{len(witnesses)}", flush=True)
    total = len(witnesses)
    random_topn = 100 * layout.n / math.factorial(layout.n)
    extractor_results = {
        name: {"topn_pct": 100 * topn[name] / total, "top1_pct": 100 * top1[name] / total}
        for name in EXTRACTORS
    }
    return {
        "union_topn_pct": 100 * union / total,
        "best_topn_pct": max(row["topn_pct"] for row in extractor_results.values()),
        "random_topn_pct": random_topn,
        "num_test": total,
        "extractors": extractor_results,
    }
