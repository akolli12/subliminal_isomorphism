"""Optimal and top-k assignment on an (n x n) cost matrix.

`best_assignment` is the Hungarian algorithm (top-1 witness).
`top_k_assignments` is Murty's algorithm (paper ref [17]) for the k
lowest-cost assignments, used to evaluate top-n recovery. Murty scales to
n = 9, where brute-forcing all n! assignments (~3.6e5) per test instance is
prohibitive.
"""

import heapq
import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

INF = 1e9


def best_assignment(cost: np.ndarray):
    """Return (assignment as tuple, total cost) via Hungarian."""
    r, c = linear_sum_assignment(cost)
    return tuple(c.tolist()), float(cost[r, c].sum())


def _solve_constrained(cost, includes, excludes):
    """Hungarian on `cost` forcing/forbidding (row -> col) pairs.

    includes: dict row->col that must be used. excludes: set of (row, col)
    forbidden. Returns (assignment tuple, total cost) or (None, inf) if
    infeasible.
    """
    n = cost.shape[0]
    work = cost.copy()
    for (r, c) in excludes:
        work[r, c] = INF
    forced_rows = set(includes)
    forced_cols = set(includes.values())
    free_rows = [r for r in range(n) if r not in forced_rows]
    free_cols = [c for c in range(n) if c not in forced_cols]

    base = sum(cost[r, includes[r]] for r in includes)
    if not free_rows:
        assign = [includes[r] for r in range(n)]
        return tuple(assign), float(base)

    sub = work[np.ix_(free_rows, free_cols)]
    rr, cc = linear_sum_assignment(sub)
    if sub[rr, cc].sum() >= INF:
        return None, float("inf")
    assign = dict(includes)
    for i, j in zip(rr, cc):
        assign[free_rows[i]] = free_cols[j]
    total = base + float(sub[rr, cc].sum())
    return tuple(assign[r] for r in range(n)), total


def top_k_assignments(cost: np.ndarray, k: int):
    """The k lowest-cost assignments as a list of (assignment tuple, cost).

    Murty's partitioning. For n <= 7 with large k this is still cheap; we cap
    node expansion implicitly through the heap.
    """
    n = cost.shape[0]
    a0, c0 = best_assignment(cost)
    # node = (cost, assignment, includes dict, excludes set)
    heap = [(c0, a0, (), ())]
    results = []
    seen = set()
    while heap and len(results) < k:
        c, assign, inc_items, exc_items = heapq.heappop(heap)
        if assign in seen:
            continue
        seen.add(assign)
        results.append((assign, c))

        includes = dict(inc_items)
        excludes = set(exc_items)
        # Partition over rows not yet fixed by `includes`.
        fixed = set(includes)
        free_rows = [r for r in range(n) if r not in fixed]
        cur_inc = dict(includes)
        cur_exc = set(excludes)
        for r in free_rows:
            node_exc = set(cur_exc)
            node_exc.add((r, assign[r]))
            child, cc = _solve_constrained(cost, cur_inc, node_exc)
            if child is not None:
                heapq.heappush(heap, (cc, child,
                                      tuple(sorted(cur_inc.items())),
                                      tuple(sorted(node_exc))))
            cur_inc = dict(cur_inc)
            cur_inc[r] = assign[r]
    return results


def brute_force_assignments(cost: np.ndarray, k=None):
    """Exact ranking by enumerating all n! assignments (small n only)."""
    n = cost.shape[0]
    rows = np.arange(n)
    scored = sorted(
        (float(cost[rows, list(p)].sum()), p) for p in itertools.permutations(range(n)))
    if k is not None:
        scored = scored[:k]
    return [(p, c) for c, p in scored]
