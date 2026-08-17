"""Batched faithful graph-conditioned attack for fixed G0 or fixed G1."""

import math

import torch
import torch.nn.functional as F

from data_generation.data import constrained_permutations, graphs, permutations, permute_graph
from src.config import set_seed
from src.extraction import EPS, EXTRACTORS, ExtractorBank, recover
from src.layouts import graph_layout, graph_prefix
from src.sampling import sample_psi


@torch.no_grad()
def graph_attack(model, n, max_n, fixed, num_instances, k1, k2, seed,
                 mask_phi=False, sample_batch=8192):
    """Exact rewrite of attack_graph_perinstance_batched."""
    rows_per_forward = max(1, sample_batch // k2)
    set_seed(seed)
    layout = graph_layout(n, max_n)
    zero_blocks = (layout["phi"],) if mask_phi else ()

    test_generator = torch.Generator().manual_seed(seed + 700)
    g0s = graphs(num_instances, n, test_generator)
    true_phis = permutations(num_instances, n, test_generator)
    g1s = permute_graph(g0s, true_phis, n)

    tau_raw = torch.zeros(num_instances, n, n, n, n)
    tau_log = torch.zeros(num_instances, n, n, n, n)
    phi_generator = torch.Generator().manual_seed(seed)

    for j in range(n):
        for u in range(n):
            candidate_phis = constrained_permutations(k1, n, j, u, phi_generator)
            repeated_phis = candidate_phis[None].expand(
                num_instances, k1, -1
            ).reshape(num_instances * k1, n)
            if fixed == "G0":
                left = g0s[:, None].expand(
                    num_instances, k1, -1
                ).reshape(num_instances * k1, -1)
                right = permute_graph(left, repeated_phis, n)
            else:
                right = g1s[:, None].expand(
                    num_instances, k1, -1
                ).reshape(num_instances * k1, -1)
                left = permute_graph(right, torch.argsort(repeated_phis, 1), n)
            contexts = graph_prefix(left, right, repeated_phis, max_n)
            marginals = torch.zeros(num_instances * k1, n, n)
            for start_index in range(0, num_instances * k1, rows_per_forward):
                context_batch = contexts[start_index:start_index + rows_per_forward]
                sampled_psis = sample_psi(
                    model, context_batch.repeat_interleave(k2, 0), layout,
                    valid=True, zero_blocks=zero_blocks,
                )
                one_hot = F.one_hot(sampled_psis.cpu().long(), n).float().view(
                    len(context_batch), k2, n, n
                )
                marginals[start_index:start_index + len(context_batch)] = one_hot.mean(1)
            marginals = marginals.view(num_instances, k1, n, n)
            tau_raw[:, :, j, :, u] = marginals.mean(1)
            tau_log[:, :, j, :, u] = marginals.clamp_min(EPS).log().mean(1)

    test_marginals = torch.zeros(num_instances, n, n)
    test_contexts = graph_prefix(g0s, g1s, true_phis, max_n)
    for start_index in range(0, num_instances, rows_per_forward):
        context_batch = test_contexts[start_index:start_index + rows_per_forward]
        sampled_psis = sample_psi(
            model, context_batch.repeat_interleave(k2, 0), layout,
            valid=True, zero_blocks=zero_blocks,
        )
        one_hot = F.one_hot(sampled_psis.cpu().long(), n).float().view(
            len(context_batch), k2, n, n
        )
        test_marginals[start_index:start_index + len(context_batch)] = one_hot.mean(1)

    top1 = {name: 0 for name in EXTRACTORS}
    topn = {name: 0 for name in EXTRACTORS}
    union = 0
    for instance_index in range(num_instances):
        scores = recover(
            ExtractorBank(tau_raw[instance_index], tau_log[instance_index]),
            test_marginals[instance_index],
            tuple(true_phis[instance_index].tolist()), n,
        )
        union += any(score["topn"] for score in scores.values())
        for name, score in scores.items():
            top1[name] += score["top1"]
            topn[name] += score["topn"]
    return {
        "n": n,
        "fix": fixed,
        "num_test": num_instances,
        "random_topn_pct": 100 * n / math.factorial(n),
        "union_topn_pct": 100 * union / num_instances,
        "extractors": {
            name: {"topn_pct": 100 * topn[name] / num_instances,
                   "top1_pct": 100 * top1[name] / num_instances}
            for name in EXTRACTORS
        },
    }
