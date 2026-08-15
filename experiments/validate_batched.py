"""Validate the batched graph attack: (1) deterministic-identical context
construction (bit-exact), (2) recovery agrees with the slow version within
Monte-Carlo noise at matched instance count, (3) it is actually faster, and
(4) timing at n=8,9 to see how many instances are feasible.
"""
import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from subliminal.model import TinyTransformer
from subliminal.multi import specials
from subliminal.multi_graph import graph_seq_len, graph_multi_layout
from subliminal.data import rand_graphs, rand_perms, rand_perms_with_constraint, apply_perm_to_graph
from graph_perinstance import (attack_graph_perinstance, attack_graph_perinstance_batched,
                               _context_fn, _prefix_batch, EXTRACTORS)

MAX_N = 9; dev = 'cuda'
m = TinyTransformer(specials(MAX_N)['vocab'], graph_seq_len(MAX_N), 256, 4, 8).to(dev)
m.load_state_dict(torch.load('checkpoints/mt_graph_base_seed0_exponential_n9.pt', map_location=dev))
m.eval()


def det_context_check(n, fix):
    """Bit-exact: batched builds identical context tokens to slow for same phi'."""
    g = torch.Generator().manual_seed(700)
    g0 = rand_graphs(3, n, g); phi = rand_perms(3, n, g)
    g1 = apply_perm_to_graph(g0, phi, n)
    gg = torch.Generator().manual_seed(0)
    phip = rand_perms_with_constraint(8, n, 1, 2, gg)      # a (j,u) cell
    ok = True
    for i in range(3):
        slow_ctx = _context_fn(g0[i], g1[i], n, MAX_N, fix)(phip)   # slow path
        # batched path for this instance/cell
        if fix == 'G0':
            g0c = g0[i].unsqueeze(0).expand(8, -1); g1c = apply_perm_to_graph(g0c, phip, n)
        else:
            g1c = g1[i].unsqueeze(0).expand(8, -1); g0c = apply_perm_to_graph(g1c, torch.argsort(phip, 1), n)
        bat_ctx = _prefix_batch(g0c, g1c, phip, MAX_N)
        ok &= torch.equal(slow_ctx, bat_ctx)
    return ok


print("=== (1) DETERMINISTIC context construction (must be bit-exact) ===", flush=True)
for n in [5, 8]:
    for fix in ['G0', 'G1']:
        print(f"  n={n} fix-{fix}: contexts identical = {det_context_check(n, fix)}", flush=True)

print("\n=== (2)+(3) recovery agreement + speedup at matched N (n=6, N=80) ===", flush=True)
for fix in ['G0', 'G1']:
    t0 = time.time(); rs = attack_graph_perinstance(m, 6, MAX_N, fix=fix, num_instances=80, k1=48, k2=48, seed=0); ts = time.time() - t0
    t0 = time.time(); rb = attack_graph_perinstance_batched(m, 6, MAX_N, fix=fix, num_instances=80, k1=48, k2=48, seed=0); tb = time.time() - t0
    diffs = [abs(rs['extractors'][x]['topn_pct'] - rb['extractors'][x]['topn_pct']) for x in EXTRACTORS]
    print(f"  fix-{fix}: slow {ts:.0f}s vs batched {tb:.0f}s  -> {ts/tb:.1f}x faster", flush=True)
    print(f"           per-extractor recovery (slow / batched):", flush=True)
    for x in EXTRACTORS:
        print(f"             {x:24} {rs['extractors'][x]['topn_pct']:5.1f} / {rb['extractors'][x]['topn_pct']:5.1f}", flush=True)
    print(f"           max diff = {max(diffs):.1f}pp (noise at N=80 ~= {100/80**0.5*0.5:.1f}pp)", flush=True)

print("\n=== (4) feasibility: batched timing at n=8,9 vs instance count ===", flush=True)
for n in [8, 9]:
    for N in [120, 300]:
        t0 = time.time(); r = attack_graph_perinstance_batched(m, n, MAX_N, fix='G0', num_instances=N, k1=48, k2=48, seed=0); dt = time.time() - t0
        best = max(r['extractors'][x]['topn_pct'] for x in EXTRACTORS)
        print(f"  n={n} N={N}: {dt:.0f}s  best={best:.1f}%  (resolution 1/{N}={100/N:.2f}%)", flush=True)
