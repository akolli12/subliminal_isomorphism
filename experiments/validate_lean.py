import sys, time, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from subliminal.model import TinyTransformer
from subliminal.multi import specials
from subliminal.multi_graph import graph_seq_len
from graph_perinstance import (attack_graph_perinstance,
                               attack_graph_perinstance_batched, EXTRACTORS)

MAX_N = 9
dev = "cuda"
m = TinyTransformer(specials(MAX_N)["vocab"], graph_seq_len(MAX_N), 256, 4, 8).to(dev)
m.load_state_dict(torch.load("checkpoints/mt_graph_base_seed0_exponential_n9.pt",
                             map_location=dev))
m.eval()

print("=== speedup + agreement (n=6, N=24) ===", flush=True)
t0 = time.time()
rs = attack_graph_perinstance(m, 6, MAX_N, fix="G0", num_instances=24,
                              k1=48, k2=48, seed=0)
ts = time.time() - t0
t0 = time.time()
rb = attack_graph_perinstance_batched(m, 6, MAX_N, fix="G0", num_instances=24,
                                      k1=48, k2=48, seed=0)
tb = time.time() - t0
print(f"  slow {ts:.0f}s  batched {tb:.0f}s  -> {ts/tb:.1f}x faster", flush=True)
for x in EXTRACTORS:
    print(f"    {x:24} slow {rs['extractors'][x]['topn_pct']:5.1f}  "
          f"batched {rb['extractors'][x]['topn_pct']:5.1f}", flush=True)

print("=== n=8,9 batched feasibility ===", flush=True)
for n in [8, 9]:
    for N in [120, 300]:
        t0 = time.time()
        r = attack_graph_perinstance_batched(m, n, MAX_N, fix="G0",
                                             num_instances=N, k1=48, k2=48, seed=0)
        dt = time.time() - t0
        best = max(r["extractors"][x]["topn_pct"] for x in EXTRACTORS)
        print(f"  n={n} N={N}: {dt:.0f}s  best={best:.1f}%  "
              f"(1/{N}={100/N:.2f}% resolution)", flush=True)
