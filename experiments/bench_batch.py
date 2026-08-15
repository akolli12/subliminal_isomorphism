import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from subliminal.model import TinyTransformer
from subliminal.multi import specials
from subliminal.multi_graph import graph_seq_len
from graph_perinstance import attack_graph_perinstance_batched, EXTRACTORS

MAX_N = 9
dev = "cuda"
m = TinyTransformer(specials(MAX_N)["vocab"], graph_seq_len(MAX_N), 256, 4, 8).to(dev)
m.load_state_dict(torch.load("checkpoints/mt_graph_base_seed0_exponential_n9.pt",
                             map_location=dev))
m.eval()

# small warmup to load kernels
_ = attack_graph_perinstance_batched(m, 5, MAX_N, fix="G0", num_instances=8,
                                     k1=8, k2=8, seed=0, sample_batch=4096)
for n, N in [(5, 60), (8, 60)]:
    print(f"\n### n={n} N={N} ###", flush=True)
    for sb in [8192, 32768, 98304, 196608]:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        try:
            r = attack_graph_perinstance_batched(m, n, MAX_N, fix="G0",
                                                 num_instances=N, k1=48, k2=48,
                                                 seed=0, sample_batch=sb)
            torch.cuda.synchronize()
            dt = time.time() - t0
            peak = torch.cuda.max_memory_allocated() / 1e9
            best = max(r["extractors"][x]["topn_pct"] for x in EXTRACTORS)
            print(f"  sample_batch={sb:7d}: {dt:6.1f}s  peak={peak:5.2f}GB  best={best:.1f}%",
                  flush=True)
        except RuntimeError as e:
            print(f"  sample_batch={sb:7d}: OOM/err ({str(e)[:60]})", flush=True)
            torch.cuda.empty_cache()
