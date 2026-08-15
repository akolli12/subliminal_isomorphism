"""Empirical null baseline: recovery of a GENUINELY uniform sampler through the
SAME extractor (matched K). This is the correct null for finite-sample runs —
the single-witness extractor has a small upward bias at small n (it selects the
noisiest coordinate), so the theoretical 100*n/n! understates the null there.
"""
import itertools, math, json, os, argparse
import torch, torch.nn.functional as F
from subliminal.data import rand_perms
from subliminal.tau import ExtractorBank, EXTRACTORS
from subliminal.extract import recover_scores

def null_for(n, K, seed=0, cap=720):
    g = torch.Generator().manual_seed(seed)
    um = lambda: F.one_hot(rand_perms(K, n, g), n).float().mean(0)
    tr = torch.zeros(n,n,n,n); tl = torch.zeros(n,n,n,n)
    for j in range(n):
        for u in range(n):
            M = torch.stack([um() for _ in range(K)])
            tr[:,j,:,u]=M.mean(0); tl[:,j,:,u]=M.clamp_min(1e-12).log().mean(0)
    bank = ExtractorBank(tr, tl)
    perms = list(itertools.permutations(range(n)))
    if len(perms) > cap:
        import random; random.Random(seed).shuffle(perms); perms = perms[:cap]
    top = {m:0 for m in EXTRACTORS}; union=0
    for p in perms:
        sc = recover_scores(bank, um(), tuple(p), n); a=False
        for m in EXTRACTORS:
            top[m]+=sc[m]['topn']; a|=sc[m]['topn']
        union+=a
    T=len(perms)
    return {"theoretical_topn_pct":100*n/math.factorial(n),
            "empirical_null_best_pct": max(100*top[m]/T for m in EXTRACTORS),
            "empirical_null_union_pct": 100*union/T, "num_test":T}

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--ns",type=int,nargs="+",default=[4,5,6,7])
    ap.add_argument("--K",type=int,default=128); a=ap.parse_args()
    out={"K":a.K,"by_n":{}}
    for n in a.ns:
        out["by_n"][n]=null_for(n,a.K)
        r=out["by_n"][n]
        print(f"n={n}: theoretical {r['theoretical_topn_pct']:.2f}%  |  empirical null best {r['empirical_null_best_pct']:.1f}%  union {r['empirical_null_union_pct']:.1f}%")
    json.dump(out, open("results/empirical_null.json","w"), indent=2)
    print("wrote results/empirical_null.json")
