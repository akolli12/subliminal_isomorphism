import sys, random, itertools
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import linear_sum_assignment

import config
from model import TinyTransformer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N = config.N
EPS = 1e-12
print(f'N = {N}, device = {DEVICE}')

def load_model(path):
    m = TinyTransformer().to(DEVICE)
    m.load_state_dict(torch.load(path, map_location=DEVICE))
    m.eval()
    return m

@torch.no_grad()
def sample_psi_unconstrained(model, phis):
    model.eval()
    if phis.dim() == 1:
        phis = phis.unsqueeze(0)
    B = phis.shape[0]
    psi_start = config.PSI.start
    psi_buf = torch.zeros((B, N), dtype=torch.long, device=DEVICE)
    seq = torch.cat([phis.to(DEVICE), psi_buf], dim=1)
    for j in range(N):
        logits = model(seq)
        step_logits = logits[:, psi_start - 1 + j, :]
        probs = F.softmax(step_logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        seq[:, psi_start + j] = sampled
    return seq[:, psi_start:psi_start + N]

def is_perm_rows(samples, n):
    sorted_, _ = torch.sort(samples, dim=1)
    expected = torch.arange(n, device=samples.device).unsqueeze(0).expand_as(sorted_)
    return (sorted_ == expected).all(dim=1)

@torch.no_grad()
def sample_psi_rejection(model, phis, max_attempts=1000):
    model.eval()
    if phis.dim() == 1:
        phis = phis.unsqueeze(0)
    B = phis.shape[0]
    phis = phis.to(DEVICE)
    out     = torch.zeros((B, N), dtype=torch.long, device=DEVICE)
    pending = torch.ones(B,        dtype=torch.bool, device=DEVICE)
    for _ in range(max_attempts):
        idx = torch.nonzero(pending, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            break
        psis_try = sample_psi_unconstrained(model, phis[idx])
        valid = is_perm_rows(psis_try, N)
        out[idx[valid]] = psis_try[valid]
        pending[idx[valid]] = False
    if pending.any():
        raise RuntimeError(
            f'{int(pending.sum().item())} rows still invalid after {max_attempts} attempts'
        )
    return out

@torch.no_grad()                                                                                                            
def sample_psi_rejection_fallback(model, phis, max_attempts=1000):
    """Whole-sequence rejection: resample any row whose psi isn't a permutation.                                            
                                                                                                                            
    Soft fail: if some rows are still invalid after max_attempts, keep their                                                
    last (possibly repeating) sample and print a warning instead of raising.                                                
    """         
    model.eval()                                                                                                            
    if phis.dim() == 1:                                                                                                     
        phis = phis.unsqueeze(0)
    B = phis.shape[0]                                                                                                       
    phis = phis.to(DEVICE)
    out     = torch.zeros((B, N), dtype=torch.long, device=DEVICE)
    pending = torch.ones(B,        dtype=torch.bool, device=DEVICE)                                                         
                                                                                                                            
    for _ in range(max_attempts):                                                                                           
        idx = torch.nonzero(pending, as_tuple=False).squeeze(1)                                                             
        if idx.numel() == 0:
            break
        psis_try = sample_psi_unconstrained(model, phis[idx])
        valid = is_perm_rows(psis_try, N)                                                                                   
        # Always write this attempt; rows that are still pending will get
        # overwritten next round, valid rows are frozen by `pending=False`.                                                 
        out[idx] = psis_try                                                                                                 
        pending[idx[valid]] = False                                                                                         
                                                                                                                            
    if pending.any():
        n_failed = int(pending.sum().item())                                                                                
        print(f"  [warn] {n_failed}/{B} rows still invalid after {max_attempts} "
            f"attempts; keeping last (possibly-invalid) sample")                                                          
                                                                                                                            
    return out      

def marginal_matrix(samples, n=N):
    return F.one_hot(samples.long(), n).float().mean(dim=0)

def random_phis_with_constraint(B, j, u, n=N, seed=None):
    g = torch.Generator()
    if seed is not None: g.manual_seed(seed)
    other_positions = torch.tensor([i for i in range(n) if i != j], dtype=torch.long)
    other_values    = torch.tensor([v for v in range(n) if v != u], dtype=torch.long)
    keys = torch.rand((B, n - 1), generator=g)
    perms = keys.argsort(dim=1)
    perm_values = other_values[perms]
    out = torch.zeros((B, n), dtype=torch.long)
    out[:, j] = u
    out[:, other_positions] = perm_values
    return out

def to_tensor(tau_dict):
    """Pack {(i, j, v, u): scalar} → tensor of shape (N, N, N, N) indexed [i, j, v, u]."""
    T = torch.zeros((N, N, N, N), device=DEVICE)
    for i in range(N):
        for j in range(N):
            for v in range(N):
                for u in range(N):
                    T[i, j, v, u] = tau_dict[(i, j, v, u)]
    return T

def create_taus(K1, K2, CHUNK):
    tau     = {}
    tau_log = {}

    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)

   
    for j in range(N):
        for u in range(N):
            phis = random_phis_with_constraint(K1, j=j, u=u)
            sum_p     = torch.zeros((N, N), device=DEVICE)
            sum_log_p = torch.zeros((N, N), device=DEVICE)

            if K2 <= CHUNK:
                phi_chunk = max(1, CHUNK // K2)
                psi_subchunk = K2
            else:
                phi_chunk = 1
                psi_subchunk = CHUNK

            for phi_start in range(0, K1, phi_chunk):
                P = min(phi_chunk, K1 - phi_start)
                this_phis = phis[phi_start:phi_start + P]
                counts = torch.zeros((P, N, N), device=DEVICE)
                for psi_start in range(0, K2, psi_subchunk):
                    S = min(psi_subchunk, K2 - psi_start)
                    phi_rep = this_phis.repeat_interleave(S, dim=0)
                    psis    = sample_psi_rejection_fallback(model, phi_rep)
                    psis    = psis.view(P, S, N)
                    counts += F.one_hot(psis.long(), N).float().sum(dim=1)
                M_per_phi = counts / K2
                sum_p     += M_per_phi.sum(dim=0)
                sum_log_p += torch.log(M_per_phi.clamp_min(EPS)).sum(dim=0)

            avg_p     = sum_p / K1
            avg_log_p = sum_log_p / K1
            for i in range(N):
                for v in range(N):
                    tau[(i, j, v, u)]     = avg_p[i, v]
                    tau_log[(i, j, v, u)] = avg_log_p[i, v]
            print(f'  cell (j={j}, u={u}) done')
    return tau, tau_log

def pick_witnesses(tau_dict):
    """Return (opts_max_spread, opts_max_min_spread): each {j: (i_j, v_j)}."""
    opts_ms, opts_mm = {}, {}
    for j in range(N):
        scores_ms = torch.zeros((N, N))
        scores_mm = torch.zeros((N, N))
        for i in range(N):
            for v in range(N):
                u_func = torch.tensor([tau_dict[(i, j, v, u)].item()
                                       if torch.is_tensor(tau_dict[(i, j, v, u)])
                                       else tau_dict[(i, j, v, u)]
                                       for u in range(N)])
                scores_ms[i, v] = u_func.max() - u_func.min()
                sorted_, _ = torch.sort(u_func, descending=True)
                scores_mm[i, v] = sorted_[0] - sorted_[1]
        i_ms = (scores_ms.argmax() // N).item(); v_ms = (scores_ms.argmax() % N).item()
        i_mm = (scores_mm.argmax() // N).item(); v_mm = (scores_mm.argmax() % N).item()
        opts_ms[j], opts_mm[j] = (i_ms, v_ms), (i_mm, v_mm)
    return opts_ms, opts_mm

def build_cost(kind, opts_dict, tau_dict, mm, log_mm):
    """Return (N, N) cost matrix on CPU."""
    if kind == 'raw':
        C = torch.zeros((N, N))
        for j in range(N):
            i_j, v_j = opts_dict[j]
            p_j = mm[i_j, v_j]
            for u in range(N):
                C[j, u] = torch.abs(tau_dict[(i_j, j, v_j, u)] - p_j)
        return C
    if kind == 'log':
        C = torch.zeros((N, N))
        for j in range(N):
            i_j, v_j = opts_dict[j]
            lp_j = log_mm[i_j, v_j]
            for u in range(N):
                C[j, u] = torch.abs(tau_dict[(i_j, j, v_j, u)] - lp_j)
        return C

    # aggregate methods: broadcast (N, N) test marginal vs. (N, N, N, N) tau tensor
    if kind == 'agg_l2_raw':
        diff = mm.unsqueeze(1).unsqueeze(3) - tau_tensor
        return (diff.abs() ** 2).sum(dim=(0, 2)).sqrt().cpu()
    if kind == 'agg_l1_raw':
        diff = mm.unsqueeze(1).unsqueeze(3) - tau_tensor
        return diff.abs().sum(dim=(0, 2)).cpu()
    if kind == 'agg_linf_raw':
        diff = mm.unsqueeze(1).unsqueeze(3) - tau_tensor
        return diff.abs().amax(dim=(0, 2)).cpu()
    if kind == 'agg_l2_log':
        diff = log_mm.unsqueeze(1).unsqueeze(3) - tau_log_tensor
        return (diff.abs() ** 2).sum(dim=(0, 2)).sqrt().cpu()
    if kind == 'agg_l1_log':
        diff = log_mm.unsqueeze(1).unsqueeze(3) - tau_log_tensor
        return diff.abs().sum(dim=(0, 2)).cpu()
    if kind == 'agg_linf_log':
        diff = log_mm.unsqueeze(1).unsqueeze(3) - tau_log_tensor
        return diff.abs().amax(dim=(0, 2)).cpu()
    raise ValueError(f'unknown kind: {kind}')

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate all methods")
    parser.add_argument('--checkpoint', type=str, default=str(PROJECT_ROOT / 'checkpoints' / f'model_{config.STEPS}_{N}.pt'),
                        help='Path to the model checkpoint to load')
    parser.add_argument('--K1_exp', type=int, default=9,                                                                        
                        help="K1 = 2**K1_exp; phis per (j, u) (default: %(default)d → K1 = %(default)d^2 = ...)")               
    parser.add_argument('--K2_exp', type=int, default=9,                                                                        
                        help="K2 = 2**K2_exp; psis per phi (default: %(default)d)")
    parser.add_argument('--chunk_exp', type=int, default=17,                                                                    
                        help="chunk = 2**chunk_exp; rows per model.forward call (default: %(default)d)") 
    parser.add_argument('--num_evals', type=int, default=None,                                                                  
                        help="Number of evaluations to run (default: None — use full set)")   
    args = parser.parse_args()

    CHECKPOINT = PROJECT_ROOT / 'checkpoints' / args.checkpoint

    model = load_model(CHECKPOINT)

    K1    = 2 ** args.K1_exp # phis per (j, u)                                                                                                
    K2    = 2 ** args.K2_exp # psis per phi (large to keep log-bias small)
    CHUNK = 2 ** args.chunk_exp # rows per model.forward call                                                                                          
    
    print(f"K1 = 2**{args.K1_exp} = {K1}")                                                                                      
    print(f"K2 = 2**{args.K2_exp} = {K2}")
    print(f"CHUNK = 2**{args.chunk_exp} = {CHUNK}") 

    # K1 = 2**9      # phis per (j, u)
    # K2 = 2**9         # psis per phi (large to keep log-bias small)
    # CHUNK = 2**17     # rows per model.forward call

    tau, tau_log = create_taus(K1, K2, CHUNK)
    tau_tensor     = to_tensor(tau)
    tau_log_tensor = to_tensor(tau_log)
    torch.save(tau_tensor, PROJECT_ROOT / f'data/tau_tensor_{N}.pt')
    torch.save(tau_log_tensor, PROJECT_ROOT / f'data/tau_log_tensor_{N}.pt')

    opts_max_spread,     opts_max_min_spread     = pick_witnesses(tau)
    opts_max_spread_log, opts_max_min_spread_log = pick_witnesses(tau_log)

                                            # forward-pass batch size
    permutations = list(itertools.permutations(range(N)))

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)

    # (witness_dict_or_None, tau_dict_or_None, kind)
    METHODS = {
        'max-min raw':       (opts_max_min_spread,     tau,     'raw'),
        'max     raw':       (opts_max_spread,         tau,     'raw'),
        'max-min log':       (opts_max_min_spread_log, tau_log, 'log'),
        'max     log':       (opts_max_spread_log,     tau_log, 'log'),
        'aggregate L2  log': (None, None, 'agg_l2_log'),
        'aggregate L1  log': (None, None, 'agg_l1_log'),
        'aggregate L\u221E  log': (None, None, 'agg_linf_log'),
        'aggregate L2  raw': (None, None, 'agg_l2_log'),
        'aggregate L1  raw': (None, None, 'agg_l1_raw'),
        'aggregate L\u221E  raw': (None, None, 'agg_linf_raw'),
    }

    ranks = {m: [] for m in METHODS}
    top1  = {m: 0  for m in METHODS}
    topN  = {m: 0  for m in METHODS}
    union_top1 = 0
    union_topN = 0

    if args.num_evals is not None:
        np.random.seed(seed)
        permutations = np.array(permutations)
        np.random.shuffle(permutations)
        permutations = [tuple(int(x) for x in p) for p in permutations[:args.num_evals]]

    for permutation in permutations:
        base = torch.tensor(permutation, device=DEVICE).long()
        counts = torch.zeros((N, N), device=DEVICE)
        for start in range(0, K2, CHUNK):
            S = min(CHUNK, K2 - start)
            perm_rep = base.unsqueeze(0).expand(S, -1).contiguous()
            psis = sample_psi_rejection(model, perm_rep)
            counts += F.one_hot(psis.long(), N).float().sum(dim=0)
        mm     = counts / K2
        log_mm = torch.log(mm.clamp_min(EPS))

        truth_tup = tuple(permutation)
        print(f'truth = {truth_tup}')
        iter_in_top1 = iter_in_topN = False

        for name, (opts_dict, tau_dict, kind) in METHODS.items():
            C = build_cost(kind, opts_dict, tau_dict, mm, log_mm)
            full   = top_k_assignments(C, k=None)
            top_N  = full[:N]
            rank   = next(i + 1 for i, (p, _) in enumerate(full) if tuple(p.tolist()) == truth_tup)
            in_top = rank <= N
            ranks[name].append(rank)
            if rank == 1:
                top1[name] += 1
                iter_in_top1 = True
            if in_top:
                topN[name] += 1
                iter_in_topN = True
            print(f'  {name:<22}  best={full[0][0].tolist()} (cost {full[0][1]:.4f})  '
                f'rank={rank}/{len(permutations)}  in-top-{N}={in_top}')
        if iter_in_top1: union_top1 += 1
        if iter_in_topN: union_topN += 1
        print()

    T = len(permutations)
    print('=' * 90)
    print(f'Summary over all {T} permutations of S_{N}')
    print(f'Random baselines: top-1 ~ {100/T:.1f}%, top-{N} ~ {100*N/T:.1f}%, mean rank ~ {(T+1)/2:.1f}')
    print('-' * 90)
    print(f'{"method":<24} {"top-1":>10} {"top-N":>10} {"mean rank":>14}')
    for m in METHODS:
        print(f'{m:<24} {top1[m]}/{T:<6} {topN[m]}/{T:<6} {sum(ranks[m])/T:>12.2f}')
    print('-' * 90)
    print(f'{"union (any method)":<24} {union_top1}/{T:<6} {union_topN}/{T:<6}')



