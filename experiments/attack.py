"""Reusable attack routine: (train if needed) -> estimate tau (cache) -> extract.

Used by all perm-prover table scripts (Tables 1, 2). Keeps tau estimation
cached on disk keyed by (tag, n) so re-running extraction is cheap.
"""

import os

import torch

from _common import ckpt_path, tau_path, full_or_sample_perms
from subliminal.configs import CONFIGS, SEED_TAU, SEED_EVAL
from subliminal.contexts import PermContext
from subliminal.extract import run_extraction
from subliminal.layout import perm_layout
from subliminal.tau import estimate_tau, ExtractorBank
from subliminal.train import load_prover

# Full S_n eval where cheap; a fixed seeded sample for the largest n.
MAX_TEST = {4: None, 5: None, 6: None, 7: None, 8: 5000, 9: 5000}


def get_or_build_tau(model, layout, tag, n, context_fn, zero_blocks=(),
                     force=False):
    """Estimate (or load cached) tau under the given sampling regime.

    zero_blocks is threaded into estimate_tau so that, for the witness-masking
    defense, tau reflects the same masked distribution the extractor observes
    at test time — a consistent, honest pipeline.
    """
    raw_p, log_p = tau_path(tag, n, "raw"), tau_path(tag, n, "log")
    if not force and os.path.exists(raw_p) and os.path.exists(log_p):
        print(f"  loading cached tau for {tag} n={n}")
        return torch.load(raw_p), torch.load(log_p)
    cfg = CONFIGS[n]
    tau_raw, tau_log = estimate_tau(
        model, layout, k1=cfg.k1, k2=cfg.k2, seed=SEED_TAU,
        context_fn=context_fn, zero_blocks=zero_blocks)
    torch.save(tau_raw, raw_p)
    torch.save(tau_log, log_p)
    return tau_raw, tau_log


def attack_perm_prover(tag, n, *, zero_blocks_name=None, force_tau=False):
    """Attack a perm-only prover checkpoint. Returns a results dict.

    zero_blocks_name: if 'phi', apply the witness-masking defense at inference
    (zero the phi block when sampling psi). tau is estimated under the SAME
    masking so the extractor uses the model's actually-observable distribution.
    """
    cfg = CONFIGS[n]
    layout = perm_layout(n)
    model = load_prover(ckpt_path(tag, n), layout,
                        d_model=cfg.d_model, n_heads=cfg.n_heads,
                        n_layers=cfg.n_layers)
    ctx = PermContext(layout)
    zero_blocks = (layout["phi"],) if zero_blocks_name == "phi" else ()
    tau_tag = tag if zero_blocks_name is None else f"{tag}_maskphi"

    tau_raw, tau_log = get_or_build_tau(model, layout, tau_tag, n, ctx,
                                        zero_blocks=zero_blocks, force=force_tau)
    bank = ExtractorBank(tau_raw, tau_log)
    perms = full_or_sample_perms(n, MAX_TEST[n], SEED_EVAL)
    contexts = [torch.tensor(p) for p in perms]
    truths = [p for p in perms]
    res = run_extraction(model, layout, bank, test_contexts=contexts,
                         true_witnesses=truths, k2=cfg.k2, chunk=1 << 15,
                         seed=SEED_EVAL, zero_blocks=zero_blocks)
    res["tag"] = tag
    res["defense"] = zero_blocks_name or "none"
    return res


def attack_on_perms(model, layout, tau_raw, tau_log, perms, *, k2, seed):
    """Attack an already-loaded model+tau on an explicit list of test perms.

    `perms` is a list of tuples. Returns the same results dict as
    run_extraction. Used by the memorization ablations, which attack
    hand-picked seen / unseen witness sets rather than all of S_n.
    """
    bank = ExtractorBank(tau_raw, tau_log)
    contexts = [torch.tensor(p) for p in perms]
    truths = list(perms)
    return run_extraction(model, layout, bank, test_contexts=contexts,
                          true_witnesses=truths, k2=k2, chunk=1 << 15, seed=seed)
