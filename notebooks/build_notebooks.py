"""Generate the demo notebooks (run this to (re)create notebooks/*.ipynb).

Each notebook is self-contained: it trains a small model inline (~1-2 min on a
GPU) so it runs without the large checkpoints. Execute with, e.g.:
    jupyter nbconvert --to notebook --execute notebooks/01_quickstart_attack.ipynb
"""

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


_CELL_ID = [0]


def _next_id():
    _CELL_ID[0] += 1
    return f"cell{_CELL_ID[0]:03d}"


def md(*lines):
    return {"cell_type": "markdown", "id": _next_id(), "metadata": {},
            "source": [l + "\n" for l in lines]}


def code(*lines):
    return {"cell_type": "code", "id": _next_id(), "metadata": {},
            "execution_count": None, "outputs": [], "source": [l + "\n" for l in lines]}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                           "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }


PATH_SETUP = code(
    "import sys, os",
    "sys.path.insert(0, os.path.abspath('..'))   # repo root",
    "import torch, itertools, math",
    "torch.manual_seed(0)",
    "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'",
    "print('device:', DEVICE)",
)


# ---------------------------------------------------------------------------
# 01 — quickstart: end-to-end witness-extraction attack
# ---------------------------------------------------------------------------
def nb_quickstart():
    return notebook([
        md("# Quickstart: extracting the witness from a self-proving model",
           "",
           "We train a tiny transformer as a GMW graph-isomorphism prover at",
           "`n=5`, confirm it is a **usable prover** (valid commitments ψ and a",
           "correct inverse ψ⁻¹), then run the **polynomial-time extractor**",
           "(coordinate marginals → Hungarian/Murty top-n) and show it recovers",
           "the secret permutation far above random guessing."),
        PATH_SETUP,
        md("### Train a small baseline prover (n=5)"),
        code(
            "from subliminal.layout import perm_layout",
            "from subliminal.data import make_perm_dataset, build_perm_sequences",
            "from subliminal.train import train_prover",
            "n = 5; layout = perm_layout(n)",
            "phi, psi = make_perm_dataset(3000, n, seed=0)",
            "vphi, vpsi = make_perm_dataset(500, n, seed=1)",
            "model = train_prover(layout, {'psi':'ce','psi_inv':'ce','phi_psi_inv':'ce'},",
            "    build_perm_sequences(phi, psi), build_perm_sequences(vphi, vpsi),",
            "    steps=4000, batch=32, lr=3e-4, seed=0,",
            "    ckpt_path='/tmp/quickstart_n5.pt', eval_every=10**9, log_every=1000)",
        ),
        md("### It's a usable prover: valid ψ and correct ψ⁻¹"),
        code(
            "from subliminal.diagnostics import psi_valid_diag, psi_inv_correct_diag",
            "from subliminal.data import rand_perms",
            "g = torch.Generator().manual_seed(7)",
            "ctx = rand_perms(2000, n, g)",
            "print('psi-valid   :', round(100*psi_valid_diag(model, ctx, layout),1), '%')",
            "seqs = build_perm_sequences(*make_perm_dataset(1000, n, 9))",
            "print('psi^-1 acc  :', round(100*psi_inv_correct_diag(model, seqs, layout),1), '%')",
        ),
        md("### Estimate the coordinate-marginal table τ and run the extractor"),
        code(
            "from subliminal.contexts import PermContext",
            "from subliminal.tau import estimate_tau, ExtractorBank, EXTRACTORS",
            "from subliminal.extract import run_extraction",
            "tau_raw, tau_log = estimate_tau(model, layout, k1=128, k2=128, seed=42,",
            "                                context_fn=PermContext(layout))",
            "bank = ExtractorBank(tau_raw, tau_log)",
            "perms = list(itertools.permutations(range(n)))     # all 120 witnesses",
            "res = run_extraction(model, layout, bank,",
            "    test_contexts=[torch.tensor(p) for p in perms],",
            "    true_witnesses=perms, k2=128, chunk=1<<15, seed=0)",
            "print('random baseline top-n: %.2f%%' % res['random_topn_pct'])",
            "for m in EXTRACTORS:",
            "    print(f\"  {m:24s} {res['extractors'][m]['topn_pct']:5.1f}%\")",
            "print('UNION (any extractor): %.1f%%' % res['union']['topn_pct'])",
        ),
        md("The extractor recovers the witness at rates **orders of magnitude**",
           "above the random baseline — zero-knowledge leaks."),
    ])


# ---------------------------------------------------------------------------
# 02 — the two defenses + the phi-dependence metric
# ---------------------------------------------------------------------------
def nb_defenses():
    return notebook([
        md("# Defenses: closing the leak",
           "",
           "The leak exists because the commitment ψ depends on the witness φ.",
           "We show the two defenses and measure the **φ-dependence** of ψ —",
           "the artifact-free quantity (the paper's own leakage metric):",
           "`max spread of P[ψ(i)=v | φ] across different φ`.",
           "",
           "- **Simulator-aligned** (§4.1): train ψ to uniform-on-remaining ⇒ ψ",
           "  becomes uniform and φ-independent.",
           "- **Witness-masking** (§4.2): zero the φ block at inference ⇒ ψ",
           "  becomes φ-independent (though not necessarily uniform)."),
        PATH_SETUP,
        code(
            "from subliminal.layout import perm_layout",
            "from subliminal.data import make_perm_dataset, build_perm_sequences, rand_perms",
            "from subliminal.train import train_prover",
            "from subliminal.contexts import PermContext",
            "from subliminal.sample import sample_psi, marginal_matrix",
            "n = 5; layout = perm_layout(n)",
            "tr = build_perm_sequences(*make_perm_dataset(3000, n, 0))",
            "vl = build_perm_sequences(*make_perm_dataset(500, n, 1))",
            "base = train_prover(layout, {'psi':'ce','psi_inv':'ce','phi_psi_inv':'ce'},",
            "    tr, vl, steps=4000, batch=32, lr=3e-4, seed=0, ckpt_path='/tmp/def_base.pt',",
            "    eval_every=10**9, log_every=10**9)",
            "simal = train_prover(layout, {'psi':'uniform','psi_inv':'ce','phi_psi_inv':'ce'},",
            "    tr, vl, steps=4000, batch=32, lr=3e-4, seed=0, ckpt_path='/tmp/def_sa.pt',",
            "    eval_every=10**9, log_every=10**9)",
        ),
        md("### Measure φ-dependence (the leak channel) and non-uniformity of ψ"),
        code(
            "def stats(model, zero_blocks=()):",
            "    g = torch.Generator().manual_seed(1)",
            "    Ms = []",
            "    for p in rand_perms(6, n, g):",
            "        ctx = p.unsqueeze(0).repeat(4000, 1)",
            "        Ms.append(marginal_matrix(sample_psi(model, ctx, layout, valid=True,",
            "                                             zero_blocks=zero_blocks), n))",
            "    Ms = torch.stack(Ms)",
            "    phidep = (Ms.max(0).values - Ms.min(0).values).abs().max().item()",
            "    nonunif = (Ms - 1.0/n).abs().max().item()",
            "    return phidep, nonunif",
            "print('model            phi-dependence   non-uniformity')",
            "print('baseline (leak)  %.3f            %.3f' % stats(base))",
            "print('simulator-aligned %.3f            %.3f' % stats(simal))",
            "print('witness-masked   %.3f            %.3f' % stats(base, (layout['phi'],)))",
        ),
        md("Both defenses drive **φ-dependence to the sampling-noise floor**",
           "(~0.02), closing the leak. Simulator-aligned additionally makes ψ",
           "uniform; witness-masking leaves ψ biased but φ-independent — which is",
           "all zero-knowledge requires."),
    ])


# ---------------------------------------------------------------------------
# 03 — one shared length-generalizing prover (delimiter tokens)
# ---------------------------------------------------------------------------
def nb_shared():
    return notebook([
        md("# One shared prover for all n (length generalization)",
           "",
           "A single model with **delimiter tokens** handles variable n:",
           "`[BOS | φ | SEP | ψ | SEP | ψ⁻¹ | SEP | φ∘ψ⁻¹]`. We train it on a",
           "mix of n=4..7 and show it is a usable prover — and still leaks — at",
           "every length."),
        PATH_SETUP,
        code(
            "from subliminal.multi import (specials, multi_seq_len, multi_layout,",
            "    build_multi_batch, MultiContext, IGNORE)",
            "from subliminal.model import TinyTransformer",
            "from subliminal.data import rand_perms",
            "import torch.nn.functional as F",
            "MAX_N = 7; NS = [4,5,6,7]",
            "g = torch.Generator().manual_seed(0)",
            "ns, phis, psis = [], [], []",
            "for nn in NS:",
            "    c = 3000",
            "    ns += [nn]*c; phis += list(rand_perms(c, nn, g)); psis += list(rand_perms(c, nn, g))",
            "toks, labels = build_multi_batch(ns, phis, psis, MAX_N)",
            "toks, labels = toks.to(DEVICE), labels.to(DEVICE)",
            "model = TinyTransformer(specials(MAX_N)['vocab'], multi_seq_len(MAX_N), 256, 4, 8).to(DEVICE)",
            "opt = torch.optim.AdamW(model.parameters(), lr=3e-4)",
            "for step in range(8000):",
            "    idx = torch.randint(0, toks.shape[0], (64,), device=DEVICE)",
            "    loss = F.cross_entropy(model(toks[idx]).reshape(-1, specials(MAX_N)['vocab']),",
            "                           labels[idx].reshape(-1), ignore_index=IGNORE)",
            "    opt.zero_grad(); loss.backward(); opt.step()",
            "    if step % 2000 == 0: print('step', step, 'loss %.3f' % loss.item())",
        ),
        md("### Usable prover + leak at every n, from the same model"),
        code(
            "from subliminal.diagnostics import psi_valid_diag, psi_inv_correct_diag",
            "from subliminal.tau import estimate_tau, ExtractorBank, EXTRACTORS",
            "from subliminal.extract import run_extraction",
            "import itertools, random",
            "for n in NS:",
            "    lay = multi_layout(n, MAX_N); ctxfn = MultiContext(n, MAX_N)",
            "    gg = torch.Generator().manual_seed(1)",
            "    valid = psi_valid_diag(model, ctxfn(rand_perms(1000, n, gg)), lay)",
            "    tr, tl = estimate_tau(model, lay, k1=64, k2=64, seed=42, context_fn=ctxfn)",
            "    perms = list(itertools.permutations(range(n)))",
            "    if len(perms) > 360: random.Random(0).shuffle(perms); perms = perms[:360]",
            "    ctxs = [ctxfn(torch.tensor(p).unsqueeze(0))[0] for p in perms]",
            "    res = run_extraction(model, lay, ExtractorBank(tr, tl), test_contexts=ctxs,",
            "        true_witnesses=perms, k2=64, chunk=1<<15, seed=0)",
            "    print(f'n={n}: psi-valid {100*valid:5.1f}%  union top-n {res[\"union\"][\"topn_pct\"]:5.1f}%'",
            "          f'  (random {res[\"random_topn_pct\"]:.2g}%)')",
        ),
        md("One model, every length: usable prover, and the leak persists."),
    ])


# ---------------------------------------------------------------------------
# 04 — graph-conditioned prover (full GMW-GI input)
# ---------------------------------------------------------------------------
def nb_graph():
    return notebook([
        md("# Graph-conditioned prover (the full GMW-GI input)",
           "",
           "We include the actual graphs `G0, G1` in the sequence, with",
           "`G1 = φ(G0)`, and check that conditioning on the graphs does not",
           "change the leak (the attack lives in the ψ-vs-φ structure)."),
        PATH_SETUP,
        code(
            "from subliminal.data import rand_graphs, rand_perms, apply_perm_to_graph, is_isomorphism",
            "n = 5; g = torch.Generator().manual_seed(0)",
            "g0 = rand_graphs(4, n, g); phi = rand_perms(4, n, g)",
            "g1 = apply_perm_to_graph(g0, phi, n)",
            "print('G1 = phi(G0) holds for all rows:', bool(is_isomorphism(phi, g0, g1, n).all()))",
        ),
        md("The graph-conditioned sequence layout:"),
        code(
            "from subliminal.multi_graph import graph_multi_layout, graph_seq_len, build_graph_multi_example",
            "MAX_N = 7",
            "toks, labels = build_graph_multi_example(n, g0[0], phi[0], rand_perms(1, n, g)[0], MAX_N)",
            "lay = graph_multi_layout(n, MAX_N)",
            "print('sequence length (max):', graph_seq_len(MAX_N))",
            "for name in ['g0','g1','phi','psi','psi_inv','phi_psi_inv']:",
            "    print(f'  {name:12s} tokens:', toks[lay[name]].tolist())",
        ),
        md("Training a full graph-conditioned shared model and running the",
           "attack follows exactly the same recipe — see",
           "`experiments/multi_graph.py`. The result: **with ≈ without**",
           "`(G0,G1)` conditioning, so abstracting the graphs away does not",
           "change the leak."),
    ])


def main():
    nbs = {
        "01_quickstart_attack.ipynb": nb_quickstart(),
        "02_defenses.ipynb": nb_defenses(),
        "03_shared_length_generalizing.ipynb": nb_shared(),
        "04_graph_conditioned.ipynb": nb_graph(),
    }
    for name, nb in nbs.items():
        with open(os.path.join(HERE, name), "w") as f:
            json.dump(nb, f, indent=1)
        print("wrote", name)


if __name__ == "__main__":
    main()
