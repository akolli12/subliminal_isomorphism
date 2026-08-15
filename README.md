# Subliminal Learning Attacks on Zero-Knowledge Self-Proving Models

Reproducible experiments for *Subliminal Learning Attacks on Zero-Knowledge
Self-Proving Models*. We train a small transformer as a prover for the
**GMW graph-isomorphism (GMW-GI) zero-knowledge protocol**, then show that a
simple coordinate-wise **witness extractor** — marginal estimation plus the
Hungarian algorithm — recovers the secret isomorphism from the prover's
commitment randomness at rates far above random guessing. Two defenses
(simulator-aligned training and witness masking) close the leak.

Every table in the paper maps to one seeded command. Results are written as
JSON and collated into `results/REPORT.md` next to the paper's published
numbers.

---

## The setup in one paragraph

An honest GMW-GI prover, given an instance `x = (G0, G1)` and witness
`φ` (the permutation with `φ(G0) = G1`), samples commitment randomness
`ψ ~ Unif(S_n)` **independently** of the witness. Zero-knowledge holds because
`ψ` carries no information about `φ`. A *trained* LLM prover, however, does not
sample perfectly uniformly: its per-coordinate marginals
`P[ψ(i) = v | x, φ]` deviate from `1/n` in a **witness-correlated** way. The
extractor estimates a table `τ` of these marginals, builds an `n × n` cost
matrix per test instance, and solves an assignment problem to read off `φ`.

The prover emits four permutation blocks per instance (paper §D.2):

```
[ φ | ψ | ψ⁻¹ | φ∘ψ⁻¹ ]          # permutation-only prover (main paper)
[ G0 | G1 | φ | ψ | ψ⁻¹ | φ∘ψ⁻¹ ] # graph-conditioned prover (§D.3, new experiment)
```

`φ` is the internally-computed witness (chain-of-thought, not revealed to the
verifier); `ψ` is the commitment; `ψ⁻¹` and `φ∘ψ⁻¹` are the auxiliary blocks
needed to answer both verifier challenges.

---

## Install

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
pytest -q                     # 35 correctness tests, no GPU/model needed
```

A CUDA GPU is required for the full sweep (original runs: single RTX 4090).

---

## Notebooks (start here)

Self-contained demos in `notebooks/` (each trains a small model inline, ~1–2 min):

| Notebook | Shows |
|---|---|
| `01_quickstart_attack.ipynb` | End-to-end: train a prover → estimate τ → extract the witness ≫ random. |
| `02_defenses.ipynb` | Both defenses via the **φ-dependence** metric (leak channel → noise floor). |
| `03_shared_length_generalizing.ipynb` | One delimiter-token model, usable + leaking at every n. |
| `04_graph_conditioned.ipynb` | The full GMW-GI input (`G1 = φ(G0)`). |

Regenerate with `python notebooks/build_notebooks.py`.

---

## One shared length-generalizing prover (delimiter tokens)

Beyond the per-n models, a **single** model handles all n via delimiter tokens
(`subliminal/multi.py`, `subliminal/multi_graph.py`) — the length-generalizing
prover the paper's §5.3 names as the direct path to strengthening the claim.
One model per experiment; witness-masking reuses the baseline model.

```bash
python experiments/multi_prover.py  --total 48000 --split paper --steps 30000 --extract  # baseline leak
python experiments/multi_defenses.py --total 48000 --steps 30000                          # both defenses
python experiments/multi_graph.py    --total 48000 --steps 30000                          # with/without (G0,G1)
python experiments/multi_extract.py  --ns 8                                               # n=8 (n=9 overnight)
```

The dataset budget is split across n by `--split {equal,linear,paper,factorial}`;
`paper` (∝ the per-n Table-4 sizes) keeps every n usable while leaking most.
Trained on n=4–9; the leak is reported at n=4–7 (n=8,9 via `multi_extract.py`).

---

## Reproduce every table (per-n models)

Each script trains any missing checkpoints, estimates `τ` (cached under
`data/`), runs the extractors, and writes JSON to `results/`.

```bash
# One shot: train all provers, then run all evaluations, then collate.
python experiments/train_all.py                 # 14 checkpoints (~20 min)
python experiments/run_table1.py                # Table 1  baseline, n=4..9
python experiments/run_table2.py                # Table 2  two defenses, n=4..6
python experiments/run_table3.py                # Table 3  (G0,G1)-conditioning, n=4,5
python experiments/run_table5.py                # Table 5  validity diagnostics
python experiments/run_graph_experiment.py      # NEW      non-abstracted graph prover
python experiments/make_tables.py               # -> results/REPORT.md (ours vs paper)
```

| Paper table | Script | What it measures |
|---|---|---|
| **Table 1** | `run_table1.py` | Top-n witness recovery on the baseline prover, `n=4..9`, six extractors. |
| **Table 2** | `run_table2.py` | Recovery on the simulator-aligned and witness-masked provers, `n=4..9`, **with a same-config baseline control**. |
| **Table 3** | `run_table3.py` | Best single-witness recovery **with vs without** `(G0,G1)` conditioning. |
| **Table 4** | `subliminal/configs.py` | Per-`n` training configuration (dataset size, steps, batch). |
| **Table 5** | `run_table5.py` | `ψ`-validity and `ψ⁻¹`-correctness diagnostics. |
| **New** | `run_graph_experiment.py` | Non-abstracted graph prover: witness-finding both directions + extraction under two conditionings. |
| **Ablation A** | `ablation_correlation.py` | Recovery on **seen** vs **unseen** witnesses — unseen leak too ⇒ the leak is correlational, not memorization. |
| **Ablation B** | `ablation_overtraining.py` | Functionality (ψ-valid, ψ⁻¹) + recovery vs steps — a **usable** prover leaks; not an undertraining artifact. |

Everything is seeded (`subliminal/seeding.py`, seeds recorded in each JSON):
dataset seed 0, training seed 0, `τ` seed 42, evaluation seed 0. Statistical
reproduction is exact given the same PyTorch/GPU; small deltas versus the paper
at `n=4,5` are sampling noise (the `n=4` test set is all 24 permutations, so a
single instance is 4.17%).

---

## The new experiment: no `(G0,G1)` abstraction

The main paper hands the prover the witness `φ` as input (§5.1), to sidestep
the difficulty of *solving* graph isomorphism and the confound of multiple
witnesses. `run_graph_experiment.py` runs the harder, honest version: a
**graph-learned** prover that must compute `φ` from `(G0,G1)` itself, then
commit `ψ`. We report:

- **Witness-finding accuracy, both directions.**
  Forward: present `(G0,G1)`, ask for `φ` with `φ(G0)=G1`.
  Reverse: present `(G1,G0)`, ask for `φ'` with `φ'(G1)=G0`.
  Both exact-match to the generating permutation and the (weaker)
  valid-isomorphism rate (which credits automorphism-equivalent witnesses).

- **Witness-extraction recovery under two conditionings.**
  `true-φ`: the extractor conditions `ψ` on `[G0|G1|φ_true]` — isolates leakage.
  `self-φ`: the extractor conditions on `[G0|G1]` only and the model generates
  its **own** `φ`, then `ψ` — the full self-witnessing pipeline
  (Assumption 3.2). Recovery here folds in witness-finding error.

Because random graphs may have automorphisms, exact-match witness-finding is a
lower bound; the valid-isomorphism rate is the fair figure for "did the model
find *a* witness." Both are reported so the multiple-witness confound is
visible rather than hidden.

---

## The claim, precisely

**The success metric, everywhere, is top-n witness recovery by a
polynomial-time extractor** (coordinate marginals → cost matrix → Hungarian for
top-1, Murty for top-n; `O(n⁵)` total). A poly-time attacker beats the random
baseline, so zero-knowledge leaks. The metric is inherently *correlational* —
the cost matrix is built from the marginals `P[ψ(i)=v | φ(j)=u]` and the
assignment solver exploits correlations across coordinates.

The claim is about a **usable prover**, not an undertrained one. An
undertrained network is not a prover at all: it cannot emit valid commitment
permutations or compute the inverse, so it fails the verifier's challenges.
Following the paper's Table-5 bar, we call a prover *usable* once its ψ are
valid permutations and its ψ⁻¹ is correct (the `c=0` branch the leak lives on).
The result: **a prover trained just enough to be usable already leaks the
witness**, and the leak persists with more training — it is not an
undertraining artifact. Two ablations, both scored with top-n recovery:

- **Ablation A (correlational reach).** Top-n recovery vs a witness's Hamming
  distance to the training set: witnesses near the training distribution recover
  far above random, so the leak follows *correlation* (it reaches unseen
  witnesses close to trained-on ones), not exact membership.
- **Ablation B (usable ≠ safe).** Functionality (ψ-valid, ψ⁻¹) and top-n
  recovery vs training steps: the prover becomes usable right around the paper's
  own training budget and leaks ~65% top-n (vs 4% random), holding across the
  whole usable regime. (φ∘ψ⁻¹, the harder `c=1` composition, stays low at this
  model size — matched to the self-proving-models setup it needs more data or a
  larger model — and is reported but not required.)

The defenses make `ψ` independent of `φ` (uniform target / attention mask) while
keeping the prover usable, dropping top-n recovery to the random baseline
(Table 2, a same-config controlled comparison).

## Table 2 is a controlled experiment (no cheating)

A defense that is a *training* intervention only takes effect once training has
converged. On the Table-4 dataset sizes (as few as 100 pairs) the
simulator-aligned prover under-converges: its shared network memorizes the tiny
dataset and leaks `φ` into `ψ` residually, so recovery stays above the random
floor. We therefore train the Table-2 provers on enough data/steps to converge
(`subliminal/configs.py::DEFENSE_TRAIN`).

To rule out the trivial explanation "the defense only works because of more
data," Table 2 also trains a **baseline-control at the identical config** (same
data, same steps, standard loss). Empirically the control **still leaks**
(e.g. `n=4`: control union 54% vs random 16.7%) while both defenses hit the
floor — so the KL loss and the attention mask, not the training budget, close
the leak. All three numbers are reported side by side in `REPORT.md`.

The Table 1 baseline is left at the paper's exact Table-4 config, so it still
demonstrates leakage under genuinely naive training.

## Repository layout

```
subliminal/            # importable library (pure, testable)
  layout.py            #   token-block layouts for both prover families
  model.py             #   TinyTransformer (§D.1); supports witness-masking
  data.py              #   permutation & graph generation, G1=φ(G0), sequences
  train.py             #   training loop; standard + uniform-KL (defense §4.1) losses
  sample.py            #   autoregressive ψ / witness sampling (+ masking)
  tau.py               #   τ-table estimation (Eq. 5) + six extractors (§5.1)
  assignment.py        #   Hungarian (top-1) + Murty (top-n), scales to n=9
  extract.py           #   run the attack, score top-1 / top-n recovery
  diagnostics.py       #   ψ-validity / ψ⁻¹-correctness (Table 5)
  contexts.py          #   context builders (perm vs graph)
  configs.py           #   per-n config (Table 4) + seeds
  multi.py             #   shared length-generalizing prover (delimiter tokens)
  multi_graph.py       #   shared graph-conditioned prover (G0,G1 in the input)
experiments/           # thin CLI drivers (see experiments/README.md for the map)
notebooks/             # self-contained demos + build_notebooks.py
tests/                 # pytest suite: core + multi + multi_graph + splits (35 tests)
results/               # JSON outputs + REPORT.md (git-ignored except REPORT.md)
```

---

## Correctness notes

- **Murty's algorithm** (`assignment.py`) is unit-tested to agree exactly with
  brute-force ranking; it is what makes top-n evaluation feasible at `n=9`
  (`9! ≈ 3.6·10⁵` assignments per instance otherwise).
- **Witness masking** (§4.2) is implemented by zeroing the `φ` token embeddings
  (positional embeddings retained), so `P[ψ|x,φ] = P[ψ]` exactly — avoiding the
  empty-attention-row pathology a pure attention mask would create at the first
  `ψ` position. For a fair pipeline, `τ` is re-estimated under the same masking.
- **Simulator-aligned training** (§4.1) replaces the `ψ` cross-entropy with soft
  CE against the autoregressive factorization of `Unif(S_n)` (uniform over the
  not-yet-used values at each position); its loss floor is `log(n!)/n`, verified
  in `tests/test_core.py`.
