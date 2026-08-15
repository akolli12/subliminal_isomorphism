# Experiments map

Thin CLI drivers over the `subliminal/` library. Two families: **per-n models**
(one model per n, the paper's original tables) and **shared models** (one
delimiter-token model for all n).

## Per-n models (paper tables)

| Script | Produces |
|---|---|
| `train_provers.py` | Train one prover (`baseline` / `sim-aligned` / `graph-given` / `graph-learned`) at a given n. |
| `train_all.py` | Train every per-n checkpoint. |
| `run_table1.py` | Table 1 — baseline leak, n=4–9. |
| `run_table2.py` | Table 2 — the two defenses, controlled. |
| `run_table3.py` | Table 3 — with/without (G0,G1) conditioning. |
| `run_table5.py` | Table 5 — validity diagnostics. |
| `run_graph_experiment.py` | Non-abstracted graph prover: witness-finding both directions + true-φ/self-φ extraction. |
| `run_sweep.py` | n-major driver running Tables 1/2/3 + ablations per n. |
| `ablation_correlation.py` | Ablation A — seen vs unseen (correlational leak). |
| `ablation_overtraining.py` | Ablation B — a *usable* prover leaks (functionality vs recovery). |
| `attack.py` | Reusable attack routine (tau cache + extraction). |
| `make_tables.py` | Collate `results/*.json` → `results/REPORT.md` vs the paper. |

## Shared length-generalizing models (delimiter tokens)

| Script | Produces |
|---|---|
| `multi_prover.py` | Train one shared prover on n=4–9; per-n usability + leak. `--split {equal,linear,paper,factorial}`. |
| `multi_defenses.py` | Simulator-aligned (new model) + witness-masked (reuse baseline). |
| `multi_graph.py` | Graph-conditioned shared model; with/without (G0,G1). |
| `multi_extract.py` | Extraction-only pass at chosen n (e.g. `--ns 8`; n=9 overnight). |
| `multi_sweep.sh` | Sweep size × split for the shared model. |

## Conventions

- Seeds are fixed and recorded in every result JSON (`subliminal/configs.py`).
- `checkpoints/`, `data/`, and `results/*.log` are git-ignored (regenerable).
- The success metric everywhere is **top-n recovery** by the polynomial-time
  extractor (Hungarian top-1 / Murty top-n).
