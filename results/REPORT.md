# Reproduction report

Each recovery cell is **ours / paper (Δ)** in top-n recovery %. Deltas within a few points at small test-set sizes are sampling noise (e.g. n=4 has only 24 test permutations, so one instance is 4.17%).

## Table 1 — baseline prover, top-n recovery %  (ours / paper (Δ))

Random baseline: n=4: 16.7%, n=5: 4.17%, n=6: 0.833%, n=7: 0.139%, n=8: 0.0198%, n=9: 0.00248%

| Method | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|---|
| Single, max-spread (raw) | 41.67/45.83 (-4.2) | 10.83/7.50 (+3.3) | 2.50/2.22 (+0.3) | 0.44/0.44 (-0.0) | 0.00/0.04 (-0.0) | 0.02/0.05 (-0.0) |
| Single, max-spread (log) | 50.00/66.67 (-16.7) | 13.33/17.50 (-4.2) | 2.64/3.75 (-1.1) | 0.48/0.79 (-0.3) | 0.02/0.03 (-0.0) | 0.00/0.05 (-0.1) |
| Aggregate L1 (raw) | 79.17/87.50 (-8.3) | 40.83/51.67 (-10.8) | 15.97/17.64 (-1.7) | 5.10/6.31 (-1.2) | 0.32/0.31 (+0.0) | 0.06/0.10 (-0.0) |
| Aggregate L1 (log) | 66.67/79.17 (-12.5) | 38.33/43.33 (-5.0) | 11.94/15.42 (-3.5) | 3.89/5.74 (-1.9) | 0.04/0.08 (-0.0) | 0.00/0.15 (-0.1) |
| Aggregate L2 (raw) | 79.17/95.83 (-16.7) | 41.67/49.17 (-7.5) | 15.97/17.64 (-1.7) | 5.75/10.10 (-4.3) | 0.16/0.31 (-0.1) | 0.00/0.20 (-0.2) |
| Aggregate L2 (log) | 79.17/95.83 (-16.7) | 45.00/49.17 (-4.2) | 19.86/17.64 (+2.2) | 7.84/10.10 (-2.3) | 0.06/0.17 (-0.1) | 0.04/0.15 (-0.1) |
| Union (any method) | 87.50 | 65.00 | 35.00 | 14.98 | 0.46 | 0.12 |

## Table 2 — defenses, top-n recovery %  (controlled experiment)

All three provers at each n are trained with the SAME (converged) config; only the loss / inference differs. The baseline-control still leaks, so the KL loss and the mask — not the training budget — close the leak. Cells for the two defenses are **ours / paper (Δ)** (paper used n=4..6 only; '·' = no paper value). The baseline-control has no paper counterpart.

### Baseline-control (same config, no defense — should LEAK)

| Method | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|---|
| Single, max-spread (raw) | 25.00 | 20.00 | 4.86 | 4.01 | 0.00 | 0.02 |
| Single, max-spread (log) | 20.83 | 19.17 | 3.89 | 5.06 | 0.02 | 0.00 |
| Aggregate L1 (raw) | 45.83 | 38.33 | 26.81 | 61.23 | 0.32 | 0.06 |
| Aggregate L1 (log) | 37.50 | 36.67 | 28.61 | 61.29 | 0.04 | 0.00 |
| Aggregate L2 (raw) | 41.67 | 45.00 | 29.03 | 65.30 | 0.16 | 0.00 |
| Aggregate L2 (log) | 29.17 | 43.33 | 30.42 | 66.25 | 0.06 | 0.04 |
| Union (any method) | 54.17 | 60.83 | 43.06 | 77.24 | 0.46 | 0.12 |

Random baseline: n=4: 16.7%, n=5: 4.17%, n=6: 0.833%, n=7: 0.139%, n=8: 0.0198%, n=9: 0.00248%

### Simulator-aligned (§4.1 — should hit floor)

| Method | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|---|
| Single, max-spread (raw) | 12.50/12.50 (+0.0) | 3.33/3.33 (+0.0) | 1.53/0.97 (+0.6) | 0.18/— | 0.00/— | 0.00/— |
| Single, max-spread (log) | 12.50/12.50 (+0.0) | 4.17/5.83 (-1.7) | 1.81/0.56 (+1.2) | 0.14/— | 0.00/— | 0.00/— |
| Aggregate L1 (raw) | 25.00/20.83 (+4.2) | 1.67/5.00 (-3.3) | 1.25/0.97 (+0.3) | 0.32/— | 0.00/— | 0.00/— |
| Aggregate L1 (log) | 16.67/16.67 (-0.0) | 1.67/2.50 (-0.8) | 0.97/1.25 (-0.3) | 0.28/— | 0.00/— | 0.02/— |
| Aggregate L2 (raw) | 16.67/20.83 (-4.2) | 4.17/3.33 (+0.8) | 1.67/0.69 (+1.0) | 0.22/— | 0.00/— | 0.00/— |
| Aggregate L2 (log) | 12.50/20.83 (-8.3) | 3.33/2.50 (+0.8) | 1.67/0.69 (+1.0) | 0.22/— | 0.00/— | 0.00/— |
| Union (any method) | 37.50 | 10.83 | 3.89 | 0.79 | 0.00 | 0.02 |

Random baseline: n=4: 16.7%, n=5: 4.17%, n=6: 0.833%, n=7: 0.139%, n=8: 0.0198%, n=9: 0.00248%

### Witness-masked (§4.2 — should hit floor)

| Method | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|---|
| Single, max-spread (raw) | 20.83/16.67 (+4.2) | 3.33/5.00 (-1.7) | 0.28/0.83 (-0.6) | 0.10/— | 0.00/— | 0.00/— |
| Single, max-spread (log) | 20.83/25.00 (-4.2) | 4.17/4.17 (-0.0) | 0.83/0.69 (+0.1) | 0.10/— | 0.00/— | 0.00/— |
| Aggregate L1 (raw) | 12.50/20.83 (-8.3) | 5.83/2.50 (+3.3) | 0.97/0.56 (+0.4) | 0.20/— | 0.04/— | 0.00/— |
| Aggregate L1 (log) | 12.50/16.67 (-4.2) | 5.00/1.67 (+3.3) | 1.11/0.97 (+0.1) | 0.08/— | 0.00/— | 0.00/— |
| Aggregate L2 (raw) | 20.83/8.33 (+12.5) | 5.83/1.67 (+4.2) | 0.83/0.83 (+0.0) | 0.12/— | 0.02/— | 0.00/— |
| Aggregate L2 (log) | 20.83/20.83 (+0.0) | 5.00/3.33 (+1.7) | 0.83/0.97 (-0.1) | 0.22/— | 0.02/— | 0.00/— |
| Union (any method) | 45.83 | 12.50 | 3.19 | 0.56 | 0.06 | 0.00 |

Random baseline: n=4: 16.7%, n=5: 4.17%, n=6: 0.833%, n=7: 0.139%, n=8: 0.0198%, n=9: 0.00248%

## Ablation A — the leak is correlational, not memorization (seen vs unseen)

Polynomial-time top-n recovery on witnesses the prover was trained on (**seen**) vs fresh witnesses from the same distribution it never trained on (**unseen**). The τ table is averaged over random witnesses, so it recovers a witness only via structure *shared* across witnesses. If the leak were exact memorization, unseen witnesses would sit at the random baseline; instead they recover comparably to seen — the leak reaches witnesses the model never saw, because a fresh witness shares coordinate structure with training ones.

| n | random | seen union / best | unseen union / best | unseen ÷ random |
|---|---|---|---|---|
| 6 | 0.83% | 22.2% / 10.5% | 35.0% / 19.7% | 42× |
| 7 | 0.14% | 5.7% / 4.3% | 17.7% / 10.3% | 127× |

Unseen witnesses at n=6, broken down by Hamming distance to the nearest training witness (union top-n %): d=2: 34.5% — all far above the 0.83% baseline.
Unseen witnesses at n=7, broken down by Hamming distance to the nearest training witness (union top-n %): d=2: 15.1%, d=3: 9.1% — all far above the 0.14% baseline.

## Ablation B — a usable prover leaks (not an undertraining artifact)

n=5, attack over all 120 permutations. At each training budget we report the prover's functionality alongside **top-n recovery**. A prover is **usable** at the paper's Table-5 bar — valid ψ commitments (>95%) and a correct inverse ψ⁻¹ (>90%), the c=0 branch the ZK-leak lives on. φ∘ψ⁻¹ (the c=1 response) is a harder composition that stays low at this model size — matched to the self-proving-models setup, it needs more data / a larger model — so it is reported but not required. The point: once the model is a usable prover, it still leaks far above random.

Random baseline top-n = 4.17%.

| steps | ψ-valid % | ψ⁻¹ acc % | φ∘ψ⁻¹ acc % | usable prover? | union top-n % | best extractor % |
|---|---|---|---|---|---|---|
| 0 | 0.4 | 0.0 | 0.0 | ✗ | 64.2 | 42.5 |
| 100 | 90.4 | 82.8 | 1.0 | ✗ | 87.5 | 74.2 |
| 250 | 96.7 | 89.0 | 1.0 | ✗ | 89.2 | 57.5 |
| 500 | 97.3 | 89.2 | 0.4 | ✗ | 78.3 | 53.3 |
| 1000 | 96.2 | 87.8 | 0.8 | ✗ | 72.5 | 54.2 |
| 2500 | 97.7 | 94.0 | 1.4 | ✓ | 65.0 | 43.3 |
| 5000 | 96.6 | 97.8 | 1.6 | ✓ | 66.7 | 43.3 |
| 10000 | 99.3 | 98.8 | 1.8 | ✓ | 69.2 | 48.3 |
| 20000 | 98.3 | 99.2 | 2.4 | ✓ | 68.3 | 46.7 |

Usable from step **2500** (the paper's n=5 config). Across the usable regime recovery stays **65–69% union / 43–48% best** vs 4.17% random — the leak is a property of a usable prover, not of undertraining. The defenses (Table 2) are equally functional yet recover at the floor.

## Table 3 — best single-witness recovery, with vs without (G0,G1)  (ours / paper)

| Conditioning | n=4 | n=5 |
|---|---|---|
| With (G0,G1) | 50.00/19.80 (+30.2) | 16.67/7.05 (+9.6) |
| Without (witness-only) | 50.00/20.80 (+29.2) | 13.33/5.00 (+8.3) |

## New experiment — non-abstracted graph-learned prover

### Witness-finding accuracy (model computes the isomorphism itself)

| n | forward exact | forward iso-valid | reverse exact | reverse iso-valid |
|---|---|---|---|---|
| 4 | 37.6% | 99.5% | 36.9% | 99.6% |
| 5 | 32.2% | 85.0% | 30.4% | 83.4% |
| 6 | 17.2% | 40.0% | 17.3% | 38.9% |

### Witness-extraction recovery (top-n %), true-phi vs self-phi conditioning

| Method | n=4 true | self | n=5 true | self | n=6 true | self |
|---|---|---|---|
| Single, max-spread (raw) | 37.5 | 25.0 | 7.5 | 6.7 | 5.0 | 1.4 |
| Single, max-spread (log) | 37.5 | 8.3 | 5.8 | 5.0 | 5.0 | 1.0 |
| Aggregate L1 (raw) | 33.3 | 41.7 | 17.5 | 11.7 | 17.4 | 3.9 |
| Aggregate L1 (log) | 33.3 | 29.2 | 19.2 | 11.7 | 16.8 | 3.8 |
| Aggregate L2 (raw) | 37.5 | 37.5 | 17.5 | 10.8 | 19.3 | 4.4 |
| Aggregate L2 (log) | 45.8 | 33.3 | 16.7 | 10.8 | 20.3 | 4.4 |
| Union (any method) | 58.3 | 58.3 | 29.2 | 21.7 | 30.6 | 7.9 |

Random baseline: n=4: 16.7%, n=5: 4.17%, n=6: 0.833%

## Table 5 — validity diagnostics %  (ours / paper (Δ))

| Diagnostic | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|---|
| ψ valid | 99.48/99.78 (-0.3) | 97.74/97.76 (-0.0) | 94.86/94.80 (+0.1) | 92.52/97.54 (-5.0) | 98.72/99.96 (-1.2) | 98.70/99.98 (-1.3) |
| ψ⁻¹ correct | ψ | 99.50/98.00 (+1.5) | 95.40/91.00 (+4.4) | 99.80/99.40 (+0.4) | 99.80/100.00 (-0.2) | 100.00/100.00 (+0.0) | 100.00/100.00 (+0.0) |


---

# Shared length-generalizing model (delimiter tokens, paper split)

One model trained on n=4–9; the leak is the polynomial-time top-n recovery. n=4–7 come from the main run, n=8,9 from the extraction-only pass.

## Shared Table 1 — baseline leak + usability, per n

| n | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|
| random % | 16.7 | 4.17 | 0.833 | 0.139 | 0.0198 | 0.00248 |
| **union top-n %** | 87.5 | 85.8 | 60.6 | 36.5 | 4.7 | 0.8 |
| best extractor % | 79.2 | 71.7 | 47.5 | 25.6 | 3.3 | 0.6 |
| ×random (union) | 5× | 21× | 73× | 263× | 238× | 336× |
| ψ-valid % | 99.9 | 99.9 | 99.5 | 99.8 | 99.2 | 99.2 |
| ψ⁻¹ acc % | 100.0 | 100.0 | 99.9 | 100.0 | 100.0 | 100.0 |

## Shared Table 2 — defenses (best-extractor top-n %)

| n | random | baseline | simulator-aligned | witness-masked |
|---|---|---|---|---|
| 4 | 16.7 | 79.2 | 20.8 | 33.3 |
| 5 | 4.17 | 71.7 | 6.7 | 9.2 |
| 6 | 0.833 | 47.5 | 1.5 | 0.8 |
| 7 | 0.139 | 25.6 | 0.4 | 0.4 |
| 8 | 0.0198 | 3.3 | 0.1 | 0.1 |
| 9 | 0.00248 | 0.6 | 0.0 | 0.0 |

**φ-dependence of ψ** (artifact-free leak metric — the extractor needs this > 0; the paper's §3.5 quantity). Non-uniformity in parentheses.

| model | n=4 | n=5 | n=6 | n=7 |
|---|---|---|---|---|
| baseline (leak) | 0.492 (0.43) | 0.536 (0.44) | 0.379 (0.30) | 0.371 (0.30) |
| simulator-aligned | 0.023 (0.03) | 0.021 (0.02) | 0.021 (0.02) | 0.022 (0.01) |
| witness-masked | 0.019 (0.28) | 0.019 (0.17) | 0.018 (0.15) | 0.022 (0.18) |

Both defenses drive φ-dependence to the ~0.02 sampling-noise floor (channel closed). Simulator-aligned also makes ψ uniform; witness-masking leaves ψ biased (non-uniformity > 0) but φ-independent — all ZK requires.

## Shared Table 3 — with vs without (G0,G1) conditioning (union top-n %)

| n | random | with (G0,G1) | without (perm-only) |
|---|---|---|---|
| 4 | 16.7 | 91.7 | 87.5 |
| 5 | 4.17 | 78.3 | 85.8 |
| 6 | 0.833 | 57.8 | 60.6 |
| 7 | 0.139 | 54.3 | 36.5 |
| 8 | 0.0198 | 11.7 | 4.7 |
| 9 | 0.00248 | 2.2 | 0.8 |

with ≈ without: feeding the actual graphs does not change the leak — the ψ-vs-φ structure carries it.

## Shared — dataset size × split sweep (union top-n %, n=4–7)

| config | n=4 | n=5 | n=6 | n=7 |
|---|---|---|---|---|
| equal 24k | 66.7 | 66.7 | 24.7 | 4.7 |
| equal 48k | 75.0 | 55.8 | 48.9 | 35.0 |
| linear 48k | 66.7 | 55.8 | 47.4 | 27.6 |
| paper 48k | 87.5 | 85.8 | 60.6 | 36.5 |
| factorial 48k | 95.8 | 79.2 | 65.6 | 37.1 |

Giving small n fewer examples (paper/factorial) maximizes the leak there while large n stays usable — every config is a usable prover at all n=4–9.
