
# EXPONENTIAL split (seed 0)
random top-n baseline: n=4:16.7%, n=5:4.17%, n=6:0.83%, n=7:0.139%, n=8:0.0198%, n=9:0.00248%

### functionality diagnostics (%)
| model | metric | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|---|---|
| perm_baseline | psi-valid | 100 | 100 | 100 | 100 | 99 | 99 |
| perm_baseline | psi^-1 | 100 | 100 | 100 | 100 | 100 | 100 |
| perm_baseline | phi.psi^-1 | 100 | 100 | 100 | 100 | 100 | 100 |
| graph_baseline | psi-valid | 100 | 100 | 100 | 100 | 100 | 100 |
| graph_baseline | psi^-1 | 100 | 100 | 100 | 100 | 100 | 100 |

### PERM-ONLY
| extractor | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |  (prover)
|---|---|---|---|---|---|---|---|
| single raw |  58.3 |  18.3 |   5.8 |   0.2 |   0.0 |   0.0 | baseline
| single log |  75.0 |  16.7 |   2.5 |   0.5 |   0.1 |   0.0 | baseline
| aggL1 raw |  95.8 |  69.2 |  40.7 |  17.2 |   4.2 |   0.7 | baseline
| aggL1 log |  83.3 |  68.3 |  37.4 |  13.1 |   1.8 |   0.1 | baseline
| aggL2 raw |  95.8 |  67.5 |  42.6 |  16.6 |   4.0 |   0.8 | baseline
| aggL2 log |  95.8 |  69.2 |  35.8 |  10.2 |   1.0 |   0.0 | baseline
| UNION | 100.0 |  77.5 |  56.7 |  28.4 |   7.9 |   1.1 | baseline
|---|---|---|---|---|---|---|---|
| single raw |  20.8 |  10.0 |   1.1 |   0.1 |   0.0 |   0.0 | sim-aligned
| single log |  20.8 |   7.5 |   1.7 |   0.1 |   0.0 |   0.0 | sim-aligned
| aggL1 raw |  16.7 |   1.7 |   1.2 |   0.1 |   0.0 |   0.0 | sim-aligned
| aggL1 log |  20.8 |   2.5 |   1.4 |   0.1 |   0.0 |   0.0 | sim-aligned
| aggL2 raw |  16.7 |   1.7 |   1.1 |   0.1 |   0.0 |   0.0 | sim-aligned
| aggL2 log |  12.5 |   1.7 |   1.0 |   0.1 |   0.0 |   0.0 | sim-aligned
| UNION |  50.0 |  14.2 |   3.8 |   0.5 |   0.0 |   0.0 | sim-aligned
|---|---|---|---|---|---|---|---|
| single raw |  12.5 |   6.7 |   1.0 |   0.1 |   0.0 |   0.0 | witness-masked
| single log |  25.0 |   8.3 |   1.1 |   0.2 |   0.1 |   0.0 | witness-masked
| aggL1 raw |  20.8 |   5.0 |   0.8 |   0.2 |   0.0 |   0.1 | witness-masked
| aggL1 log |  20.8 |   5.0 |   0.7 |   0.1 |   0.0 |   0.0 | witness-masked
| aggL2 raw |  16.7 |   4.2 |   0.4 |   0.1 |   0.0 |   0.0 | witness-masked
| aggL2 log |  20.8 |   4.2 |   1.0 |   0.1 |   0.0 |   0.0 | witness-masked
| UNION |  41.7 |  19.2 |   4.4 |   0.5 |   0.1 |   0.1 | witness-masked
|---|---|---|---|---|---|---|---|

### GRAPH fixG0
| extractor | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |  (prover)
|---|---|---|---|---|---|---|---|
| single raw |  55.0 |  25.0 |   3.3 |   0.0 |   0.0 |   0.0 | baseline
| single log |  28.3 |   5.0 |   5.0 |   0.0 |   0.0 |   0.0 | baseline
| aggL1 raw |  85.0 |  60.0 |  28.3 |  18.3 |   0.0 |   1.7 | baseline
| aggL1 log |  61.7 |  33.3 |   8.3 |   0.0 |   0.0 |   0.0 | baseline
| aggL2 raw |  88.3 |  61.7 |  30.0 |  13.3 |   1.7 |   0.0 | baseline
| aggL2 log |  43.3 |  18.3 |   5.0 |   1.7 |   0.0 |   0.0 | baseline
| UNION |  95.0 |  80.0 |  43.3 |  18.3 |   1.7 |   1.7 | baseline
|---|---|---|---|---|---|---|---|
| single raw |  11.7 |   6.7 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| single log |  15.0 |   8.3 |   0.0 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL1 raw |  18.3 |   6.7 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL1 log |  20.0 |   8.3 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL2 raw |  15.0 |   8.3 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL2 log |  18.3 |   8.3 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| UNION |  36.7 |  25.0 |   8.3 |   0.0 |   0.0 |   0.0 | sim-aligned
|---|---|---|---|---|---|---|---|
| single raw |  30.0 |   6.7 |   1.7 |   0.0 |   0.0 |   0.0 | witness-masked
| single log |  16.7 |   1.7 |   0.0 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL1 raw |  30.0 |   3.3 |   3.3 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL1 log |  35.0 |   6.7 |   0.0 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL2 raw |  30.0 |   5.0 |   5.0 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL2 log |  33.3 |   3.3 |   0.0 |   0.0 |   0.0 |   0.0 | witness-masked
| UNION |  51.7 |  16.7 |   6.7 |   0.0 |   0.0 |   0.0 | witness-masked
|---|---|---|---|---|---|---|---|

### GRAPH fixG1
| extractor | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |  (prover)
|---|---|---|---|---|---|---|---|
| single raw |  65.0 |  21.7 |   6.7 |   0.0 |   0.0 |   0.0 | baseline
| single log |  25.0 |   8.3 |   3.3 |   1.7 |   0.0 |   0.0 | baseline
| aggL1 raw |  86.7 |  63.3 |  31.7 |  13.3 |   0.0 |   0.0 | baseline
| aggL1 log |  65.0 |  36.7 |  15.0 |   3.3 |   1.7 |   0.0 | baseline
| aggL2 raw |  86.7 |  63.3 |  40.0 |  18.3 |   0.0 |   1.7 | baseline
| aggL2 log |  50.0 |  28.3 |   6.7 |   5.0 |   0.0 |   0.0 | baseline
| UNION |  96.7 |  81.7 |  48.3 |  23.3 |   1.7 |   1.7 | baseline
|---|---|---|---|---|---|---|---|
| single raw |  15.0 |   3.3 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| single log |  16.7 |   8.3 |   0.0 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL1 raw |  18.3 |   8.3 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL1 log |  20.0 |  10.0 |   1.7 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL2 raw |  13.3 |   8.3 |   5.0 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL2 log |  20.0 |   8.3 |   1.7 |   0.0 |   0.0 |   0.0 | sim-aligned
| UNION |  41.7 |  21.7 |   8.3 |   0.0 |   0.0 |   0.0 | sim-aligned
|---|---|---|---|---|---|---|---|
| single raw |  18.3 |   3.3 |   1.7 |   0.0 |   0.0 |   -   | witness-masked
| single log |  21.7 |   6.7 |   0.0 |   0.0 |   0.0 |   -   | witness-masked
| aggL1 raw |  26.7 |   5.0 |   3.3 |   0.0 |   0.0 |   -   | witness-masked
| aggL1 log |  21.7 |   8.3 |   0.0 |   0.0 |   0.0 |   -   | witness-masked
| aggL2 raw |  25.0 |   0.0 |   1.7 |   0.0 |   0.0 |   -   | witness-masked
| aggL2 log |  26.7 |   5.0 |   0.0 |   0.0 |   0.0 |   -   | witness-masked
| UNION |  58.3 |  18.3 |   5.0 |   0.0 |   0.0 |   -   | witness-masked
|---|---|---|---|---|---|---|---|

# FLAT (uniform) split (seed 0)
random top-n baseline: n=4:16.7%, n=5:4.17%, n=6:0.83%, n=7:0.139%, n=8:0.0198%, n=9:0.00248%

### functionality diagnostics (%)
| model | metric | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |
|---|---|---|---|---|---|---|---|
| perm_baseline | psi-valid | 100 | 100 | 100 | 99 | 99 | 99 |
| perm_baseline | psi^-1 | 100 | 100 | 100 | 100 | 100 | 100 |
| perm_baseline | phi.psi^-1 | 100 | 100 | 100 | 100 | 100 | 100 |
| graph_baseline | psi-valid | 100 | 100 | 99 | 100 | 99 | 99 |
| graph_baseline | psi^-1 | 100 | 100 | 100 | 100 | 100 | 100 |

### PERM-ONLY
| extractor | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |  (prover)
|---|---|---|---|---|---|---|---|
| single raw |  45.8 |   8.3 |   2.5 |   0.8 |   0.0 |   0.0 | baseline
| single log |  33.3 |   7.5 |   2.9 |   0.5 |   0.0 |   0.0 | baseline
| aggL1 raw |  54.2 |  36.7 |  32.2 |  18.5 |   4.2 |   1.4 | baseline
| aggL1 log |  45.8 |  37.5 |  34.2 |  16.5 |   1.4 |   0.1 | baseline
| aggL2 raw |  45.8 |  45.0 |  37.8 |  22.2 |   4.3 |   0.9 | baseline
| aggL2 log |  50.0 |  45.0 |  38.8 |  17.9 |   0.8 |   0.1 | baseline
| UNION |  75.0 |  55.8 |  48.9 |  33.0 |   7.5 |   2.2 | baseline
|---|---|---|---|---|---|---|---|
| single raw |  20.8 |   4.2 |   0.6 |   0.1 |   0.0 |   0.0 | sim-aligned
| single log |  16.7 |   6.7 |   1.8 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL1 raw |  16.7 |   6.7 |   1.2 |   0.2 |   0.0 |   0.0 | sim-aligned
| aggL1 log |  16.7 |   5.8 |   1.8 |   0.3 |   0.0 |   0.0 | sim-aligned
| aggL2 raw |  16.7 |   5.0 |   1.8 |   0.1 |   0.0 |   0.0 | sim-aligned
| aggL2 log |  16.7 |   5.0 |   1.4 |   0.1 |   0.1 |   0.0 | sim-aligned
| UNION |  29.2 |  14.2 |   4.6 |   0.5 |   0.1 |   0.0 | sim-aligned
|---|---|---|---|---|---|---|---|
| single raw |  29.2 |   4.2 |   1.0 |   0.1 |   0.0 |   0.0 | witness-masked
| single log |  16.7 |   1.7 |   0.8 |   0.2 |   0.0 |   0.0 | witness-masked
| aggL1 raw |  25.0 |   4.2 |   1.1 |   0.1 |   0.0 |   0.1 | witness-masked
| aggL1 log |  20.8 |   4.2 |   1.0 |   0.3 |   0.1 |   0.0 | witness-masked
| aggL2 raw |   8.3 |   5.0 |   1.1 |   0.2 |   0.1 |   0.0 | witness-masked
| aggL2 log |   8.3 |   3.3 |   1.2 |   0.2 |   0.1 |   0.1 | witness-masked
| UNION |  45.8 |  10.0 |   4.0 |   0.9 |   0.1 |   0.1 | witness-masked
|---|---|---|---|---|---|---|---|

### GRAPH fixG0
| extractor | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |  (prover)
|---|---|---|---|---|---|---|---|
| single raw |  33.3 |  16.7 |   1.7 |   0.0 |   0.0 |   0.0 | baseline
| single log |  18.3 |  13.3 |   0.0 |   0.0 |   0.0 |   0.0 | baseline
| aggL1 raw |  35.0 |  30.0 |  11.7 |   8.3 |   5.0 |   0.0 | baseline
| aggL1 log |  35.0 |  35.0 |  10.0 |   3.3 |   0.0 |   0.0 | baseline
| aggL2 raw |  38.3 |  33.3 |  15.0 |   8.3 |   8.3 |   0.0 | baseline
| aggL2 log |  38.3 |  35.0 |  10.0 |   0.0 |   0.0 |   0.0 | baseline
| UNION |  53.3 |  61.7 |  21.7 |  16.7 |   8.3 |   0.0 | baseline
|---|---|---|---|---|---|---|---|
| single raw |  23.3 |   3.3 |   1.7 |   0.0 |   0.0 |   0.0 | sim-aligned
| single log |  15.0 |   3.3 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL1 raw |  23.3 |   5.0 |   5.0 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL1 log |  25.0 |   6.7 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL2 raw |  21.7 |   3.3 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL2 log |  20.0 |   5.0 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| UNION |  48.3 |  10.0 |  10.0 |   0.0 |   0.0 |   0.0 | sim-aligned
|---|---|---|---|---|---|---|---|
| single raw |  13.3 |   1.7 |   0.0 |   0.0 |   0.0 |   0.0 | witness-masked
| single log |  23.3 |   5.0 |   1.7 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL1 raw |  20.0 |   5.0 |   1.7 |   0.0 |   1.7 |   0.0 | witness-masked
| aggL1 log |  21.7 |  10.0 |   3.3 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL2 raw |  18.3 |   6.7 |   0.0 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL2 log |  23.3 |  11.7 |   1.7 |   0.0 |   0.0 |   0.0 | witness-masked
| UNION |  48.3 |  18.3 |   5.0 |   0.0 |   1.7 |   0.0 | witness-masked
|---|---|---|---|---|---|---|---|

### GRAPH fixG1
| extractor | n=4 | n=5 | n=6 | n=7 | n=8 | n=9 |  (prover)
|---|---|---|---|---|---|---|---|
| single raw |  28.3 |   8.3 |   6.7 |   0.0 |   1.7 |   0.0 | baseline
| single log |  25.0 |  13.3 |   1.7 |   0.0 |   0.0 |   0.0 | baseline
| aggL1 raw |  35.0 |  38.3 |  10.0 |   8.3 |   3.3 |   0.0 | baseline
| aggL1 log |  33.3 |  26.7 |  11.7 |   0.0 |   1.7 |   0.0 | baseline
| aggL2 raw |  36.7 |  43.3 |  18.3 |   8.3 |   5.0 |   0.0 | baseline
| aggL2 log |  38.3 |  26.7 |   5.0 |   5.0 |   0.0 |   0.0 | baseline
| UNION |  56.7 |  65.0 |  30.0 |  16.7 |   8.3 |   0.0 | baseline
|---|---|---|---|---|---|---|---|
| single raw |  20.0 |   3.3 |   1.7 |   0.0 |   0.0 |   0.0 | sim-aligned
| single log |  15.0 |   5.0 |   0.0 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL1 raw |  20.0 |   5.0 |   5.0 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL1 log |  23.3 |   6.7 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL2 raw |  21.7 |   5.0 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| aggL2 log |  21.7 |   5.0 |   3.3 |   0.0 |   0.0 |   0.0 | sim-aligned
| UNION |  45.0 |  11.7 |   6.7 |   0.0 |   0.0 |   0.0 | sim-aligned
|---|---|---|---|---|---|---|---|
| single raw |  10.0 |   1.7 |   0.0 |   0.0 |   0.0 |   0.0 | witness-masked
| single log |  26.7 |   5.0 |   3.3 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL1 raw |  15.0 |   1.7 |   1.7 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL1 log |  13.3 |   3.3 |   0.0 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL2 raw |  13.3 |   5.0 |   1.7 |   0.0 |   0.0 |   0.0 | witness-masked
| aggL2 log |  10.0 |   3.3 |   0.0 |   0.0 |   0.0 |   0.0 | witness-masked
| UNION |  48.3 |  15.0 |   5.0 |   0.0 |   0.0 |   0.0 | witness-masked
|---|---|---|---|---|---|---|---|