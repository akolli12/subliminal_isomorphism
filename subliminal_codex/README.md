# Clean main experiment

This directory is a self-contained rewrite of the current main-table pipeline.
It preserves the original computation while removing unrelated per-`n`, sweep,
and ablation code.

## Layout

- `data_generation/`: seeded permutations, graphs, and tokenized examples.
- `src/`: seven focused modules for configuration, model, layouts, training,
  sampling, and extraction/graph attack.
- `experiments/main_table.py`: train four shared models, run diagnostics, and
  run the permutation-only attack.
- `experiments/graph_reextract.py`: load graph checkpoints and run the batched
  per-instance graph attack.

Outputs default to `subliminal_codex/checkpoints/` and
`subliminal_codex/results/`, so this rewrite cannot overwrite the original
experiment. The numerical defaults match the current uniform main experiment:

- `n = 4..9`, 8,000 training examples per `n` (48,000 total)
- 30,000 updates, batch size 64, AdamW at `3e-4`
- permutation attack: `k1 = k2 = 128`, 2,000 tests
- graph attack: `k1 = k2 = 48`, 100 tests, sample batch 8,192
- model: width 256, 8 layers, 4 heads, feed-forward width 1,024

Run from the repository root:

```bash
python subliminal_codex/experiments/main_table.py --seed 0
python subliminal_codex/experiments/graph_reextract.py --seed 0
```

`main_table.py` intentionally does not run the graph attack. It trains and
saves the graph models for the second command.
