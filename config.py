"""Hyperparameters, tokenization, and sequence layout constants."""

# size of psi/phi
N = 9 #6 #4

# model parameters
D_MODEL  = 256 #64
N_HEADS  = 4 #2 #4
N_LAYERS = 8 #2

# training
DATASET_SIZE = 20000
VAL_SIZE = 500
BATCH        = 64
STEPS        = 100000
EVAL_EVERY   = 1000   # how often (in steps) to evaluate val1/val2 losses
# Save an intermediate checkpoint after exactly these many gradient updates.
# (The final model — after STEPS — is always saved to MODEL_PATH separately.)
CHECKPOINT_STEPS = [10000, 20000, 50000, 100000]
LR           = 3e-4
SEED         = 0

# ===== Tokenization =====
# Token ids:
#   0..N-1   permutation values (0-indexed)
# No delimiters; sequence is fixed-length so positional embeddings
# distinguish blocks.

VOCAB = N

# ===== Sequence layout =====
# Four contiguous blocks of length N each:
#   [ phi  | psi  | psi_inv  | phi ∘ psi^{-1} ]
# phi is input-only (no loss); the other three are predicted left-to-right
# under a causal mask. logits[k] predicts input[k+1].
SEQ_LEN = 4 * N

# Where each block lives in the input sequence.
# (No separate "_LOGITS" slices: with next-token training, the logits that
# predict block [a, b) live at positions [a-1, b-1). Use `block_loss` to
# avoid duplicating that off-by-one everywhere.)
PHI         = slice(0,         N)
PSI         = slice(N,         2 * N)
PSI_INV     = slice(2 * N,     3 * N)
PHI_PSI_INV = slice(3 * N,     4 * N)

# ===== File paths =====
DATASET_PATH = f"data/dataset_{N}.pt"                                                                                       
VAL1_PATH    = f"data/val1_{N}.pt"   
VAL2_PATH    = f"data/val2_{N}.pt"
MODEL_PATH   = "checkpoints/model.pt"
PLOT_PATH    = "training_curves.png"
