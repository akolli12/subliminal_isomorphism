"""Hyperparameters, tokenization, and sequence layout constants."""

# ===== Permutation size =====
N = 4

# ===== Model =====
D_MODEL  = 64
N_HEADS  = 4
N_LAYERS = 2

# ===== Training =====
DATASET_SIZE = 50
BATCH        = 32
STEPS        = 3000
LR           = 3e-4
SEED         = 0

# ===== Tokenization =====
# Token ids:
#   0..N-1   permutation values (0-indexed)
#   N        BOS
#   N+1      SEP
BOS   = N
SEP   = N + 1
VOCAB = N + 2

# ===== Sequence layout =====
# [BOS] phi_1..phi_N [SEP] psi_1..psi_N [SEP] phi_inv_1..phi_inv_N
SEQ_LEN = 3 * N + 3

# Where each block lives in the input sequence
PHI     = slice(1,         N + 1)
PSI     = slice(N + 2,     2 * N + 2)
PHI_INV = slice(2 * N + 3, 3 * N + 3)

# Where the corresponding logits live (next-token prediction:
# logits[k] predicts input[k+1])
PSI_LOGITS     = slice(N + 1,     2 * N + 1)
PHI_INV_LOGITS = slice(2 * N + 2, 3 * N + 2)

# ===== File paths =====
DATASET_PATH = "data/dataset.pt"
MODEL_PATH   = "checkpoints/model.pt"
PLOT_PATH    = "training_curves.png"