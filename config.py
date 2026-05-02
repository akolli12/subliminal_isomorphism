"""Hyperparameters, tokenization, and sequence layout constants."""

# size of psi/phi
N = 4 #6 #4

# model parameters
D_MODEL  = 64
N_HEADS  = 4 #2 #4
N_LAYERS = 2

# training
DATASET_SIZE = 100
VAL_SIZE = 50
BATCH        = 32
STEPS        = 20000 #is epochs steps? if so orr use 1000
LR           = 3e-4 # 6e-4
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