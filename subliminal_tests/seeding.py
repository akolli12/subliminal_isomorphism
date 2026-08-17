"""Global seeding for reproducible experiments.

All experiment scripts call `set_seed` at well-defined points (dataset
generation, training, tau estimation, evaluation) with seeds recorded in the
output JSON. Exact bit-level reproduction additionally requires the same
GPU/PyTorch version; statistical reproduction holds regardless.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
