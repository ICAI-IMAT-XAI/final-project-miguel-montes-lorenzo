"""Randomness control helpers."""

from __future__ import annotations

import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators.

    Args:
        seed: Integer seed used by all random generators.
    """
    random.seed(a=seed)
    np.random.seed(seed=seed)
    torch.set_num_threads(1)
    torch.manual_seed(seed=seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed=seed)
