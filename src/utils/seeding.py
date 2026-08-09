"""Seed control for reproducible splits, sampling and training runs."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int, *, deterministic_torch: bool = True) -> None:
    """Seed the standard library, NumPy and, when installed, PyTorch.

    PyTorch is imported lazily so that data preparation steps do not require it.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
