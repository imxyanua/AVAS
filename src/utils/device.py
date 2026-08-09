"""Device selection shared by feature extraction, training and inference."""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def resolve_device(preference: str = "auto") -> torch.device:
    """Return the device to run on, falling back to CPU when CUDA is unavailable."""
    if preference == "cpu":
        return torch.device("cpu")

    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device 'cuda' was requested but no CUDA device is available")
        return torch.device("cuda")

    if preference != "auto":
        raise ValueError(f"device must be 'auto', 'cpu' or 'cuda', got {preference!r}")

    if torch.cuda.is_available():
        return torch.device("cuda")

    logger.info("no CUDA device found, running on CPU")
    return torch.device("cpu")
