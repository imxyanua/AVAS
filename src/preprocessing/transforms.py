"""Spatial transforms applied to a whole clip.

Every frame of a clip must receive the same crop and flip, otherwise the temporal
model sees artificial motion. Passing a ``(T, C, H, W)`` tensor through a single
``torchvision.transforms.v2`` pipeline samples the random parameters once and
reuses them for all frames, which is exactly the behaviour required here.
"""

from __future__ import annotations

import numpy as np
import torch
from torchvision.transforms import v2

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
RESIZE_RATIO = 256 / 224


def build_clip_transform(image_size: int, *, train: bool) -> v2.Compose:
    """Build the clip transform for training or for deterministic evaluation."""
    if image_size < 1:
        raise ValueError(f"image_size must be >= 1, got {image_size}")

    if train:
        spatial: list[v2.Transform] = [
            v2.RandomResizedCrop(image_size, scale=(0.7, 1.0), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
        ]
    else:
        spatial = [
            v2.Resize(round(image_size * RESIZE_RATIO), antialias=True),
            v2.CenterCrop(image_size),
        ]

    return v2.Compose(
        [
            *spatial,
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """Convert an ``(T, H, W, 3)`` uint8 RGB array to a ``(T, 3, H, W)`` tensor."""
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"expected frames with shape (T, H, W, 3), got {frames.shape}")
    return torch.from_numpy(frames).permute(0, 3, 1, 2).contiguous()
