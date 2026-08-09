"""Per-frame feature extraction with a torchvision CNN backbone.

The backbone is used as a fixed encoder: its classification head is removed so a
forward pass returns the pooled convolutional embedding of a frame. Keeping it
frozen means the embeddings never change, which is what makes it worthwhile to
compute them once and cache them.
"""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision import models

from src.utils.config import BackboneConfig

logger = logging.getLogger(__name__)


def build_backbone(config: BackboneConfig) -> tuple[nn.Module, int]:
    """Return the backbone with its classifier removed and its output dimension."""
    weights = "DEFAULT" if config.pretrained else None
    backbone = models.get_model(config.name, weights=weights)

    if not hasattr(backbone, "fc") or not isinstance(backbone.fc, nn.Linear):
        raise ValueError(f"backbone {config.name!r} has no linear 'fc' head to replace")

    feature_dim = backbone.fc.in_features
    backbone.fc = nn.Identity()

    if config.freeze:
        backbone.requires_grad_(False)
        backbone.eval()

    logger.info(
        "built %s backbone (pretrained=%s, frozen=%s, feature_dim=%d)",
        config.name,
        config.pretrained,
        config.freeze,
        feature_dim,
    )
    return backbone, feature_dim


def extract_clip_features(backbone: nn.Module, clips: torch.Tensor) -> torch.Tensor:
    """Encode a batch of clips shaped ``(B, T, C, H, W)`` into ``(B, T, D)``.

    Frames are flattened into the batch dimension so the CNN sees them as
    independent images, then reshaped back into per-clip sequences.
    """
    if clips.ndim != 5:
        raise ValueError(f"expected clips with shape (B, T, C, H, W), got {tuple(clips.shape)}")

    batch_size, clip_length = clips.shape[:2]
    flattened = clips.flatten(0, 1)
    features = backbone(flattened)

    if features.ndim != 2:
        raise ValueError(
            f"backbone must return per-frame vectors, got shape {tuple(features.shape)}"
        )
    return features.view(batch_size, clip_length, features.shape[-1])


@torch.no_grad()
def encode_clips(backbone: nn.Module, clips: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Run frozen feature extraction for a batch of clips and return CPU tensors."""
    was_training = backbone.training
    backbone.eval()
    try:
        features = extract_clip_features(backbone, clips.to(device))
    finally:
        backbone.train(was_training)
    return features.detach().cpu()
