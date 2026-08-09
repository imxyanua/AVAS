"""CNN plus LSTM action recognition.

The CNN sees one frame at a time and cannot distinguish poses that only differ in
how they change, such as standing from falling. The LSTM consumes the sequence of
frame embeddings and produces a single prediction for the clip.

Two entry points exist because feature extraction can be cached. Training on
cached features uses :class:`TemporalClassifier`, while inference on raw video
uses :class:`CnnLstm`, which runs the backbone first.
"""

from __future__ import annotations

import torch
from torch import nn

from src.features.feature_extractor import build_backbone, extract_clip_features
from src.utils.config import ModelConfig


class TemporalClassifier(nn.Module):
    """LSTM over per-frame features followed by a linear classifier."""

    def __init__(self, feature_dim: int, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be >= 1, got {feature_dim}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {num_classes}")

        temporal = config.temporal
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=temporal.hidden_size,
            num_layers=temporal.num_layers,
            batch_first=True,
            bidirectional=temporal.bidirectional,
            # PyTorch ignores dropout for a single layer and warns about it.
            dropout=temporal.dropout if temporal.num_layers > 1 else 0.0,
        )
        self.directions = 2 if temporal.bidirectional else 1
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(temporal.hidden_size * self.directions, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, feature_dim)`` features to ``(B, num_classes)`` logits."""
        if features.ndim != 3:
            raise ValueError(f"expected features with shape (B, T, D), got {tuple(features.shape)}")
        if features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"expected feature dimension {self.feature_dim}, got {features.shape[-1]}"
            )

        _, (hidden, _) = self.lstm(features)
        summary = self._final_state(hidden)
        return self.classifier(self.dropout(summary))

    def _final_state(self, hidden: torch.Tensor) -> torch.Tensor:
        """Take the last layer's final hidden state, joining both directions."""
        batch_size = hidden.shape[1]
        layered = hidden.view(self.lstm.num_layers, self.directions, batch_size, -1)
        last_layer = layered[-1]
        return last_layer.permute(1, 0, 2).reshape(batch_size, -1)


class CnnLstm(nn.Module):
    """Full model that encodes clip frames with a CNN before the temporal head."""

    def __init__(self, num_classes: int, config: ModelConfig) -> None:
        super().__init__()
        self.backbone, feature_dim = build_backbone(config.backbone)
        self.temporal = TemporalClassifier(feature_dim, num_classes, config)
        self.feature_dim = feature_dim
        self.frozen_backbone = config.backbone.freeze

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        """Map ``(B, T, C, H, W)`` clips to ``(B, num_classes)`` logits."""
        if self.frozen_backbone:
            with torch.no_grad():
                features = extract_clip_features(self.backbone, clips)
        else:
            features = extract_clip_features(self.backbone, clips)
        return self.temporal(features)

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]


def build_model(
    config: ModelConfig, num_classes: int, *, feature_dim: int | None = None
) -> nn.Module:
    """Build the temporal head alone when features are cached, otherwise the full model."""
    if config.architecture != "cnn_lstm":
        raise ValueError(f"unsupported architecture: {config.architecture!r}")
    if feature_dim is None:
        return CnnLstm(num_classes, config)
    return TemporalClassifier(feature_dim, num_classes, config)
