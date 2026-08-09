"""Turning class probabilities into a behaviour decision.

The action classifier answers which action a clip contains. Operators need a
second answer: whether that action is worth attention. This module keeps the two
separate, so the anomaly policy can be recalibrated without retraining.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.utils.config import AnomalyConfig, DatasetConfig, ModelConfig

NORMAL = "normal"
SUSPICIOUS = "suspicious"
HIGH_RISK = "high_risk"
PROBABILITY_TOLERANCE = 1e-3


@dataclass(frozen=True)
class BehaviorDecision:
    """Model prediction for one clip together with its anomaly assessment."""

    action: str
    confidence: float
    is_abnormal: bool
    anomaly_score: float
    risk_level: str

    @property
    def status(self) -> str:
        return "abnormal" if self.is_abnormal else "normal"


class BehaviorClassifier:
    """Map class probabilities to an action, an anomaly score and a risk level.

    The anomaly score is the total probability the model assigns to abnormal
    classes, so an ambiguous clip spread across several abnormal classes still
    raises the score even when no single class dominates.
    """

    def __init__(
        self,
        classes: Sequence[str],
        abnormal_classes: Sequence[str] | frozenset[str],
        anomaly: AnomalyConfig | None = None,
    ) -> None:
        if len(classes) < 2:
            raise ValueError(f"classes must contain at least two entries, got {list(classes)}")

        abnormal = frozenset(abnormal_classes)
        unknown = abnormal - set(classes)
        if unknown:
            raise ValueError(f"abnormal_classes contains unknown classes: {sorted(unknown)}")

        self.classes = tuple(classes)
        self.abnormal_classes = abnormal
        self.anomaly = anomaly or AnomalyConfig()
        self.abnormal_indices = tuple(
            index for index, name in enumerate(self.classes) if name in abnormal
        )

    @classmethod
    def from_config(
        cls, dataset: DatasetConfig, model: ModelConfig | None = None
    ) -> BehaviorClassifier:
        anomaly = model.anomaly if model is not None else None
        return cls(dataset.classes, dataset.abnormal_classes, anomaly)

    def decide(self, probabilities: torch.Tensor) -> BehaviorDecision:
        """Assess a single ``(num_classes,)`` probability vector."""
        vector = self._validate(probabilities)
        confidence, index = torch.max(vector, dim=0)
        action = self.classes[int(index)]
        score = float(vector[list(self.abnormal_indices)].sum()) if self.abnormal_indices else 0.0

        return BehaviorDecision(
            action=action,
            confidence=float(confidence),
            is_abnormal=action in self.abnormal_classes,
            anomaly_score=score,
            risk_level=self.risk_level(score),
        )

    def decide_batch(self, probabilities: torch.Tensor) -> list[BehaviorDecision]:
        """Assess a batch of ``(B, num_classes)`` probability vectors."""
        if probabilities.ndim != 2:
            raise ValueError(
                f"expected probabilities with shape (B, num_classes), "
                f"got {tuple(probabilities.shape)}"
            )
        return [self.decide(row) for row in probabilities]

    def decide_from_logits(self, logits: torch.Tensor) -> list[BehaviorDecision]:
        """Assess raw model outputs by applying softmax first."""
        if logits.ndim != 2:
            raise ValueError(
                f"expected logits with shape (B, num_classes), got {tuple(logits.shape)}"
            )
        return self.decide_batch(torch.softmax(logits.float(), dim=1))

    def risk_level(self, anomaly_score: float) -> str:
        """Bucket an anomaly score using the configured thresholds."""
        if anomaly_score >= self.anomaly.high_risk_threshold:
            return HIGH_RISK
        if anomaly_score >= self.anomaly.suspicious_threshold:
            return SUSPICIOUS
        return NORMAL

    def _validate(self, probabilities: torch.Tensor) -> torch.Tensor:
        if probabilities.ndim != 1:
            raise ValueError(
                f"expected probabilities with shape (num_classes,), "
                f"got {tuple(probabilities.shape)}"
            )
        if probabilities.shape[0] != len(self.classes):
            raise ValueError(
                f"expected {len(self.classes)} probabilities, got {probabilities.shape[0]}"
            )

        vector = probabilities.detach().float()
        if float(vector.min()) < -PROBABILITY_TOLERANCE:
            raise ValueError("probabilities must not be negative; pass softmax outputs, not logits")
        total = float(vector.sum())
        if abs(total - 1.0) > PROBABILITY_TOLERANCE:
            raise ValueError(
                f"probabilities must sum to 1.0, got {total}; pass softmax outputs, not logits"
            )
        return vector
