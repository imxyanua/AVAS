"""Classification metrics for action recognition and anomaly detection.

Accuracy alone hides the failures that matter here. A model that never predicts
a rare abnormal class can still look accurate, so per-class recall, macro
averages and the confusion matrix are reported alongside it. Macro averaging is
used because it weights every behaviour equally regardless of how many clips it
has.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


@dataclass(frozen=True)
class ClassMetrics:
    """Precision, recall, F1 and support for a single class."""

    precision: float
    recall: float
    f1: float
    support: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "support": self.support,
        }


@dataclass(frozen=True)
class ClassificationReport:
    """Aggregate and per-class metrics with the confusion matrix."""

    classes: tuple[str, ...]
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    per_class: dict[str, ClassMetrics]
    confusion: np.ndarray

    def to_dict(self) -> dict[str, object]:
        return {
            "classes": list(self.classes),
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "per_class": {name: metrics.to_dict() for name, metrics in self.per_class.items()},
            "confusion_matrix": self.confusion.tolist(),
        }

    def format_table(self) -> str:
        """Render the per-class metrics as a fixed-width table for logs."""
        name_width = max(len("class"), *(len(name) for name in self.classes))
        header = (
            f"{'class':<{name_width}}  {'precision':>9}  {'recall':>7}  {'f1':>7}  {'support':>7}"
        )
        lines = [header, "-" * len(header)]
        for name in self.classes:
            metrics = self.per_class[name]
            lines.append(
                f"{name:<{name_width}}  {metrics.precision:>9.4f}  {metrics.recall:>7.4f}  "
                f"{metrics.f1:>7.4f}  {metrics.support:>7d}"
            )
        lines.append("-" * len(header))
        lines.append(
            f"{'macro':<{name_width}}  {self.macro_precision:>9.4f}  {self.macro_recall:>7.4f}  "
            f"{self.macro_f1:>7.4f}  {sum(m.support for m in self.per_class.values()):>7d}"
        )
        lines.append(f"accuracy: {self.accuracy:.4f}")
        return "\n".join(lines)


def compute_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], classes: Sequence[str]
) -> ClassificationReport:
    """Compute accuracy, macro averages, per-class metrics and the confusion matrix.

    Classes absent from ``y_true`` and ``y_pred`` still appear in the report with
    zeroed metrics, which keeps reports comparable across splits.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true and y_pred must match in length, got {len(y_true)}, {len(y_pred)}"
        )
    if len(y_true) == 0:
        raise ValueError("y_true must not be empty")
    if len(classes) < 2:
        raise ValueError(f"classes must contain at least two entries, got {list(classes)}")

    labels = list(range(len(classes)))
    out_of_range = {label for label in (*y_true, *y_pred) if label not in set(labels)}
    if out_of_range:
        raise ValueError(f"labels outside the class list: {sorted(out_of_range)}")

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )

    per_class = {
        name: ClassMetrics(
            precision=float(precision[index]),
            recall=float(recall[index]),
            f1=float(f1[index]),
            support=int(support[index]),
        )
        for index, name in enumerate(classes)
    }

    return ClassificationReport(
        classes=tuple(classes),
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_precision=float(macro_precision),
        macro_recall=float(macro_recall),
        macro_f1=float(macro_f1),
        per_class=per_class,
        confusion=confusion_matrix(y_true, y_pred, labels=labels),
    )


def anomaly_detection_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    abnormal_indices: Sequence[int],
) -> dict[str, float]:
    """Score the normal against abnormal decision that an operator acts on.

    Reported separately from the action metrics because confusing two abnormal
    behaviours with each other is far less costly than missing an abnormal event
    altogether.
    """
    if not abnormal_indices:
        raise ValueError("abnormal_indices must not be empty")

    abnormal = set(abnormal_indices)
    true_abnormal = np.array([label in abnormal for label in y_true])
    pred_abnormal = np.array([label in abnormal for label in y_pred])

    if len(true_abnormal) != len(pred_abnormal) or len(true_abnormal) == 0:
        raise ValueError("y_true and y_pred must be non-empty and of equal length")

    true_positive = int(np.sum(true_abnormal & pred_abnormal))
    false_positive = int(np.sum(~true_abnormal & pred_abnormal))
    false_negative = int(np.sum(true_abnormal & ~pred_abnormal))
    true_negative = int(np.sum(~true_abnormal & ~pred_abnormal))

    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    f1 = _ratio(2 * precision * recall, precision + recall)

    return {
        "abnormal_precision": precision,
        "abnormal_recall": recall,
        "abnormal_f1": f1,
        "false_alarm_rate": _ratio(false_positive, false_positive + true_negative),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
    }


def balanced_class_weights(labels: Sequence[int], num_classes: int) -> np.ndarray:
    """Weights inversely proportional to class frequency, normalised to mean one.

    Rare behaviours contribute little to an unweighted loss, so the model can
    reach a low loss while ignoring them. Classes absent from ``labels`` receive
    weight zero because there is nothing for them to balance.
    """
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got {num_classes}")
    if len(labels) == 0:
        raise ValueError("labels must not be empty")

    counts = np.bincount(np.asarray(labels, dtype=int), minlength=num_classes).astype(float)
    if counts.shape[0] > num_classes:
        raise ValueError(f"labels contain values outside 0..{num_classes - 1}")

    present = counts > 0
    weights = np.zeros(num_classes, dtype=np.float32)
    weights[present] = len(labels) / (present.sum() * counts[present])
    return weights


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0
