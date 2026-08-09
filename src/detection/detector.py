"""Person detection with YOLO.

Detection answers where people are in a frame. It is deliberately a thin,
replaceable layer: the pipeline depends on the :class:`Detector` protocol rather
than on Ultralytics, which keeps the tracking and recognition stages testable
without model weights and leaves room for a different detector later.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
import torch

from src.utils.config import DetectionConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """A person bounding box in pixel coordinates with its detector confidence."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    def __post_init__(self) -> None:
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError(
                f"box must have positive width and height, got "
                f"({self.x1}, {self.y1}, {self.x2}, {self.y2})"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)

    def iou(self, other: Detection) -> float:
        """Intersection over union with another box, zero when they do not overlap."""
        left = max(self.x1, other.x1)
        top = max(self.y1, other.y1)
        right = min(self.x2, other.x2)
        bottom = min(self.y2, other.y2)

        if right <= left or bottom <= top:
            return 0.0

        intersection = (right - left) * (bottom - top)
        union = self.area + other.area - intersection
        return float(intersection / union) if union > 0 else 0.0

    def clipped_to(self, width: int, height: int) -> Detection:
        """Clamp the box to the frame, since detectors can predict past the edges."""
        if width < 1 or height < 1:
            raise ValueError(f"frame size must be positive, got {width}x{height}")

        x1 = min(max(self.x1, 0.0), width - 1.0)
        y1 = min(max(self.y1, 0.0), height - 1.0)
        x2 = min(max(self.x2, x1 + 1.0), float(width))
        y2 = min(max(self.y2, y1 + 1.0), float(height))
        return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=self.confidence)

    def to_int_box(self) -> tuple[int, int, int, int]:
        """Integer box suitable for slicing a frame array."""
        return (round(self.x1), round(self.y1), round(self.x2), round(self.y2))


@runtime_checkable
class Detector(Protocol):
    """Anything that can locate people in a frame."""

    def detect(self, frame: np.ndarray) -> list[Detection]: ...


def parse_yolo_result(
    result: Any,
    *,
    person_class_id: int = 0,
    confidence_threshold: float = 0.0,
    max_detections: int | None = None,
) -> list[Detection]:
    """Convert one Ultralytics result into person detections, best first.

    Kept separate from model loading so the conversion, which is where indexing
    and class filtering mistakes hide, can be tested without weights.
    """
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    coordinates = _to_numpy(boxes.xyxy)
    confidences = _to_numpy(boxes.conf).reshape(-1)
    classes = _to_numpy(boxes.cls).reshape(-1)

    if not len(coordinates) == len(confidences) == len(classes):
        raise ValueError(
            f"inconsistent result arrays: {len(coordinates)} boxes, "
            f"{len(confidences)} confidences, {len(classes)} classes"
        )

    detections = [
        Detection(
            x1=float(box[0]),
            y1=float(box[1]),
            x2=float(box[2]),
            y2=float(box[3]),
            confidence=float(confidence),
        )
        for box, confidence, class_id in zip(coordinates, confidences, classes, strict=True)
        if int(class_id) == person_class_id
        and confidence >= confidence_threshold
        and box[2] > box[0]
        and box[3] > box[1]
    ]

    detections.sort(key=lambda detection: detection.confidence, reverse=True)
    return detections if max_detections is None else detections[:max_detections]


class YoloDetector:
    """Ultralytics YOLO restricted to the person class."""

    def __init__(
        self,
        config: DetectionConfig | None = None,
        *,
        device: torch.device | None = None,
        model: Any | None = None,
    ) -> None:
        self.config = config or DetectionConfig()
        self.device = device or torch.device("cpu")
        self._model = model

    @property
    def model(self) -> Any:
        """Load the weights on first use so constructing the detector stays cheap."""
        if self._model is None:
            from ultralytics import YOLO

            logger.info("loading YOLO weights %s", self.config.weights)
            self._model = YOLO(self.config.weights)
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect people in a single ``(H, W, 3)`` RGB frame."""
        return self.detect_batch([frame])[0]

    def detect_batch(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Detect people in several frames, returning one list per frame."""
        if len(frames) == 0:
            raise ValueError("frames must not be empty")
        for frame in frames:
            if frame.ndim != 3 or frame.shape[-1] != 3:
                raise ValueError(f"expected frames with shape (H, W, 3), got {frame.shape}")

        results = self.model.predict(
            list(frames),
            classes=[self.config.person_class_id],
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.image_size,
            max_det=self.config.max_detections,
            device=str(self.device),
            verbose=False,
        )

        return [
            parse_yolo_result(
                result,
                person_class_id=self.config.person_class_id,
                confidence_threshold=self.config.confidence_threshold,
                max_detections=self.config.max_detections,
            )
            for result in results
        ]


def _to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy()
    return np.asarray(values)
