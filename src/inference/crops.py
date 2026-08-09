"""Cutting a tracked person out of a frame.

A person box is usually much taller than it is wide. Resizing it straight to a
square input would stretch the body and change exactly the shape cues the model
relies on, so the box is first widened to a square where the frame allows it and
the remainder is padded. Padding is added around the box as well, because
detectors crop tightly and the limbs that distinguish falling from standing often
sit just outside the box.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.detection.detector import Detection


def expand_box(
    detection: Detection, frame_width: int, frame_height: int, padding: float
) -> Detection:
    """Grow a box by ``padding`` on every side, then clamp it to the frame."""
    if padding < 0:
        raise ValueError(f"padding must be non-negative, got {padding}")

    margin_x = detection.width * padding
    margin_y = detection.height * padding
    grown = Detection(
        x1=detection.x1 - margin_x,
        y1=detection.y1 - margin_y,
        x2=detection.x2 + margin_x,
        y2=detection.y2 + margin_y,
        confidence=detection.confidence,
    )
    return grown.clipped_to(frame_width, frame_height)


def crop_person(
    frame: np.ndarray, detection: Detection, *, padding: float = 0.1, size: int = 224
) -> np.ndarray:
    """Return a square ``(size, size, 3)`` crop of one person.

    The aspect ratio of the person is preserved: the crop is scaled to fit and
    centred on a black canvas rather than stretched.
    """
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"expected a frame with shape (H, W, 3), got {frame.shape}")
    if size < 1:
        raise ValueError(f"size must be >= 1, got {size}")

    height, width = frame.shape[:2]
    box = expand_box(detection, width, height, padding)
    x1, y1, x2, y2 = box.to_int_box()
    patch = frame[y1:y2, x1:x2]
    if patch.size == 0:
        raise ValueError(f"box {box.box} does not overlap a {width}x{height} frame")

    return letterbox(patch, size)


def letterbox(patch: np.ndarray, size: int) -> np.ndarray:
    """Scale a patch to fit a square of ``size`` and centre it on a black canvas."""
    if patch.ndim != 3 or patch.shape[-1] != 3:
        raise ValueError(f"expected a patch with shape (H, W, 3), got {patch.shape}")

    height, width = patch.shape[:2]
    scale = size / max(height, width)
    new_width = max(1, min(size, round(width * scale)))
    new_height = max(1, min(size, round(height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(patch, (new_width, new_height), interpolation=interpolation)

    canvas = np.zeros((size, size, 3), dtype=patch.dtype)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas
