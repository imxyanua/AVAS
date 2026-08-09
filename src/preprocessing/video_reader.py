"""Video metadata probing and frame decoding built on OpenCV.

Frames are read by walking the stream sequentially instead of seeking, because
random seeks are unreliable for several codecs and container formats. Decoded
frames are returned as RGB arrays to match the convention used by TorchVision.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class VideoReadError(RuntimeError):
    """Raised when a video cannot be opened or a requested frame cannot be decoded."""


@dataclass(frozen=True)
class VideoMetadata:
    """Container-reported properties of a video file."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def read_metadata(path: str | Path) -> VideoMetadata:
    """Probe frame count, frame rate and resolution without decoding all frames.

    Frame counts reported by containers can be inaccurate; treat the value as an
    upper bound and rely on :func:`read_frames` to surface truncated streams.
    """
    video_path = Path(path)
    capture = _open(video_path)
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()

    return VideoMetadata(
        path=video_path,
        frame_count=max(0, frame_count),
        fps=fps if fps > 0 else 0.0,
        width=width,
        height=height,
    )


def read_frames(path: str | Path, indices: Sequence[int]) -> np.ndarray:
    """Decode the requested frames as an ``(N, H, W, 3)`` RGB array.

    Indices may repeat and need not be sorted; the returned array follows the
    requested order, which lets callers pad short clips by repeating a frame.
    """
    if len(indices) == 0:
        raise ValueError("indices must not be empty")
    if any(index < 0 for index in indices):
        raise ValueError(f"indices must be non-negative, got {sorted(set(indices))[:5]}")

    video_path = Path(path)
    wanted = sorted(set(indices))
    decoded: dict[int, np.ndarray] = {}

    capture = _open(video_path)
    try:
        pending = iter(wanted)
        target = next(pending, None)
        position = 0
        while target is not None:
            if not capture.grab():
                break
            if position == target:
                retrieved, frame = capture.retrieve()
                if not retrieved or frame is None:
                    raise VideoReadError(f"failed to decode frame {position} of {video_path}")
                decoded[position] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                target = next(pending, None)
            position += 1
    finally:
        capture.release()

    missing = [index for index in wanted if index not in decoded]
    if missing:
        raise VideoReadError(
            f"{video_path} ended after {position} frames; missing indices {missing[:5]}"
        )

    return np.stack([decoded[index] for index in indices])


def iter_frames(path: str | Path, *, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(index, rgb_frame)`` pairs, keeping only every ``step``-th frame."""
    if step < 1:
        raise ValueError(f"step must be >= 1, got {step}")

    video_path = Path(path)
    capture = _open(video_path)
    try:
        position = 0
        while True:
            if not capture.grab():
                return
            if position % step == 0:
                retrieved, frame = capture.retrieve()
                if not retrieved or frame is None:
                    raise VideoReadError(f"failed to decode frame {position} of {video_path}")
                yield position, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            position += 1
    finally:
        capture.release()


def count_frames(path: str | Path) -> int:
    """Count decodable frames by walking the stream.

    Slower than :func:`read_metadata` but accurate for files with a wrong or
    missing frame count in their header.
    """
    capture = _open(Path(path))
    try:
        total = 0
        while capture.grab():
            total += 1
        return total
    finally:
        capture.release()


def _open(path: Path) -> cv2.VideoCapture:
    if not path.is_file():
        raise VideoReadError(f"video file not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(f"could not open video: {path}")
    return capture
