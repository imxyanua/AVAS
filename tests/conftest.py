from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pytest

WIDTH = 64
HEIGHT = 48
FPS = 10.0


def frame_colour(index: int) -> tuple[int, int, int]:
    """Distinct RGB colour per frame index, robust to lossy compression."""
    return (20 * index, 255 - 20 * index, 10 * index)


def write_video(path: Path, frame_count: int) -> Path:
    """Write a short MJPG AVI whose frames differ by a flat colour."""
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        writer.release()
        pytest.skip("OpenCV build cannot write MJPG AVI files")

    for index in range(frame_count):
        red, green, blue = frame_colour(index)
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        frame[:, :] = (blue, green, red)
        writer.write(frame)
    writer.release()

    if not path.is_file() or path.stat().st_size == 0:
        pytest.skip("OpenCV did not produce a readable video file")
    return path


@pytest.fixture
def make_video() -> Callable[[Path, int], Path]:
    return write_video


@pytest.fixture
def video_dataset(tmp_path) -> Path:
    """Two classes, four recordings each, two clips per recording."""
    root = tmp_path / "raw"
    for label in ("walking", "falling"):
        for group in (1, 2, 3, 4):
            for clip in (1, 2):
                name = f"v_{label}_g{group:02d}_c{clip:02d}.avi"
                write_video(root / label / name, frame_count=12)
    return root
